from __future__ import annotations

import argparse
import copy
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from utils.student_model import (
    StudentImageModel,
    configure_trainable_student,
    count_trainable_parameters,
    get_trainable_parameters,
)
from utils.track1_utils import (
    load_hard_negative_cache,
    load_image_embedding_cache,
    load_text_embedding_cache,
    load_track1_image_table,
)


def build_image_name_aliases(image_name: str) -> list[str]:
    aliases = [image_name]
    basename = Path(image_name).name
    if basename not in aliases:
        aliases.append(basename)

    if basename.startswith("coco_"):
        coco2014_name = "coco2014_" + basename[len("coco_"):]
        if coco2014_name not in aliases:
            aliases.append(coco2014_name)
    if basename.startswith("coco2014_"):
        coco_name = "coco_" + basename[len("coco2014_"):]
        if coco_name not in aliases:
            aliases.append(coco_name)

    return aliases


class Track1DistillDataset(Dataset):
    def __init__(
        self,
        *,
        img_list_path: str,
        image_folder: str,
        preprocess,
        text_ids: torch.Tensor | None,
        text_embeddings: torch.Tensor | None,
        teacher_image_names: list[str],
        teacher_embeddings: torch.Tensor,
        hard_negative_lookup: dict[int, list[int]] | None,
        num_hard_negatives: int,
        strict_teacher_coverage: bool,
    ):
        self.preprocess = preprocess
        self.text_embeddings = text_embeddings
        self.text_index = (
            {int(text_id): index for index, text_id in enumerate(text_ids.tolist())}
            if text_ids is not None and text_embeddings is not None
            else {}
        )
        self.text_supervision_enabled = text_embeddings is not None and len(self.text_index) > 0
        self.teacher_index: dict[str, torch.Tensor] = {}
        for index, image_name in enumerate(teacher_image_names):
            teacher_embedding = teacher_embeddings[index]
            for alias in build_image_name_aliases(image_name):
                self.teacher_index.setdefault(alias, teacher_embedding)
        self.hard_negative_lookup = hard_negative_lookup or {}
        self.num_hard_negatives = max(0, num_hard_negatives)
        self.teacher_embedding_dim = int(teacher_embeddings.shape[1])
        self.text_embedding_dim = int(text_embeddings.shape[1]) if text_embeddings is not None else None
        self.missing_teacher_images: list[str] = []
        self.missing_positive_images: list[str] = []
        self.skipped_missing_teacher_count = 0
        self.skipped_missing_positive_count = 0

        image_table = load_track1_image_table(img_list_path, image_folder)
        self.total_rows = int(len(image_table))
        self.samples: list[dict[str, object]] = []

        for row in tqdm(
            image_table.itertuples(index=False),
            total=len(image_table),
            desc="Preparing training samples",
            unit="image",
            dynamic_ncols=True,
        ):
            teacher_embedding = self.teacher_index.get(row.image_name)
            if teacher_embedding is None:
                self.missing_teacher_images.append(row.image_name)
                self.skipped_missing_teacher_count += 1
                if strict_teacher_coverage:
                    raise ValueError(f"Missing teacher embedding for image: {row.image_name}")
                continue

            positive_indices: list[int] = []
            hard_negative_ids: list[int] = []
            if self.text_supervision_enabled:
                positive_ids = [text_id for text_id in row.positive_text_ids if text_id in self.text_index]
                if not positive_ids:
                    self.skipped_missing_positive_count += 1
                    self.missing_positive_images.append(row.image_name)
                    continue

                positive_indices = [self.text_index[text_id] for text_id in positive_ids]

                seen_ids = set(positive_ids)
                for text_id in positive_ids:
                    for candidate_id in self.hard_negative_lookup.get(text_id, []):
                        if candidate_id in seen_ids or candidate_id not in self.text_index:
                            continue
                        seen_ids.add(candidate_id)
                        hard_negative_ids.append(candidate_id)
                        if len(hard_negative_ids) >= self.num_hard_negatives:
                            break
                    if len(hard_negative_ids) >= self.num_hard_negatives:
                        break

            self.samples.append(
                {
                    "image_path": row.image_path,
                    "positive_indices": positive_indices,
                    "teacher_embedding": teacher_embedding,
                    "hard_negative_ids": hard_negative_ids,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        with Image.open(sample["image_path"]) as image:
            pixel_values = self.preprocess(image.convert("RGB"))

        positive_dtype = self.text_embeddings.dtype if self.text_embeddings is not None else torch.float32
        positive_dim = self.text_embedding_dim or self.teacher_embedding_dim
        positive_embedding = torch.zeros(positive_dim, dtype=positive_dtype)
        if self.text_supervision_enabled:
            positive_indices = sample["positive_indices"]
            selected_positive_index = positive_indices[torch.randint(len(positive_indices), (1,)).item()]
            positive_embedding = self.text_embeddings[selected_positive_index]

        negative_dim = self.text_embedding_dim or self.teacher_embedding_dim
        negative_embeddings = torch.zeros(
            self.num_hard_negatives,
            negative_dim,
            dtype=positive_dtype,
        )
        negative_mask = torch.zeros(self.num_hard_negatives, dtype=torch.bool)

        if self.text_supervision_enabled:
            negative_ids = sample["hard_negative_ids"]
            for negative_index, text_id in enumerate(negative_ids):
                negative_embeddings[negative_index] = self.text_embeddings[self.text_index[text_id]]
                negative_mask[negative_index] = True

        return {
            "pixel_values": pixel_values,
            "positive_embedding": positive_embedding,
            "teacher_embedding": sample["teacher_embedding"],
            "negative_embeddings": negative_embeddings,
            "negative_mask": negative_mask,
            "sample_index": torch.tensor(index, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a distilled student image model for Track1.")
    parser.add_argument("--img-list", default="dataset/img_list.csv")
    parser.add_argument("--image-folder", default="dataset/images")
    parser.add_argument(
        "--text-embeddings",
        help="Optional text embedding cache. Required only when enabling text-based losses.",
    )
    parser.add_argument("--teacher-embeddings", default="artifacts/teacher_image_embeddings.npz")
    parser.add_argument("--hard-negatives", help="Optional hard-negative NPZ file.")
    parser.add_argument(
        "--student-model-id",
        default="hf-hub:laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K",
    )
    parser.add_argument("--output-dir", default="artifacts/student_distill_v5")
    parser.add_argument("--resume-checkpoint", help="Optional checkpoint path to continue training from.")
    parser.add_argument(
        "--resume-optimizer-state",
        action="store_true",
        help="Restore optimizer/scaler states when available in the resume checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--contrastive-weight", type=float, default=0.0)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument(
        "--distill-loss-type",
        choices=["mse", "cosine", "kl"],
        default="cosine",
        help="Distillation objective between student and teacher embeddings.",
    )
    parser.add_argument(
        "--distill-temperature",
        type=float,
        default=4.0,
        help="Temperature used for KL distillation.",
    )
    parser.add_argument("--baseline-anchor-weight", type=float, default=0.1)
    parser.add_argument(
        "--baseline-anchor-final-weight",
        type=float,
        default=0.0,
        help="Anchor weight at final epoch; weight is linearly decayed from baseline-anchor-weight.",
    )
    parser.add_argument("--hard-negative-weight", type=float, default=0.0)
    parser.add_argument("--hard-negative-margin", type=float, default=0.05)
    parser.add_argument("--num-hard-negatives", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--unfreeze-last-n-blocks", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Gradient clipping max norm. Set <= 0 to disable.",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "cosine"],
        default="cosine",
        help="Learning-rate scheduler type.",
    )
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-7,
        help="Minimum learning rate reached by cosine scheduler.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.05,
        help="Fraction of samples held out for validation metrics (0 disables).",
    )
    parser.add_argument(
        "--val-every-epochs",
        type=int,
        default=1,
        help="Evaluate validation metrics every N epochs.",
    )
    parser.add_argument(
        "--best-checkpoint-metric",
        choices=["train-loss", "val-loss", "val-recall@10"],
        default="val-loss",
        help="Metric used to choose best_loss_checkpoint.pt.",
    )
    parser.add_argument(
        "--val-recall-image-chunk-size",
        type=int,
        default=256,
        help="Validation Recall@10 image chunk size.",
    )
    parser.add_argument(
        "--val-recall-text-chunk-size",
        type=int,
        default=8192,
        help="Validation Recall@10 text chunk size.",
    )
    parser.add_argument(
        "--val-recall-max-texts",
        type=int,
        default=0,
        help="Cap candidate texts used by Validation Recall@10 (0 means all).",
    )
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--save-epoch-checkpoints", action="store_true")
    parser.add_argument("--seed", type=int, default=696)
    parser.add_argument(
        "--strict-teacher-coverage",
        action="store_true",
        help="Fail if any training image is missing in the teacher embedding cache.",
    )
    parser.add_argument(
        "--missing-teacher-log",
        help="Optional file to save image names skipped due to missing teacher embeddings.",
    )
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_hard_negative_lookup(cache_path: str | None) -> dict[int, list[int]] | None:
    if not cache_path:
        return None
    cache = load_hard_negative_cache(cache_path)
    text_ids = cache["text_ids"].tolist()
    neighbor_text_ids = cache["neighbor_text_ids"].tolist()
    return {
        int(text_id): [int(candidate_id) for candidate_id in candidate_ids]
        for text_id, candidate_ids in zip(text_ids, neighbor_text_ids)
    }


def resolve_num_workers(requested_num_workers: int | None, device: str) -> int:
    if requested_num_workers is not None:
        return max(0, requested_num_workers)
    if not device.startswith("cuda"):
        return 0
    cpu_count = os.cpu_count() or 1
    return min(8, max(2, cpu_count // 2))


def configure_cuda_runtime(*, device: str, allow_tf32: bool) -> None:
    if not device.startswith("cuda"):
        return
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = allow_tf32


def build_train_val_subsets(
    dataset: Dataset,
    *,
    val_split: float,
    seed: int,
) -> tuple[Dataset, Dataset | None]:
    total_size = len(dataset)
    if total_size <= 1 or val_split <= 0:
        return dataset, None

    val_size = max(1, min(total_size - 1, int(total_size * val_split)))
    if val_size <= 0:
        return dataset, None

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(total_size, generator=generator).tolist()
    val_indices = permutation[:val_size]
    train_indices = permutation[val_size:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_scale: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    bounded_total_steps = max(1, int(total_steps))
    bounded_warmup_steps = max(0, min(int(warmup_steps), bounded_total_steps - 1))
    clamped_min_lr_scale = min(1.0, max(0.0, float(min_lr_scale)))

    def lr_lambda(step: int) -> float:
        if bounded_warmup_steps > 0 and step < bounded_warmup_steps:
            return max(1e-8, float(step + 1) / float(bounded_warmup_steps))
        if bounded_total_steps <= bounded_warmup_steps:
            return clamped_min_lr_scale

        progress = (step - bounded_warmup_steps) / float(bounded_total_steps - bounded_warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return clamped_min_lr_scale + (1.0 - clamped_min_lr_scale) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def interpolate_weight(
    *,
    start: float,
    end: float,
    step: int,
    total_steps: int,
) -> float:
    if total_steps <= 1:
        return float(end)
    progress = min(1.0, max(0.0, step / float(total_steps - 1)))
    return float(start + (end - start) * progress)


def resolve_resume_checkpoint_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate

    if candidate.parent == Path("."):
        matches = sorted(Path(".").glob(f"**/{candidate.name}"))
        if len(matches) == 1:
            print(f"Resolved resume checkpoint '{raw_path}' -> '{matches[0]}'")
            return matches[0]
        if len(matches) > 1:
            display = "\n".join(str(path) for path in matches[:10])
            raise FileNotFoundError(
                f"Resume checkpoint '{raw_path}' is ambiguous. Use one of:\n{display}"
            )

    raise FileNotFoundError(f"Resume checkpoint does not exist: {candidate}")


def evaluate_val_recall_at_k(
    *,
    image_embeddings: torch.Tensor,
    sample_indices: torch.Tensor,
    dataset: Track1DistillDataset,
    text_embeddings: torch.Tensor,
    top_k: int,
    image_chunk_size: int,
    text_chunk_size: int,
    max_texts: int,
    device: str,
) -> float | None:
    if image_embeddings.numel() == 0:
        return None

    total_texts = int(text_embeddings.shape[0])
    if total_texts <= 0:
        return None
    candidate_texts = min(total_texts, max_texts) if max_texts > 0 else total_texts
    if candidate_texts <= 0:
        return None

    bounded_top_k = min(max(1, int(top_k)), candidate_texts)
    normalized_images = F.normalize(image_embeddings.float(), dim=-1)

    total_eval = 0
    total_hits = 0
    for image_start in range(0, normalized_images.shape[0], max(1, image_chunk_size)):
        image_end = min(image_start + max(1, image_chunk_size), normalized_images.shape[0])
        image_chunk = normalized_images[image_start:image_end].to(device, non_blocking=True)
        batch_size = image_chunk.shape[0]

        best_scores = torch.full(
            (batch_size, bounded_top_k),
            fill_value=-float("inf"),
            device=device,
            dtype=torch.float32,
        )
        best_indices = torch.full(
            (batch_size, bounded_top_k),
            fill_value=-1,
            device=device,
            dtype=torch.long,
        )

        for text_start in range(0, candidate_texts, max(1, text_chunk_size)):
            text_end = min(text_start + max(1, text_chunk_size), candidate_texts)
            text_chunk = F.normalize(
                text_embeddings[text_start:text_end].to(device, non_blocking=True),
                dim=-1,
            )
            similarity = image_chunk @ text_chunk.T
            local_top_k = min(bounded_top_k, similarity.shape[1])
            if local_top_k <= 0:
                continue

            chunk_scores, chunk_indices = torch.topk(similarity, k=local_top_k, dim=1)
            chunk_indices = chunk_indices + text_start

            merged_scores = torch.cat((best_scores, chunk_scores), dim=1)
            merged_indices = torch.cat((best_indices, chunk_indices), dim=1)
            top_scores, top_positions = torch.topk(merged_scores, k=bounded_top_k, dim=1)
            best_indices = torch.gather(merged_indices, dim=1, index=top_positions)
            best_scores = top_scores

        top_indices_cpu = best_indices.detach().cpu()
        for row in range(batch_size):
            sample_index = int(sample_indices[image_start + row].item())
            positive_indices = dataset.samples[sample_index]["positive_indices"]
            valid_positives = [index for index in positive_indices if index < candidate_texts]
            if not valid_positives:
                continue

            total_eval += 1
            positive_set = set(valid_positives)
            predicted = top_indices_cpu[row].tolist()
            if any((idx >= 0) and (idx in positive_set) for idx in predicted):
                total_hits += 1

    if total_eval <= 0:
        return None
    return float(total_hits) / float(total_eval)


def compute_losses(
    *,
    student_embeddings: torch.Tensor,
    reference_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor | None,
    teacher_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor | None,
    negative_mask: torch.Tensor | None,
    temperature: float,
    contrastive_weight: float,
    distill_weight: float,
    distill_loss_type: str,
    distill_temperature: float,
    baseline_anchor_weight: float,
    hard_negative_weight: float,
    hard_negative_margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    zero = torch.zeros((), device=student_embeddings.device)
    normalized_student = F.normalize(student_embeddings, dim=-1)

    contrastive_loss = zero
    if contrastive_weight > 0:
        if positive_embeddings is None:
            raise ValueError("positive_embeddings are required when contrastive_weight > 0")
        batch_size = student_embeddings.shape[0]
        labels = torch.arange(batch_size, device=student_embeddings.device)
        logits = student_embeddings @ positive_embeddings.T / temperature
        contrastive_loss = 0.5 * (
            F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.T, labels)
        )

    distill_loss = zero
    if distill_weight > 0:
        normalized_teacher = F.normalize(teacher_embeddings, dim=-1)

        if distill_loss_type == "cosine":
            if normalized_student.shape[-1] == normalized_teacher.shape[-1]:
                distill_loss = 1.0 - F.cosine_similarity(
                    normalized_student,
                    normalized_teacher,
                    dim=-1,
                ).mean()
            elif normalized_student.shape[0] > 1:
                batch_size = normalized_student.shape[0]
                off_diagonal_mask = ~torch.eye(
                    batch_size,
                    dtype=torch.bool,
                    device=student_embeddings.device,
                )
                student_relations = (normalized_student @ normalized_student.T)[off_diagonal_mask].view(batch_size, -1)
                teacher_relations = (normalized_teacher @ normalized_teacher.T)[off_diagonal_mask].view(batch_size, -1)
                distill_loss = 1.0 - F.cosine_similarity(
                    student_relations,
                    teacher_relations,
                    dim=-1,
                ).mean()
        elif distill_loss_type == "kl":
            if student_embeddings.shape[0] > 1:
                student_similarity = normalized_student @ normalized_student.T
                teacher_similarity = normalized_teacher @ normalized_teacher.T
                batch_size = student_embeddings.shape[0]
                off_diagonal_mask = ~torch.eye(
                    batch_size,
                    dtype=torch.bool,
                    device=student_embeddings.device,
                )
                student_logits = (student_similarity / distill_temperature)[off_diagonal_mask].view(batch_size, -1)
                teacher_probs = F.softmax(
                    (teacher_similarity / distill_temperature)[off_diagonal_mask].view(batch_size, -1),
                    dim=-1,
                )
                distill_loss = F.kl_div(
                    F.log_softmax(student_logits, dim=-1),
                    teacher_probs,
                    reduction="batchmean",
                ) * (distill_temperature ** 2)
        else:
            student_similarity = normalized_student @ normalized_student.T
            teacher_similarity = normalized_teacher @ normalized_teacher.T
            if student_embeddings.shape[0] > 1:
                off_diagonal_mask = ~torch.eye(
                    student_embeddings.shape[0],
                    dtype=torch.bool,
                    device=student_embeddings.device,
                )
                distill_loss = F.mse_loss(
                    student_similarity[off_diagonal_mask],
                    teacher_similarity[off_diagonal_mask],
                )

    baseline_anchor_loss = zero
    if baseline_anchor_weight > 0:
        baseline_anchor_loss = 1.0 - F.cosine_similarity(
            student_embeddings,
            reference_embeddings,
            dim=-1,
        ).mean()

    hard_negative_loss = zero
    if (
        hard_negative_weight > 0 and
        positive_embeddings is not None and
        negative_embeddings is not None and
        negative_mask is not None and
        bool(negative_mask.any().item())
    ):
        positive_scores = (student_embeddings * positive_embeddings).sum(dim=-1, keepdim=True)
        negative_scores = torch.einsum("bd,bnd->bn", student_embeddings, negative_embeddings)
        margin_loss = F.relu(hard_negative_margin - positive_scores + negative_scores)
        margin_loss = margin_loss * negative_mask.float()
        hard_negative_loss = margin_loss.sum() / negative_mask.float().sum().clamp_min(1.0)

    total_loss = (
        contrastive_weight * contrastive_loss +
        distill_weight * distill_loss +
        baseline_anchor_weight * baseline_anchor_loss +
        hard_negative_weight * hard_negative_loss
    )
    return total_loss, {
        "contrastive_loss": float(contrastive_loss.detach().cpu()),
        "distill_loss": float(distill_loss.detach().cpu()),
        "baseline_anchor_loss": float(baseline_anchor_loss.detach().cpu()),
        "hard_negative_loss": float(hard_negative_loss.detach().cpu()),
        "total_loss": float(total_loss.detach().cpu()),
    }


def save_training_checkpoint(
    checkpoint_path: Path,
    *,
    student_model: nn.Module,
    text_cache: dict[str, object] | None,
    teacher_cache: dict[str, object],
    trainable_modules: list[str],
    args: argparse.Namespace,
    history: list[dict[str, float]],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    best_total_loss: float | None = None,
    best_metric_name: str | None = None,
    best_metric_value: float | None = None,
    epoch: int | None = None,
) -> None:
    payload = {
        "student_model_id": args.student_model_id,
        "student_state_dict": student_model.state_dict(),
        "text_embedding_model_name": text_cache["model_name"] if text_cache is not None else None,
        "teacher_model_name": teacher_cache["model_name"],
        "trainable_modules": trainable_modules,
        "args": vars(args),
        "history": history,
        "epoch": epoch,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if best_total_loss is not None:
        payload["best_total_loss"] = float(best_total_loss)
    if best_metric_name is not None:
        payload["best_metric_name"] = best_metric_name
    if best_metric_value is not None:
        payload["best_metric_value"] = float(best_metric_value)
    torch.save(payload, checkpoint_path)


def main() -> None:
    args = parse_args()

    if args.grad_accumulation < 1:
        raise ValueError("--grad-accumulation must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    if (
        args.contrastive_weight <= 0 and
        args.distill_weight <= 0 and
        args.baseline_anchor_weight <= 0 and
        args.hard_negative_weight <= 0
    ):
        raise ValueError("At least one loss weight must be > 0.")

    text_supervision_enabled = args.contrastive_weight > 0 or args.hard_negative_weight > 0
    if text_supervision_enabled and not args.text_embeddings:
        raise ValueError(
            "Text supervision is enabled but --text-embeddings is missing. "
            "Set --contrastive-weight 0 and --hard-negative-weight 0 for image-only distillation."
        )
    if args.hard_negative_weight > 0 and not args.hard_negatives:
        raise ValueError("--hard-negatives is required when --hard-negative-weight > 0")
    if args.hard_negative_weight > 0 and args.num_hard_negatives < 1:
        raise ValueError("--num-hard-negatives must be >= 1 when --hard-negative-weight > 0")
    if args.val_split < 0 or args.val_split >= 1:
        raise ValueError("--val-split must be in [0, 1).")
    if args.val_every_epochs < 1:
        raise ValueError("--val-every-epochs must be >= 1")
    if args.lr_scheduler == "cosine" and args.lr <= 0:
        raise ValueError("--lr must be > 0 when using cosine scheduler")
    if args.distill_loss_type == "kl" and args.distill_temperature <= 0:
        raise ValueError("--distill-temperature must be > 0 for KL distillation")
    if args.val_recall_image_chunk_size < 1:
        raise ValueError("--val-recall-image-chunk-size must be >= 1")
    if args.val_recall_text_chunk_size < 1:
        raise ValueError("--val-recall-text-chunk-size must be >= 1")
    if args.val_recall_max_texts < 0:
        raise ValueError("--val-recall-max-texts must be >= 0")

    set_seed(args.seed)
    configure_cuda_runtime(device=args.device, allow_tf32=not args.no_tf32)
    num_workers = resolve_num_workers(args.num_workers, args.device)

    resume_payload = None
    resume_epoch = 0
    history: list[dict[str, float]] = []
    if args.resume_checkpoint:
        resume_path = resolve_resume_checkpoint_path(args.resume_checkpoint)
        resume_payload = torch.load(resume_path, map_location="cpu")
        resume_epoch = int(resume_payload.get("epoch", 0) or 0)
        history = list(resume_payload.get("history", []))
        checkpoint_model_id = resume_payload.get("student_model_id")
        if checkpoint_model_id:
            if checkpoint_model_id != args.student_model_id:
                print(
                    "Resume checkpoint model_id differs from --student-model-id; "
                    f"using checkpoint model_id: {checkpoint_model_id}"
                )
            args.student_model_id = checkpoint_model_id

    text_cache = load_text_embedding_cache(args.text_embeddings) if text_supervision_enabled else None
    teacher_cache = load_image_embedding_cache(args.teacher_embeddings)
    hard_negative_lookup = build_hard_negative_lookup(args.hard_negatives)

    model, preprocess_train, _ = open_clip.create_model_and_transforms(args.student_model_id)
    model = model.to(args.device)
    reference_model = StudentImageModel(copy.deepcopy(model)).to(args.device).eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad = False
    student_model = StudentImageModel(model).to(args.device)
    if args.channels_last and args.device.startswith("cuda"):
        reference_model = reference_model.to(memory_format=torch.channels_last)
        student_model = student_model.to(memory_format=torch.channels_last)
    trainable_modules = configure_trainable_student(
        student_model,
        unfreeze_last_n_blocks=args.unfreeze_last_n_blocks,
    )

    if resume_payload is not None:
        load_result = student_model.load_state_dict(resume_payload["student_state_dict"], strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            print(
                "Resume checkpoint compatibility: "
                f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
            )
        print(f"Resuming from checkpoint: {args.resume_checkpoint} (epoch={resume_epoch})")

    if args.gradient_checkpointing and hasattr(student_model.clip_model, "set_grad_checkpointing"):
        student_model.clip_model.set_grad_checkpointing(True)

    text_ids = None
    text_embeddings = None
    if text_cache is not None:
        text_ids = torch.as_tensor(text_cache["text_ids"], dtype=torch.long)
        text_embeddings = torch.as_tensor(text_cache["embeddings"], dtype=torch.float32)

    if text_supervision_enabled and text_embeddings is not None:
        text_embedding_dim = int(text_embeddings.shape[1])
        if student_model.embedding_dim != text_embedding_dim:
            raise ValueError(
                "Text embedding dimension mismatch: "
                f"student_dim={student_model.embedding_dim}, text_dim={text_embedding_dim}. "
                "Regenerate text embeddings with the same backbone used by --student-model-id "
                "or switch --student-model-id to match the text cache."
            )

    teacher_embeddings = torch.as_tensor(teacher_cache["embeddings"], dtype=torch.float32)

    if (
        args.distill_weight > 0 and
        args.distill_loss_type == "cosine" and
        student_model.embedding_dim != int(teacher_embeddings.shape[1])
    ):
        print(
            "Distill dimension mismatch detected "
            f"(student_dim={student_model.embedding_dim}, teacher_dim={int(teacher_embeddings.shape[1])}). "
            "Using relational cosine distillation over intra-batch similarities."
        )

    dataset = Track1DistillDataset(
        img_list_path=args.img_list,
        image_folder=args.image_folder,
        preprocess=preprocess_train,
        text_ids=text_ids,
        text_embeddings=text_embeddings,
        teacher_image_names=list(teacher_cache["image_names"]),
        teacher_embeddings=teacher_embeddings,
        hard_negative_lookup=hard_negative_lookup,
        num_hard_negatives=args.num_hard_negatives if args.hard_negative_weight > 0 else 0,
        strict_teacher_coverage=args.strict_teacher_coverage,
    )
    if len(dataset) == 0:
        reason_lines = [
            "No valid training samples were constructed from the dataset.",
            f"rows={dataset.total_rows}, skipped_missing_teacher={dataset.skipped_missing_teacher_count}, "
            f"skipped_missing_positive={dataset.skipped_missing_positive_count}",
        ]

        if dataset.skipped_missing_teacher_count > 0:
            teacher_examples = ", ".join(dataset.missing_teacher_images[:3])
            reason_lines.append(
                "Teacher embedding coverage issue detected. "
                f"Example missing image names: {teacher_examples}"
            )
            reason_lines.append(
                "Regenerate teacher embeddings using the same img-list and image-folder used for training: "
                "python precompute_teacher_embeddings.py --img-list <same_img_list> --image-folder <same_image_folder> --output <teacher_npz>"
            )

        if dataset.skipped_missing_positive_count > 0:
            positive_examples = ", ".join(dataset.missing_positive_images[:3])
            reason_lines.append(
                "Text embedding coverage issue detected. "
                f"Example images with no matching positive text IDs: {positive_examples}"
            )
            reason_lines.append(
                "Regenerate text embeddings and hard negatives from the same txt-list: "
                "python cache_text_embeddings.py --txt-list <same_txt_list> --output <text_npz> && "
                "python mine_hard_negatives.py --embeddings <text_npz> --output <hard_neg_npz>"
            )

        raise ValueError("\n".join(reason_lines))

    if dataset.missing_teacher_images:
        missing_count = len(dataset.missing_teacher_images)
        print(f"Skipped images without teacher embeddings: {missing_count}")
        if args.missing_teacher_log:
            missing_log_path = Path(args.missing_teacher_log)
            missing_log_path.parent.mkdir(parents=True, exist_ok=True)
            missing_log_path.write_text(
                "\n".join(dataset.missing_teacher_images) + "\n",
                encoding="utf-8",
            )
            print(f"Saved missing-teacher log to {missing_log_path}")

    train_dataset, val_dataset = build_train_val_subsets(
        dataset,
        val_split=args.val_split,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=num_workers > 0,
        prefetch_factor=args.prefetch_factor if num_workers > 0 else None,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=args.device.startswith("cuda"),
            persistent_workers=num_workers > 0,
            prefetch_factor=args.prefetch_factor if num_workers > 0 else None,
        )

    optimizer = torch.optim.AdamW(
        list(get_trainable_parameters(student_model)),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.lr_scheduler == "cosine":
        estimated_steps_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accumulation))
        total_optimizer_steps = max(1, estimated_steps_per_epoch * max(1, args.epochs))
        min_lr_scale = args.min_lr / args.lr
        scheduler = build_cosine_scheduler(
            optimizer,
            total_steps=total_optimizer_steps,
            warmup_steps=args.warmup_steps,
            min_lr_scale=min_lr_scale,
        )
    scaler_enabled = args.device.startswith("cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    if args.resume_optimizer_state and resume_payload is not None:
        optimizer_state = resume_payload.get("optimizer_state_dict")
        scheduler_state = resume_payload.get("scheduler_state_dict")
        scaler_state = resume_payload.get("scaler_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            print("Restored optimizer state from resume checkpoint.")
        else:
            print("Resume checkpoint has no optimizer_state_dict; optimizer starts fresh.")
        if scheduler is not None and scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
            print("Restored scheduler state from resume checkpoint.")
        elif scheduler is not None:
            print("Resume checkpoint has no scheduler_state_dict; scheduler starts fresh.")
        if scaler_state is not None:
            scaler.load_state_dict(scaler_state)
            print("Restored GradScaler state from resume checkpoint.")
        else:
            print("Resume checkpoint has no scaler_state_dict; GradScaler starts fresh.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if history:
        best_total_loss = min(metric.get("total_loss", float("inf")) for metric in history)
    else:
        best_total_loss = float("inf")
    if resume_payload is not None and "best_total_loss" in resume_payload:
        best_total_loss = min(best_total_loss, float(resume_payload["best_total_loss"]))

    can_compute_val_recall = (
        val_loader is not None and
        text_embeddings is not None and
        text_supervision_enabled
    )

    effective_best_checkpoint_metric = args.best_checkpoint_metric
    if effective_best_checkpoint_metric == "val-recall@10" and not can_compute_val_recall:
        fallback_metric = "val-loss" if val_loader is not None else "train-loss"
        print(
            "Best checkpoint metric val-recall@10 is unavailable "
            "(requires validation split with text supervision). "
            f"Falling back to {fallback_metric}."
        )
        effective_best_checkpoint_metric = fallback_metric

    if effective_best_checkpoint_metric == "val-loss":
        checkpoint_metric_name = "val_total_loss"
        checkpoint_metric_mode = "min"
    elif effective_best_checkpoint_metric == "val-recall@10":
        checkpoint_metric_name = "val_recall_at_10"
        checkpoint_metric_mode = "max"
    else:
        checkpoint_metric_name = "total_loss"
        checkpoint_metric_mode = "min"

    best_checkpoint_metric = float("inf") if checkpoint_metric_mode == "min" else -float("inf")
    if resume_payload is not None and "best_metric_name" in resume_payload and "best_metric_value" in resume_payload:
        if resume_payload["best_metric_name"] == checkpoint_metric_name:
            best_checkpoint_metric = float(resume_payload["best_metric_value"])
    has_initialized_best_metric = (
        best_checkpoint_metric < float("inf")
        if checkpoint_metric_mode == "min"
        else best_checkpoint_metric > -float("inf")
    )
    if history and not has_initialized_best_metric:
        historical_values = [
            metric.get(checkpoint_metric_name)
            for metric in history
            if checkpoint_metric_name in metric
        ]
        if historical_values:
            if checkpoint_metric_mode == "min":
                best_checkpoint_metric = min(float(value) for value in historical_values)
            else:
                best_checkpoint_metric = max(float(value) for value in historical_values)

    print(f"Training samples: {len(train_dataset)}")
    if val_dataset is not None:
        print(f"Validation samples: {len(val_dataset)}")
    print(
        "Dataset coverage summary: "
        f"rows={dataset.total_rows}, kept={len(dataset)}, "
        f"skipped_missing_teacher={dataset.skipped_missing_teacher_count}, "
        f"skipped_missing_positive={dataset.skipped_missing_positive_count}"
    )
    print(f"Trainable modules: {', '.join(trainable_modules)}")
    print(f"Trainable parameters: {count_trainable_parameters(student_model):,}")
    print(f"DataLoader workers: {num_workers}")
    print(
        "Loss weights: "
        f"contrastive={args.contrastive_weight}, "
        f"distill={args.distill_weight}, "
        f"baseline_anchor={args.baseline_anchor_weight}, "
        f"hard_negative={args.hard_negative_weight}"
    )
    print(f"Text supervision enabled: {text_supervision_enabled}")
    print(f"Distill loss type: {args.distill_loss_type}")
    print(
        "Anchor schedule: "
        f"start={args.baseline_anchor_weight}, end={args.baseline_anchor_final_weight}"
    )
    if scheduler is not None:
        print(
            "LR scheduler: cosine "
            f"(warmup_steps={args.warmup_steps}, min_lr={args.min_lr})"
        )
    else:
        print("LR scheduler: none")
    print(f"Best checkpoint metric: {checkpoint_metric_name} ({checkpoint_metric_mode})")
    if can_compute_val_recall:
        candidate_text_count = min(int(text_embeddings.shape[0]), args.val_recall_max_texts) if args.val_recall_max_texts > 0 else int(text_embeddings.shape[0])
        print(
            "Validation Recall@10: enabled "
            f"(candidate_texts={candidate_text_count}, image_chunk={args.val_recall_image_chunk_size}, text_chunk={args.val_recall_text_chunk_size})"
        )
    else:
        print("Validation Recall@10: disabled")
    if args.device.startswith("cuda"):
        print(f"Channels-last: {args.channels_last}")
        print(f"TF32 enabled: {not args.no_tf32}")

    student_model.train()

    total_target_epoch = resume_epoch + args.epochs
    for epoch in range(args.epochs):
        current_epoch = resume_epoch + epoch + 1
        current_anchor_weight = interpolate_weight(
            start=args.baseline_anchor_weight,
            end=args.baseline_anchor_final_weight,
            step=epoch,
            total_steps=max(1, args.epochs),
        )
        optimizer.zero_grad(set_to_none=True)
        running = {
            "contrastive_loss": 0.0,
            "distill_loss": 0.0,
            "baseline_anchor_loss": 0.0,
            "hard_negative_loss": 0.0,
            "total_loss": 0.0,
        }
        optimizer_steps = 0

        progress = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Epoch {current_epoch}/{total_target_epoch}",
            dynamic_ncols=True,
        )

        for step, batch in enumerate(progress, start=1):
            pixel_values = batch["pixel_values"].to(args.device, non_blocking=True)
            if args.channels_last and args.device.startswith("cuda"):
                pixel_values = pixel_values.contiguous(memory_format=torch.channels_last)

            positive_embeddings = None
            if text_supervision_enabled:
                positive_embeddings = F.normalize(
                    batch["positive_embedding"].to(args.device, non_blocking=True),
                    dim=-1,
                )

            teacher_targets = batch["teacher_embedding"].to(args.device, non_blocking=True)

            negative_embeddings = None
            negative_mask = None
            if args.hard_negative_weight > 0:
                negative_embeddings = F.normalize(
                    batch["negative_embeddings"].to(args.device, non_blocking=True),
                    dim=-1,
                )
                negative_mask = batch["negative_mask"].to(args.device, non_blocking=True)

            autocast_context = (
                torch.autocast(device_type="cuda", enabled=True)
                if args.device.startswith("cuda")
                else nullcontext()
            )
            with autocast_context:
                with torch.no_grad():
                    reference_embeddings = reference_model(pixel_values)
                student_embeddings = student_model(pixel_values)
                loss, metrics = compute_losses(
                    student_embeddings=student_embeddings,
                    reference_embeddings=reference_embeddings,
                    positive_embeddings=positive_embeddings,
                    teacher_embeddings=teacher_targets,
                    negative_embeddings=negative_embeddings,
                    negative_mask=negative_mask,
                    temperature=args.temperature,
                    contrastive_weight=args.contrastive_weight,
                    distill_weight=args.distill_weight,
                    distill_loss_type=args.distill_loss_type,
                    distill_temperature=args.distill_temperature,
                    baseline_anchor_weight=current_anchor_weight,
                    hard_negative_weight=args.hard_negative_weight,
                    hard_negative_margin=args.hard_negative_margin,
                )
                scaled_loss = loss / args.grad_accumulation

            scaler.scale(scaled_loss).backward()

            if step % args.grad_accumulation == 0 or step == len(train_loader):
                if args.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(get_trainable_parameters(student_model)),
                        max_norm=args.grad_clip_norm,
                    )
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

            for key, value in metrics.items():
                running[key] += value

            postfix = {
                "loss": f"{metrics['total_loss']:.4f}",
                "ctr": f"{metrics['contrastive_loss']:.4f}",
                "dist": f"{metrics['distill_loss']:.4f}",
                "anchor": f"{metrics['baseline_anchor_loss']:.4f}",
                "opt": optimizer_steps,
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            }
            if args.device.startswith("cuda"):
                memory_gb = torch.cuda.memory_reserved(device=args.device) / (1024 ** 3)
                postfix["mem_gb"] = f"{memory_gb:.1f}"
            progress.set_postfix(postfix)

        progress.close()

        epoch_metrics = {key: value / len(train_loader) for key, value in running.items()}
        epoch_metrics["epoch"] = current_epoch
        epoch_metrics["optimizer_steps"] = optimizer_steps
        epoch_metrics["lr"] = float(optimizer.param_groups[0]["lr"])
        epoch_metrics["anchor_weight"] = float(current_anchor_weight)

        should_run_validation = (
            val_loader is not None and
            (current_epoch % args.val_every_epochs == 0 or current_epoch == total_target_epoch)
        )
        if should_run_validation:
            student_model.eval()
            val_running = {
                "contrastive_loss": 0.0,
                "distill_loss": 0.0,
                "baseline_anchor_loss": 0.0,
                "hard_negative_loss": 0.0,
                "total_loss": 0.0,
            }
            val_recall_embeddings: list[torch.Tensor] = []
            val_recall_sample_indices: list[torch.Tensor] = []
            with torch.inference_mode():
                for val_batch in val_loader:
                    pixel_values = val_batch["pixel_values"].to(args.device, non_blocking=True)
                    if args.channels_last and args.device.startswith("cuda"):
                        pixel_values = pixel_values.contiguous(memory_format=torch.channels_last)

                    positive_embeddings = None
                    if text_supervision_enabled:
                        positive_embeddings = F.normalize(
                            val_batch["positive_embedding"].to(args.device, non_blocking=True),
                            dim=-1,
                        )

                    teacher_targets = val_batch["teacher_embedding"].to(args.device, non_blocking=True)

                    negative_embeddings = None
                    negative_mask = None
                    if args.hard_negative_weight > 0:
                        negative_embeddings = F.normalize(
                            val_batch["negative_embeddings"].to(args.device, non_blocking=True),
                            dim=-1,
                        )
                        negative_mask = val_batch["negative_mask"].to(args.device, non_blocking=True)

                    autocast_context = (
                        torch.autocast(device_type="cuda", enabled=True)
                        if args.device.startswith("cuda")
                        else nullcontext()
                    )
                    with autocast_context:
                        reference_embeddings = reference_model(pixel_values)
                        student_embeddings = student_model(pixel_values)
                        _, val_metrics = compute_losses(
                            student_embeddings=student_embeddings,
                            reference_embeddings=reference_embeddings,
                            positive_embeddings=positive_embeddings,
                            teacher_embeddings=teacher_targets,
                            negative_embeddings=negative_embeddings,
                            negative_mask=negative_mask,
                            temperature=args.temperature,
                            contrastive_weight=args.contrastive_weight,
                            distill_weight=args.distill_weight,
                            distill_loss_type=args.distill_loss_type,
                            distill_temperature=args.distill_temperature,
                            baseline_anchor_weight=current_anchor_weight,
                            hard_negative_weight=args.hard_negative_weight,
                            hard_negative_margin=args.hard_negative_margin,
                        )

                    if can_compute_val_recall:
                        val_recall_embeddings.append(student_embeddings.detach().float().cpu())
                        val_recall_sample_indices.append(val_batch["sample_index"].detach().long().cpu())

                    for key, value in val_metrics.items():
                        val_running[key] += value

            val_metrics_avg = {f"val_{key}": value / len(val_loader) for key, value in val_running.items()}
            epoch_metrics.update(val_metrics_avg)

            if can_compute_val_recall and val_recall_embeddings and val_recall_sample_indices:
                val_recall_at_10 = evaluate_val_recall_at_k(
                    image_embeddings=torch.cat(val_recall_embeddings, dim=0),
                    sample_indices=torch.cat(val_recall_sample_indices, dim=0),
                    dataset=dataset,
                    text_embeddings=text_embeddings,
                    top_k=10,
                    image_chunk_size=args.val_recall_image_chunk_size,
                    text_chunk_size=args.val_recall_text_chunk_size,
                    max_texts=args.val_recall_max_texts,
                    device=args.device,
                )
                if val_recall_at_10 is not None:
                    epoch_metrics["val_recall_at_10"] = float(val_recall_at_10)

            student_model.train()

        if args.device.startswith("cuda"):
            epoch_metrics["max_memory_reserved_gb"] = round(
                torch.cuda.max_memory_reserved(device=args.device) / (1024 ** 3),
                3,
            )
            torch.cuda.reset_peak_memory_stats(device=args.device)
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, sort_keys=True))

        if args.save_epoch_checkpoints:
            epoch_checkpoint_path = output_dir / f"student_checkpoint_epoch_{current_epoch:02d}.pt"
            save_training_checkpoint(
                epoch_checkpoint_path,
                student_model=student_model,
                text_cache=text_cache,
                teacher_cache=teacher_cache,
                trainable_modules=trainable_modules,
                args=args,
                history=history,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_total_loss=best_total_loss,
                best_metric_name=checkpoint_metric_name,
                best_metric_value=(
                    best_checkpoint_metric
                    if (
                        best_checkpoint_metric < float("inf")
                        if checkpoint_metric_mode == "min"
                        else best_checkpoint_metric > -float("inf")
                    )
                    else None
                ),
                epoch=current_epoch,
            )

        best_total_loss = min(best_total_loss, float(epoch_metrics["total_loss"]))

        checkpoint_metric_value = epoch_metrics.get(checkpoint_metric_name)
        if checkpoint_metric_value is None and checkpoint_metric_name == "val_total_loss":
            checkpoint_metric_value = epoch_metrics["total_loss"]

        should_update_best = False
        if checkpoint_metric_value is not None:
            checkpoint_metric_value = float(checkpoint_metric_value)
            if not math.isnan(checkpoint_metric_value):
                if checkpoint_metric_mode == "min":
                    should_update_best = checkpoint_metric_value < best_checkpoint_metric
                else:
                    should_update_best = checkpoint_metric_value > best_checkpoint_metric

        if should_update_best:
            best_checkpoint_metric = float(checkpoint_metric_value)
            best_checkpoint_path = output_dir / "best_loss_checkpoint.pt"
            save_training_checkpoint(
                best_checkpoint_path,
                student_model=student_model,
                text_cache=text_cache,
                teacher_cache=teacher_cache,
                trainable_modules=trainable_modules,
                args=args,
                history=history,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_total_loss=best_total_loss,
                best_metric_name=checkpoint_metric_name,
                best_metric_value=best_checkpoint_metric,
                epoch=current_epoch,
            )

    checkpoint_path = output_dir / "student_checkpoint.pt"
    save_training_checkpoint(
        checkpoint_path,
        student_model=student_model,
        text_cache=text_cache,
        teacher_cache=teacher_cache,
        trainable_modules=trainable_modules,
        args=args,
        history=history,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        best_total_loss=best_total_loss,
        best_metric_name=checkpoint_metric_name,
        best_metric_value=(
            best_checkpoint_metric
            if (
                best_checkpoint_metric < float("inf")
                if checkpoint_metric_mode == "min"
                else best_checkpoint_metric > -float("inf")
            )
            else None
        ),
        epoch=total_target_epoch,
    )
    metrics_path = output_dir / "history.json"
    metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved best-loss checkpoint to {output_dir / 'best_loss_checkpoint.pt'}")
    print(f"Saved history to {metrics_path}")


if __name__ == "__main__":
    main()
