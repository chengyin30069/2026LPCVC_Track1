import argparse
import copy
import json
import os
from contextlib import nullcontext
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
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


class Track1DistillDataset(Dataset):
    def __init__(
        self,
        *,
        img_list_path: str,
        image_folder: str,
        preprocess,
        text_ids: torch.Tensor,
        text_embeddings: torch.Tensor,
        teacher_image_names: list[str],
        teacher_embeddings: torch.Tensor,
        hard_negative_lookup: dict[int, list[int]] | None,
        num_hard_negatives: int,
    ):
        self.preprocess = preprocess
        self.text_embeddings = text_embeddings
        self.text_index = {int(text_id): index for index, text_id in enumerate(text_ids.tolist())}
        self.teacher_index = {
            image_name: teacher_embeddings[index]
            for index, image_name in enumerate(teacher_image_names)
        }
        self.hard_negative_lookup = hard_negative_lookup or {}
        self.num_hard_negatives = num_hard_negatives

        image_table = load_track1_image_table(img_list_path, image_folder)
        self.samples: list[dict[str, object]] = []

        for row in tqdm(
            image_table.itertuples(index=False),
            total=len(image_table),
            desc="Preparing training samples",
            unit="image",
            dynamic_ncols=True,
        ):
            positive_ids = [text_id for text_id in row.positive_text_ids if text_id in self.text_index]
            if not positive_ids:
                continue
            if row.image_name not in self.teacher_index:
                raise ValueError(f"Missing teacher embedding for image: {row.image_name}")

            positive_indices = [self.text_index[text_id] for text_id in positive_ids]

            hard_negative_ids: list[int] = []
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
                    "teacher_embedding": self.teacher_index[row.image_name],
                    "hard_negative_ids": hard_negative_ids,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        image = Image.open(sample["image_path"]).convert("RGB")
        pixel_values = self.preprocess(image)
        positive_indices = sample["positive_indices"]
        selected_positive_index = positive_indices[torch.randint(len(positive_indices), (1,)).item()]
        positive_embedding = self.text_embeddings[selected_positive_index]

        negative_embeddings = torch.zeros(
            self.num_hard_negatives,
            self.text_embeddings.shape[1],
            dtype=self.text_embeddings.dtype,
        )
        negative_mask = torch.zeros(self.num_hard_negatives, dtype=torch.bool)

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
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a distilled student image model for Track1.")
    parser.add_argument("--img-list", default="dataset/img_list.csv")
    parser.add_argument("--image-folder", default="dataset/images")
    parser.add_argument("--text-embeddings", default="artifacts/text_embeddings.npz")
    parser.add_argument("--teacher-embeddings", default="artifacts/teacher_image_embeddings.npz")
    parser.add_argument("--hard-negatives", help="Optional hard-negative NPZ file.")
    parser.add_argument(
        "--student-model-id",
        default="hf-hub:laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K",
    )
    parser.add_argument("--output-dir", default="artifacts/student_distill")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--distill-weight", type=float, default=0.3)
    parser.add_argument("--baseline-anchor-weight", type=float, default=0.1)
    parser.add_argument("--hard-negative-weight", type=float, default=0.2)
    parser.add_argument("--hard-negative-margin", type=float, default=0.05)
    parser.add_argument("--num-hard-negatives", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--unfreeze-last-n-blocks", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--save-epoch-checkpoints", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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


def compute_losses(
    *,
    student_embeddings: torch.Tensor,
    reference_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor,
    negative_mask: torch.Tensor,
    temperature: float,
    distill_weight: float,
    baseline_anchor_weight: float,
    hard_negative_weight: float,
    hard_negative_margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch_size = student_embeddings.shape[0]
    labels = torch.arange(batch_size, device=student_embeddings.device)
    logits = student_embeddings @ positive_embeddings.T / temperature
    contrastive_loss = 0.5 * (
        F.cross_entropy(logits, labels) +
        F.cross_entropy(logits.T, labels)
    )

    normalized_teacher = F.normalize(teacher_embeddings, dim=-1)
    student_similarity = student_embeddings @ student_embeddings.T
    teacher_similarity = normalized_teacher @ normalized_teacher.T
    if student_embeddings.shape[0] > 1:
        off_diagonal_mask = ~torch.eye(student_embeddings.shape[0], dtype=torch.bool, device=student_embeddings.device)
        distill_loss = F.mse_loss(student_similarity[off_diagonal_mask], teacher_similarity[off_diagonal_mask])
    else:
        distill_loss = torch.zeros((), device=student_embeddings.device)

    baseline_anchor_loss = 1.0 - F.cosine_similarity(student_embeddings, reference_embeddings, dim=-1).mean()

    if negative_mask.any():
        positive_scores = (student_embeddings * positive_embeddings).sum(dim=-1, keepdim=True)
        negative_scores = torch.einsum("bd,bnd->bn", student_embeddings, negative_embeddings)
        margin_loss = F.relu(hard_negative_margin - positive_scores + negative_scores)
        margin_loss = margin_loss * negative_mask.float()
        hard_negative_loss = margin_loss.sum() / negative_mask.float().sum().clamp_min(1.0)
    else:
        hard_negative_loss = torch.zeros((), device=student_embeddings.device)

    total_loss = (
        contrastive_loss +
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
    text_cache: dict[str, object],
    teacher_cache: dict[str, object],
    trainable_modules: list[str],
    args: argparse.Namespace,
    history: list[dict[str, float]],
    epoch: int | None = None,
) -> None:
    torch.save(
        {
            "student_model_id": args.student_model_id,
            "student_state_dict": student_model.state_dict(),
            "text_embedding_model_name": text_cache["model_name"],
            "teacher_model_name": teacher_cache["model_name"],
            "trainable_modules": trainable_modules,
            "args": vars(args),
            "history": history,
            "epoch": epoch,
        },
        checkpoint_path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    configure_cuda_runtime(device=args.device, allow_tf32=not args.no_tf32)
    num_workers = resolve_num_workers(args.num_workers, args.device)

    text_cache = load_text_embedding_cache(args.text_embeddings)
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

    if args.gradient_checkpointing and hasattr(student_model.clip_model, "set_grad_checkpointing"):
        student_model.clip_model.set_grad_checkpointing(True)

    text_ids = torch.as_tensor(text_cache["text_ids"], dtype=torch.long)
    text_embeddings = torch.as_tensor(text_cache["embeddings"], dtype=torch.float32)
    teacher_embeddings = torch.as_tensor(teacher_cache["embeddings"], dtype=torch.float32)

    dataset = Track1DistillDataset(
        img_list_path=args.img_list,
        image_folder=args.image_folder,
        preprocess=preprocess_train,
        text_ids=text_ids,
        text_embeddings=text_embeddings,
        teacher_image_names=list(teacher_cache["image_names"]),
        teacher_embeddings=teacher_embeddings,
        hard_negative_lookup=hard_negative_lookup,
        num_hard_negatives=args.num_hard_negatives,
    )
    if len(dataset) == 0:
        raise ValueError("No valid training samples were constructed from the dataset.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
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
    scaler = torch.cuda.amp.GradScaler(enabled=args.device.startswith("cuda"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    best_total_loss = float("inf")

    print(f"Training samples: {len(dataset)}")
    print(f"Trainable modules: {', '.join(trainable_modules)}")
    print(f"Trainable parameters: {count_trainable_parameters(student_model):,}")
    print(f"DataLoader workers: {num_workers}")
    if args.device.startswith("cuda"):
        print(f"Channels-last: {args.channels_last}")
        print(f"TF32 enabled: {not args.no_tf32}")

    student_model.train()

    for epoch in range(args.epochs):
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
            loader,
            total=len(loader),
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            dynamic_ncols=True,
        )

        for step, batch in enumerate(progress, start=1):
            pixel_values = batch["pixel_values"].to(args.device, non_blocking=True)
            if args.channels_last and args.device.startswith("cuda"):
                pixel_values = pixel_values.contiguous(memory_format=torch.channels_last)
            positive_embeddings = F.normalize(batch["positive_embedding"].to(args.device, non_blocking=True), dim=-1)
            teacher_targets = batch["teacher_embedding"].to(args.device, non_blocking=True)
            negative_embeddings = F.normalize(batch["negative_embeddings"].to(args.device, non_blocking=True), dim=-1)
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
                    distill_weight=args.distill_weight,
                    baseline_anchor_weight=args.baseline_anchor_weight,
                    hard_negative_weight=args.hard_negative_weight,
                    hard_negative_margin=args.hard_negative_margin,
                )
                scaled_loss = loss / args.grad_accumulation

            scaler.scale(scaled_loss).backward()

            if step % args.grad_accumulation == 0 or step == len(loader):
                scaler.step(optimizer)
                scaler.update()
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
            }
            if args.device.startswith("cuda"):
                memory_gb = torch.cuda.memory_reserved(device=args.device) / (1024 ** 3)
                postfix["mem_gb"] = f"{memory_gb:.1f}"
            progress.set_postfix(postfix)

        progress.close()

        epoch_metrics = {key: value / len(loader) for key, value in running.items()}
        epoch_metrics["epoch"] = epoch + 1
        epoch_metrics["optimizer_steps"] = optimizer_steps
        if args.device.startswith("cuda"):
            epoch_metrics["max_memory_reserved_gb"] = round(
                torch.cuda.max_memory_reserved(device=args.device) / (1024 ** 3),
                3,
            )
            torch.cuda.reset_peak_memory_stats(device=args.device)
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, sort_keys=True))

        if args.save_epoch_checkpoints:
            epoch_checkpoint_path = output_dir / f"student_checkpoint_epoch_{epoch + 1:02d}.pt"
            save_training_checkpoint(
                epoch_checkpoint_path,
                student_model=student_model,
                text_cache=text_cache,
                teacher_cache=teacher_cache,
                trainable_modules=trainable_modules,
                args=args,
                history=history,
                epoch=epoch + 1,
            )

        if epoch_metrics["total_loss"] < best_total_loss:
            best_total_loss = epoch_metrics["total_loss"]
            best_checkpoint_path = output_dir / "best_loss_checkpoint.pt"
            save_training_checkpoint(
                best_checkpoint_path,
                student_model=student_model,
                text_cache=text_cache,
                teacher_cache=teacher_cache,
                trainable_modules=trainable_modules,
                args=args,
                history=history,
                epoch=epoch + 1,
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
        epoch=args.epochs,
    )
    metrics_path = output_dir / "history.json"
    metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved best-loss checkpoint to {output_dir / 'best_loss_checkpoint.pt'}")
    print(f"Saved history to {metrics_path}")


if __name__ == "__main__":
    main()