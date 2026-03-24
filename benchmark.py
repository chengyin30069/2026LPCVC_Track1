import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import open_clip
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CocoCaptions
from tqdm.auto import tqdm

from utils.student_model import StudentImageModel


ImageInputs = torch.Tensor | dict[str, torch.Tensor]


class StudentEvalWrapper(nn.Module):
    def __init__(self, clip_model: nn.Module, student_model: StudentImageModel):
        super().__init__()
        self.clip_model = clip_model
        self.student_model = student_model

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        features = self.clip_model.encode_text(token_ids)
        return F.normalize(features, dim=-1)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.student_model(pixel_values)


class TransformersEvalWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def encode_text(self, token_batch: Any) -> torch.Tensor:
        if isinstance(token_batch, dict):
            model_inputs = token_batch
        else:
            model_inputs = {
                key: value
                for key, value in token_batch.items()
                if isinstance(value, torch.Tensor)
            }
        features = extract_transformers_text_features(self.model, model_inputs)
        return features

    def encode_image(self, image_inputs: ImageInputs) -> torch.Tensor:
        if isinstance(image_inputs, dict):
            model_inputs = image_inputs
        else:
            model_inputs = {"pixel_values": image_inputs}
        features = extract_transformers_image_features(self.model, model_inputs)
        return features

    def get_logit_scale_bias(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        scale = None
        bias = None
        if hasattr(self.model, "logit_scale"):
            candidate = getattr(self.model, "logit_scale")
            if isinstance(candidate, torch.Tensor):
                scale = candidate
        if hasattr(self.model, "logit_bias"):
            candidate = getattr(self.model, "logit_bias")
            if isinstance(candidate, torch.Tensor):
                bias = candidate
        return scale, bias


def resolve_backend(model_id: str, requested_backend: str) -> str:
    if requested_backend in {"open_clip", "transformers"}:
        return requested_backend
    lowered = model_id.lower()
    if lowered.startswith("google/siglip") or lowered.startswith("google/siglip2"):
        return "transformers"
    return "open_clip"


def _coerce_feature_tensor(output_obj) -> torch.Tensor | None:
    if isinstance(output_obj, torch.Tensor):
        return output_obj
    for candidate_name in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        if hasattr(output_obj, candidate_name):
            candidate = getattr(output_obj, candidate_name)
            if isinstance(candidate, torch.Tensor):
                if candidate.ndim == 3:
                    return candidate[:, 0, :]
                return candidate
    return None


def extract_transformers_image_features(model, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    if hasattr(model, "get_image_features"):
        candidate = _coerce_feature_tensor(model.get_image_features(**inputs))
        if candidate is not None:
            return candidate

    outputs = model(**inputs)
    candidate = _coerce_feature_tensor(outputs)
    if candidate is not None:
        return candidate

    raise RuntimeError("Unable to extract image features from transformers model output")


def extract_transformers_text_features(model, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    if hasattr(model, "get_text_features"):
        candidate = _coerce_feature_tensor(model.get_text_features(**inputs))
        if candidate is not None:
            return candidate

    outputs = model(**inputs)
    candidate = _coerce_feature_tensor(outputs)
    if candidate is not None:
        return candidate

    raise RuntimeError("Unable to extract text features from transformers model output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark baseline, teacher, and student-v3 on COCO Recall@10.")
    parser.add_argument("--coco-root", default="coco2017/val2017")
    parser.add_argument("--coco-ann", default="coco2017/annotations/captions_val2017.json")
    parser.add_argument("--baseline-model-id", default="hf-hub:laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K")
    parser.add_argument("--teacher-model-id", default="google/siglip2-so400m-patch16-512")
    parser.add_argument(
        "--teacher-backend",
        choices=["auto", "open_clip", "transformers"],
        default="auto",
        help="Teacher backend. auto chooses transformers for google/siglip* models.",
    )
    parser.add_argument("--student-checkpoint", default="artifacts/student_stage2_text_align_v2/best_loss_checkpoint.pt")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument(
        "--teacher-feature-batch-size",
        type=int,
        default=32,
        help="Feature chunk batch size for teacher evaluation.",
    )
    parser.add_argument(
        "--teacher-text-feature-batch-size",
        type=int,
        default=64,
        help="Text feature chunk batch size for teacher evaluation.",
    )
    parser.add_argument(
        "--teacher-score-mode",
        choices=["cosine", "logit"],
        default="cosine",
        help="Teacher transformers similarity mode.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--transformers-amp",
        action="store_true",
        help="Enable AMP for transformers-based teacher evaluation (disabled by default for numerical stability).",
    )
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument(
        "--debug-similarity",
        action="store_true",
        help="Print average positive vs shuffled-negative similarity during evaluation.",
    )
    parser.add_argument(
        "--legacy-full-batches-only",
        action="store_true",
        help="Mimic benchmark_legacy.py behavior by skipping the last incomplete eval batch.",
    )
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def configure_cuda_runtime(device: str) -> None:
    if not device.startswith("cuda"):
        return
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def coco_collate(batch):
    images = [item[0] for item in batch]
    captions = [item[1] for item in batch]

    first_image = images[0]
    if isinstance(first_image, dict):
        collated_images: dict[str, torch.Tensor] = {}
        for key in first_image.keys():
            values = [image[key] for image in images]
            if isinstance(values[0], torch.Tensor):
                collated_images[key] = torch.stack(values, dim=0)
        return collated_images, captions

    return torch.stack(images, dim=0), captions


def load_coco_dataset(coco_root: str, coco_ann: str, preprocess, max_samples: int | None):
    dataset = CocoCaptions(root=coco_root, annFile=coco_ann, transform=preprocess)
    if max_samples is not None:
        max_samples = min(max_samples, len(dataset))
        dataset = Subset(dataset, list(range(max_samples)))
    return dataset


def encode_image_batch(
    model,
    image_batch: ImageInputs,
    *,
    feature_batch_size: int,
    device: str,
    amp_enabled: bool,
    channels_last: bool,
    normalize_output: bool = True,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []

    if isinstance(image_batch, dict):
        total_items = next(iter(image_batch.values())).shape[0]
    else:
        total_items = image_batch.shape[0]

    for start in range(0, total_items, feature_batch_size):
        end = start + feature_batch_size

        if isinstance(image_batch, dict):
            chunk: ImageInputs = {
                key: value[start:end].to(device, non_blocking=True)
                for key, value in image_batch.items()
            }
            if channels_last and device.startswith("cuda") and "pixel_values" in chunk:
                chunk["pixel_values"] = chunk["pixel_values"].contiguous(memory_format=torch.channels_last)
        else:
            chunk = image_batch[start:end].to(device, non_blocking=True)
            if channels_last and device.startswith("cuda"):
                chunk = chunk.contiguous(memory_format=torch.channels_last)

        autocast_context = (
            torch.autocast(device_type="cuda", enabled=amp_enabled and device.startswith("cuda"))
            if device.startswith("cuda")
            else nullcontext()
        )
        with autocast_context:
            features = model.encode_image(chunk)
            if normalize_output:
                features = F.normalize(features, dim=-1)
        outputs.append(features)
    return torch.cat(outputs, dim=0)


def encode_text_batch(
    model,
    tokenizer,
    captions_batch: list[list[str]],
    *,
    feature_batch_size: int,
    device: str,
    amp_enabled: bool,
    normalize_output: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    texts: list[str] = []
    owner_indices: list[int] = []
    for image_index, captions in enumerate(captions_batch):
        for caption in captions:
            texts.append(caption)
            owner_indices.append(image_index)

    if not texts:
        empty_features = torch.empty((0, 512), device=device)
        empty_owners = torch.empty((0,), device=device, dtype=torch.long)
        return empty_features, empty_owners

    outputs: list[torch.Tensor] = []
    for start in range(0, len(texts), feature_batch_size):
        token_batch = tokenizer(texts[start:start + feature_batch_size])
        if hasattr(token_batch, "to"):
            token_batch = token_batch.to(device)
        autocast_context = (
            torch.autocast(device_type="cuda", enabled=amp_enabled and device.startswith("cuda"))
            if device.startswith("cuda")
            else nullcontext()
        )
        with autocast_context:
            text_features = model.encode_text(token_batch)
            if normalize_output:
                text_features = F.normalize(text_features, dim=-1)
        outputs.append(text_features)

    owner_tensor = torch.as_tensor(owner_indices, dtype=torch.long, device=device)
    return torch.cat(outputs, dim=0), owner_tensor


def evaluate_coco_recall_at10(
    *,
    model,
    tokenizer,
    preprocess,
    args: argparse.Namespace,
    label: str,
) -> float:
    device = args.device
    amp_enabled = device.startswith("cuda") and not args.no_amp
    if isinstance(model, TransformersEvalWrapper) and not args.transformers_amp:
        amp_enabled = False

    dataset = load_coco_dataset(args.coco_root, args.coco_ann, preprocess, args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
        collate_fn=coco_collate,
    )

    model = model.to(device).eval()
    if args.channels_last and device.startswith("cuda"):
        model = model.to(memory_format=torch.channels_last)

    image_feature_batch_size = args.feature_batch_size
    text_feature_batch_size = args.feature_batch_size
    is_teacher_transformers = isinstance(model, TransformersEvalWrapper) and label == "teacher"
    use_transformers_teacher_logits = is_teacher_transformers and args.teacher_score_mode == "logit"
    if is_teacher_transformers:
        image_feature_batch_size = max(1, int(args.teacher_feature_batch_size))
        text_feature_batch_size = max(1, int(args.teacher_text_feature_batch_size))

    total_recall = 0.0
    total_images = 0
    debug_pos_similarity_sum = 0.0
    debug_neg_similarity_sum = 0.0
    debug_count = 0

    with torch.inference_mode():
        for image_batch, captions_batch in tqdm(loader, desc=f"Evaluating {label}", dynamic_ncols=True):
            if args.legacy_full_batches_only:
                current_batch = (
                    next(iter(image_batch.values())).shape[0]
                    if isinstance(image_batch, dict)
                    else image_batch.shape[0]
                )
                if current_batch < args.eval_batch_size:
                    continue

            image_features = encode_image_batch(
                model,
                image_batch,
                feature_batch_size=image_feature_batch_size,
                device=device,
                amp_enabled=amp_enabled,
                channels_last=args.channels_last,
                normalize_output=not use_transformers_teacher_logits,
            )
            text_features, text_owner_indices = encode_text_batch(
                model,
                tokenizer,
                captions_batch,
                feature_batch_size=text_feature_batch_size,
                device=device,
                amp_enabled=amp_enabled,
                normalize_output=not use_transformers_teacher_logits,
            )

            if text_features.numel() == 0:
                continue

            similarity = image_features @ text_features.T
            if use_transformers_teacher_logits:
                logit_scale, logit_bias = model.get_logit_scale_bias()
                if logit_scale is not None:
                    similarity = similarity * torch.exp(logit_scale).to(similarity.dtype)
                if logit_bias is not None:
                    similarity = similarity + logit_bias.to(similarity.dtype)
            top_k = min(10, text_features.shape[0])
            if top_k <= 0:
                continue

            top_indices = torch.topk(similarity, k=top_k, dim=1).indices
            top_owner_indices = text_owner_indices[top_indices]
            target_owner_indices = torch.arange(image_features.shape[0], device=device).unsqueeze(1)
            matches = (top_owner_indices == target_owner_indices).sum(dim=1).float()

            if args.debug_similarity:
                captions_per_image = max(1, text_features.shape[0] // max(1, image_features.shape[0]))
                positive_text_indices = (
                    torch.arange(image_features.shape[0], device=device) * captions_per_image
                ).clamp_max(text_features.shape[0] - 1)
                positive_text = text_features[positive_text_indices]
                neg_indices = torch.roll(positive_text_indices, shifts=1)
                negative_text = text_features[neg_indices]
                debug_pos_similarity_sum += float((image_features * positive_text).sum(dim=1).mean().item())
                debug_neg_similarity_sum += float((image_features * negative_text).sum(dim=1).mean().item())
                debug_count += 1

            # Match benchmark_legacy.py style: fractional recall per image uses matches/10.
            recall_per_image = matches / 10.0

            batch_size = image_features.shape[0]
            total_recall += recall_per_image.sum().item()
            total_images += batch_size

    if total_images == 0:
        raise RuntimeError("No images were evaluated. Check COCO paths and max_samples.")

    if args.debug_similarity and debug_count > 0:
        print(
            f"[{label}] mean pos sim={debug_pos_similarity_sum / debug_count:.4f}, "
            f"mean shuffled-neg sim={debug_neg_similarity_sum / debug_count:.4f}"
        )

    return total_recall / total_images


def load_openclip_model(model_id: str, device: str):
    clip_model, _, preprocess = open_clip.create_model_and_transforms(model_id)
    tokenizer = open_clip.get_tokenizer(model_id)
    clip_model = clip_model.to(device).eval()
    return clip_model, tokenizer, preprocess


def load_teacher_model(model_id: str, device: str, requested_backend: str):
    backend = resolve_backend(model_id, requested_backend)
    if backend == "open_clip":
        return load_openclip_model(model_id, device)

    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
    hf_model = AutoModel.from_pretrained(model_id, dtype=torch.float32).to(device).eval()

    def preprocess(image):
        encoded = processor(images=image, return_tensors="pt")
        return {
            key: value[0]
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }

    def tokenizer(texts: list[str]):
        max_length = None
        if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "model_max_length"):
            candidate = int(processor.tokenizer.model_max_length)
            if candidate > 0 and candidate < 100000:
                max_length = candidate
        if max_length is None and hasattr(hf_model, "config") and hasattr(hf_model.config, "text_config"):
            text_config = getattr(hf_model.config, "text_config")
            if hasattr(text_config, "max_position_embeddings"):
                candidate = int(getattr(text_config, "max_position_embeddings"))
                if candidate > 0 and candidate < 100000:
                    max_length = candidate
        kwargs = {
            "text": texts,
            "return_tensors": "pt",
            "padding": "max_length" if max_length is not None else True,
            "truncation": True,
        }
        if max_length is not None:
            kwargs["max_length"] = max_length
        return processor(**kwargs)

    return TransformersEvalWrapper(hf_model).to(device).eval(), tokenizer, preprocess


def load_student_model(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    student_model_id = checkpoint.get("student_model_id", "hf-hub:laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K")

    clip_model, _, preprocess = open_clip.create_model_and_transforms(student_model_id)
    tokenizer = open_clip.get_tokenizer(student_model_id)
    clip_model = clip_model.to(device).eval()

    student_model = StudentImageModel(clip_model).to(device)
    load_result = student_model.load_state_dict(checkpoint["student_state_dict"], strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        print(
            "Student checkpoint compatibility: "
            f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
        )

    wrapped = StudentEvalWrapper(clip_model, student_model).to(device).eval()
    return wrapped, tokenizer, preprocess


def main() -> None:
    args = parse_args()
    configure_cuda_runtime(args.device)

    if not Path(args.coco_root).exists():
        raise FileNotFoundError(f"COCO image root not found: {args.coco_root}")
    if not Path(args.coco_ann).exists():
        raise FileNotFoundError(f"COCO annotation file not found: {args.coco_ann}")
    if not Path(args.student_checkpoint).exists():
        raise FileNotFoundError(f"Student checkpoint not found: {args.student_checkpoint}")

    print(f"Using device: {args.device}")
    print(f"COCO root: {args.coco_root}")
    print(f"COCO ann: {args.coco_ann}")
    if args.max_samples is not None:
        print(f"Max samples: {args.max_samples}")

    results = {}

    baseline_model, baseline_tokenizer, baseline_preprocess = load_openclip_model(args.baseline_model_id, args.device)
    results["baseline"] = evaluate_coco_recall_at10(
        model=baseline_model,
        tokenizer=baseline_tokenizer,
        preprocess=baseline_preprocess,
        args=args,
        label="baseline",
    )
    del baseline_model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    teacher_model, teacher_tokenizer, teacher_preprocess = load_teacher_model(
        args.teacher_model_id,
        args.device,
        args.teacher_backend,
    )
    results["teacher"] = evaluate_coco_recall_at10(
         model=teacher_model,
         tokenizer=teacher_tokenizer,
         preprocess=teacher_preprocess,
         args=args,
         label="teacher",
     )
    del teacher_model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    student_model, student_tokenizer, student_preprocess = load_student_model(args.student_checkpoint, args.device)
    results["student"] = evaluate_coco_recall_at10(
        model=student_model,
        tokenizer=student_tokenizer,
        preprocess=student_preprocess,
        args=args,
        label="student",
    )
    del student_model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    print("\nValidation Recall@10")
    print(f"baseline  : {results['baseline']:.6f}")
    print(f"teacher   : {results['teacher']:.6f}")
    print(f"student   : {results['student']:.6f}")
    print("\nJSON summary")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
    

