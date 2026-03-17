import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CocoCaptions
from tqdm.auto import tqdm

from utils.student_model import StudentImageModel


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark baseline, teacher, and student-v3 on COCO Recall@10.")
    parser.add_argument("--coco-root", default="coco2017/val2017")
    parser.add_argument("--coco-ann", default="coco2017/annotations/captions_val2017.json")
    parser.add_argument("--baseline-model-id", default="hf-hub:laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K")
    parser.add_argument("--teacher-model-id", default="hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K")
    parser.add_argument("--student-checkpoint", default="artifacts/student_distill_v4/best_loss_checkpoint.pt")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    return torch.stack(images, dim=0), captions


def load_coco_dataset(coco_root: str, coco_ann: str, preprocess, max_samples: int | None):
    dataset = CocoCaptions(root=coco_root, annFile=coco_ann, transform=preprocess)
    if max_samples is not None:
        max_samples = min(max_samples, len(dataset))
        dataset = Subset(dataset, list(range(max_samples)))
    return dataset


def encode_image_batch(
    model,
    image_batch: torch.Tensor,
    *,
    feature_batch_size: int,
    device: str,
    amp_enabled: bool,
    channels_last: bool,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for start in range(0, image_batch.shape[0], feature_batch_size):
        chunk = image_batch[start:start + feature_batch_size].to(device, non_blocking=True)
        if channels_last and device.startswith("cuda"):
            chunk = chunk.contiguous(memory_format=torch.channels_last)
        autocast_context = (
            torch.autocast(device_type="cuda", enabled=amp_enabled and device.startswith("cuda"))
            if device.startswith("cuda")
            else nullcontext()
        )
        with autocast_context:
            features = model.encode_image(chunk)
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
        token_batch = tokenizer(texts[start:start + feature_batch_size]).to(device)
        autocast_context = (
            torch.autocast(device_type="cuda", enabled=amp_enabled and device.startswith("cuda"))
            if device.startswith("cuda")
            else nullcontext()
        )
        with autocast_context:
            text_features = model.encode_text(token_batch)
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

    total_recall = 0.0
    total_images = 0
    with torch.inference_mode():
        for image_batch, captions_batch in tqdm(loader, desc=f"Evaluating {label}", dynamic_ncols=True):
            image_features = encode_image_batch(
                model,
                image_batch,
                feature_batch_size=args.feature_batch_size,
                device=device,
                amp_enabled=amp_enabled,
                channels_last=args.channels_last,
            )
            text_features, text_owner_indices = encode_text_batch(
                model,
                tokenizer,
                captions_batch,
                feature_batch_size=args.feature_batch_size,
                device=device,
                amp_enabled=amp_enabled,
            )

            if text_features.numel() == 0:
                continue

            similarity = image_features @ text_features.T
            top_k = min(10, text_features.shape[0])
            top_indices = torch.topk(similarity, k=top_k, dim=1).indices

            top_owner_indices = text_owner_indices[top_indices]
            target_owner_indices = torch.arange(image_features.shape[0], device=device).unsqueeze(1)
            matches = (top_owner_indices == target_owner_indices).sum(dim=1).float()
            recall_per_image = matches / float(top_k)

            batch_size = image_features.shape[0]
            total_recall += recall_per_image.sum().item()
            total_images += batch_size

    if total_images == 0:
        raise RuntimeError("No images were evaluated. Check COCO paths and max_samples.")
    return total_recall / total_images


def load_openclip_model(model_id: str, device: str):
    clip_model, _, preprocess = open_clip.create_model_and_transforms(model_id)
    tokenizer = open_clip.get_tokenizer(model_id)
    clip_model = clip_model.to(device).eval()
    return clip_model, tokenizer, preprocess


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

    baseline_model, baseline_tokenizer, baseline_preprocess = load_openclip_model(args.baseline_model_id, args.device)
    teacher_model, teacher_tokenizer, teacher_preprocess = load_openclip_model(args.teacher_model_id, args.device)
    student_model, student_tokenizer, student_preprocess = load_student_model(args.student_checkpoint, args.device)

    results = {}
    results["baseline"] = evaluate_coco_recall_at10(
        model=baseline_model,
        tokenizer=baseline_tokenizer,
        preprocess=baseline_preprocess,
        args=args,
        label="baseline",
    )
    results["teacher"] = evaluate_coco_recall_at10(
        model=teacher_model,
        tokenizer=teacher_tokenizer,
        preprocess=teacher_preprocess,
        args=args,
        label="teacher",
    )
    results["student_v3"] = evaluate_coco_recall_at10(
        model=student_model,
        tokenizer=student_tokenizer,
        preprocess=student_preprocess,
        args=args,
        label="student_v3",
    )

    print("\nValidation Recall@10")
    print(f"baseline  : {results['baseline']:.6f}")
    print(f"teacher   : {results['teacher']:.6f}")
    print(f"student_v3: {results['student_v3']:.6f}")
    print("\nJSON summary")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
    

