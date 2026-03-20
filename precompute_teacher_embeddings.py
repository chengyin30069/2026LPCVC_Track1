from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
from pathlib import Path

import open_clip
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from utils.track1_utils import load_track1_image_table, save_image_embedding_cache


class Track1ImageDataset(Dataset):
    def __init__(
        self,
        image_names: list[str],
        image_paths: list[str],
        preprocess,
        *,
        strict_images: bool,
    ):
        self.image_names = image_names
        self.image_paths = image_paths
        self.preprocess = preprocess
        self.strict_images = strict_images

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_name = self.image_names[index]
        image_path = self.image_paths[index]
        try:
            with Image.open(image_path) as image:
                pixel_values = self.preprocess(image.convert("RGB"))
            return {
                "ok": True,
                "image_name": image_name,
                "pixel_values": pixel_values,
                "error": "",
            }
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            if self.strict_images:
                raise
            return {
                "ok": False,
                "image_name": image_name,
                "pixel_values": None,
                "error": f"{type(exc).__name__}: {exc}",
            }


def collate_teacher_batch(batch: list[dict[str, object]]) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    valid_samples = [sample for sample in batch if bool(sample["ok"])]
    invalid_samples = [sample for sample in batch if not bool(sample["ok"])]

    if not valid_samples:
        return None, invalid_samples

    return {
        "pixel_values": torch.stack([sample["pixel_values"] for sample in valid_samples], dim=0),
        "image_names": [str(sample["image_name"]) for sample in valid_samples],
    }, invalid_samples


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute teacher image embeddings for Track1.")
    parser.add_argument("--img-list", default="dataset/img_list.csv")
    parser.add_argument("--image-folder", default="dataset/images")
    parser.add_argument(
        "--model-id",
        default="hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        help="Teacher OpenCLIP model identifier.",
    )
    parser.add_argument("--output", default="artifacts/teacher_image_embeddings.npz")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument(
        "--strict-images",
        action="store_true",
        help="Fail immediately when an unreadable image is encountered.",
    )
    parser.add_argument(
        "--failed-images-log",
        help="Optional path to save a TSV of skipped images and decode errors.",
    )
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_cuda_runtime(device=args.device, allow_tf32=not args.no_tf32)
    num_workers = resolve_num_workers(args.num_workers, args.device)
    amp_enabled = not args.no_amp
    image_table = load_track1_image_table(args.img_list, args.image_folder)
    if image_table.empty:
        raise ValueError(f"No images found in {args.img_list}")

    model, _, preprocess = open_clip.create_model_and_transforms(args.model_id)
    model = model.to(args.device).eval()
    if args.channels_last and args.device.startswith("cuda"):
        model = model.to(memory_format=torch.channels_last)

    outputs: list[torch.Tensor] = []
    image_names = image_table["image_name"].tolist()
    image_paths = image_table["image_path"].tolist()
    dataset = Track1ImageDataset(
        image_names,
        image_paths,
        preprocess,
        strict_images=args.strict_images,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=num_workers > 0,
        prefetch_factor=args.prefetch_factor if num_workers > 0 else None,
        collate_fn=collate_teacher_batch,
    )

    print(f"Image loader workers: {num_workers}")
    if args.device.startswith("cuda"):
        print(f"AMP enabled: {amp_enabled}")
        print(f"Channels-last: {args.channels_last}")
        print(f"TF32 enabled: {not args.no_tf32}")

    kept_image_names: list[str] = []
    failed_samples: list[tuple[str, str]] = []

    with torch.inference_mode():
        for packed_batch, invalid_samples in tqdm(
            loader,
            total=len(loader),
            desc="Encoding teacher images",
            unit="batch",
            dynamic_ncols=True,
        ):
            for sample in invalid_samples:
                failed_samples.append((str(sample["image_name"]), str(sample["error"])))

            if packed_batch is None:
                continue

            pixel_values = packed_batch["pixel_values"]
            pixel_values = pixel_values.to(args.device, non_blocking=True)
            if args.channels_last and args.device.startswith("cuda"):
                pixel_values = pixel_values.contiguous(memory_format=torch.channels_last)
            autocast_context = (
                torch.autocast(device_type="cuda", enabled=amp_enabled and args.device.startswith("cuda"))
                if args.device.startswith("cuda")
                else nullcontext()
            )
            with autocast_context:
                features = model.encode_image(pixel_values)
                features = torch.nn.functional.normalize(features, dim=-1)
            outputs.append(features.float().cpu())
            kept_image_names.extend(packed_batch["image_names"])

    if not outputs:
        raise RuntimeError("No valid images were encoded. Check --img-list and image file integrity.")

    embeddings = torch.cat(outputs, dim=0).numpy()
    output_path = Path(args.output)
    save_image_embedding_cache(
        output_path,
        image_names=kept_image_names,
        embeddings=embeddings,
        model_name=args.model_id,
    )
    print(f"Saved {len(kept_image_names)} teacher image embeddings to {output_path}")

    skipped_count = len(failed_samples)
    if skipped_count > 0:
        failed_log_path = Path(args.failed_images_log) if args.failed_images_log else output_path.with_suffix(".skipped.tsv")
        failed_log_path.parent.mkdir(parents=True, exist_ok=True)
        with failed_log_path.open("w", encoding="utf-8") as handle:
            handle.write("image_name\terror\n")
            for image_name, error in failed_samples:
                escaped_error = error.replace("\n", " ").replace("\r", " ")
                handle.write(f"{image_name}\t{escaped_error}\n")
        print(f"Skipped unreadable images: {skipped_count}")
        print(f"Saved skipped-image log to {failed_log_path}")


if __name__ == "__main__":
    main()