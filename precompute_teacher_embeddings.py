import argparse
import os
from contextlib import nullcontext
from pathlib import Path

import open_clip
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from utils.track1_utils import load_track1_image_table, save_image_embedding_cache


class Track1ImageDataset(Dataset):
    def __init__(self, image_paths: list[str], preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.image_paths[index]) as image:
            return self.preprocess(image.convert("RGB"))


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
        default="hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
        help="Teacher OpenCLIP model identifier.",
    )
    parser.add_argument("--output", default="artifacts/teacher_image_embeddings.npz")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    dataset = Track1ImageDataset(image_paths, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=num_workers > 0,
        prefetch_factor=args.prefetch_factor if num_workers > 0 else None,
    )

    print(f"Image loader workers: {num_workers}")
    if args.device.startswith("cuda"):
        print(f"AMP enabled: {amp_enabled}")
        print(f"Channels-last: {args.channels_last}")
        print(f"TF32 enabled: {not args.no_tf32}")

    with torch.inference_mode():
        for pixel_values in tqdm(
            loader,
            total=len(loader),
            desc="Encoding teacher images",
            unit="batch",
            dynamic_ncols=True,
        ):
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

    embeddings = torch.cat(outputs, dim=0).numpy()
    output_path = Path(args.output)
    save_image_embedding_cache(
        output_path,
        image_names=image_names,
        embeddings=embeddings,
        model_name=args.model_id,
    )
    print(f"Saved {len(image_names)} teacher image embeddings to {output_path}")


if __name__ == "__main__":
    main()