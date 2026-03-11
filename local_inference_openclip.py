import argparse
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from utils.student_model import StudentImageModel
from utils.track1_utils import (
    evaluate_track1,
    load_text_embedding_cache,
    load_track1_texts,
)

# Model

class ImageEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        x = self.model.encode_image(pixel_values)
        x = F.normalize(x, dim=-1)
        return x


class TextEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        x = self.model.encode_text(input_ids)
        x = F.normalize(x, dim=-1)
        return x


class ImageFolderDataset(Dataset):
    def __init__(self, image_paths: list[Path], preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
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
    parser = argparse.ArgumentParser(description="Run local OpenCLIP inference for LPCVC Track1.")
    parser.add_argument("--txt-list", default="dataset/txt_list.csv")
    parser.add_argument("--img-list", default="dataset/img_list.csv")
    parser.add_argument("--image-folder", default="dataset/images")
    parser.add_argument(
        "--model-id",
        default="hf-hub:laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K",
    )
    parser.add_argument("--student-checkpoint", help="Optional trained student checkpoint.")
    parser.add_argument("--text-embedding-cache", help="Optional cached text embedding NPZ.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--eval-device", default="auto")
    parser.add_argument("--eval-query-chunk-size", type=int, default=256)
    parser.add_argument("--eval-key-chunk-size", type=int, default=16384)
    return parser.parse_args()


def encode_images(
    image_encoder: nn.Module,
    image_folder: str,
    preprocess,
    batch_size: int,
    device: str,
    num_workers: int,
    prefetch_factor: int,
    channels_last: bool,
    amp_enabled: bool,
) -> np.ndarray:
    image_paths = sorted(
        path
        for path in Path(image_folder).iterdir()
        if path.suffix.lower() in {".jpg", ".png", ".jpeg", ".webp"}
    )

    if not image_paths:
        raise ValueError(f"No images found in {image_folder}")

    dataset = ImageFolderDataset(image_paths, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for pixel_values in tqdm(
            loader,
            total=len(loader),
            desc="Encoding images",
            unit="batch",
            dynamic_ncols=True,
        ):
            pixel_values = pixel_values.to(device, non_blocking=True)
            if channels_last and device.startswith("cuda"):
                pixel_values = pixel_values.contiguous(memory_format=torch.channels_last)
            autocast_context = (
                torch.autocast(device_type="cuda", enabled=amp_enabled and device.startswith("cuda"))
                if device.startswith("cuda")
                else nullcontext()
            )
            with autocast_context:
                outputs.append(image_encoder(pixel_values).float().cpu())

    return torch.cat(outputs, dim=0).numpy()


def encode_texts(
    text_encoder: nn.Module,
    tokenizer,
    texts: list[str],
    batch_size: int,
    device: str,
    amp_enabled: bool,
) -> np.ndarray:
    outputs: list[torch.Tensor] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(texts), batch_size),
            total=total_batches,
            desc="Encoding texts",
            unit="batch",
            dynamic_ncols=True,
        ):
            token_batch = tokenizer(texts[start:start + batch_size]).to(device)
            autocast_context = (
                torch.autocast(device_type="cuda", enabled=amp_enabled and device.startswith("cuda"))
                if device.startswith("cuda")
                else nullcontext()
            )
            with autocast_context:
                outputs.append(text_encoder(token_batch).float().cpu())
    return torch.cat(outputs, dim=0).numpy()


if __name__ == '__main__':
    args = parse_args()
    configure_cuda_runtime(device=args.device, allow_tf32=not args.no_tf32)
    num_workers = resolve_num_workers(args.num_workers, args.device)
    amp_enabled = not args.no_amp
    checkpoint = None
    model_id = args.model_id
    if args.student_checkpoint:
        checkpoint = torch.load(args.student_checkpoint, map_location="cpu")
        model_id = checkpoint.get("student_model_id", model_id)

    model, _, preprocess = open_clip.create_model_and_transforms(model_id)
    tokenizer = open_clip.get_tokenizer(model_id)
    model = model.to(args.device).eval()
    if args.channels_last and args.device.startswith("cuda"):
        model = model.to(memory_format=torch.channels_last)

    text_encoder = TextEncoder(model).eval()
    if checkpoint is not None:
        student_model = StudentImageModel(model).to(args.device)
        load_result = student_model.load_state_dict(checkpoint["student_state_dict"], strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            print(f"Checkpoint compatibility: missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}")
        if args.channels_last and args.device.startswith("cuda"):
            student_model = student_model.to(memory_format=torch.channels_last)
        image_encoder = student_model.eval()
    else:
        image_encoder = ImageEncoder(model).eval()

    print(f"Image loader workers: {num_workers}")
    if args.device.startswith("cuda"):
        print(f"AMP enabled: {amp_enabled}")
        print(f"Channels-last: {args.channels_last}")
        print(f"TF32 enabled: {not args.no_tf32}")

    image_output = encode_images(
        image_encoder,
        args.image_folder,
        preprocess,
        args.batch_size,
        args.device,
        num_workers,
        args.prefetch_factor,
        args.channels_last,
        amp_enabled,
    )

    if args.text_embedding_cache:
        cache = load_text_embedding_cache(args.text_embedding_cache)
        text_ids, _ = load_track1_texts(args.txt_list)
        cached_text_ids = np.asarray(cache["text_ids"], dtype=np.int64)
        if not np.array_equal(cached_text_ids, text_ids):
            raise ValueError("Cached text embeddings do not match the current txt_list.csv")
        text_output = np.asarray(cache["embeddings"], dtype=np.float32)
    else:
        _, texts = load_track1_texts(args.txt_list)
        text_output = encode_texts(
            text_encoder,
            tokenizer,
            texts,
            args.batch_size,
            args.device,
            amp_enabled,
        )

    result = evaluate_track1(
        image_output,
        text_output,
        args.txt_list,
        args.img_list,
        device=args.eval_device if args.eval_device != "auto" else args.device,
        query_chunk_size=args.eval_query_chunk_size,
        key_chunk_size=args.eval_key_chunk_size,
        show_progress=True,
    )
    print(result)