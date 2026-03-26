from __future__ import annotations

import argparse
import copy
import os
import time
from contextlib import nullcontext
from pathlib import Path

import open_clip
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
)
from tqdm.auto import tqdm

from utils.track1_utils import load_track1_image_table, save_image_embedding_cache

try:
    import cv2
except ImportError:
    cv2 = None


DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


NUMPY_DTYPE_MAP: dict[str, str] = {
    "float16": "float16",
    "float32": "float32",
}


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

    for candidate_name in ("image_embeds", "pooler_output"):
        if hasattr(outputs, candidate_name):
            candidate = getattr(outputs, candidate_name)
            if isinstance(candidate, torch.Tensor):
                return candidate
    if isinstance(outputs, torch.Tensor):
        return outputs
    raise RuntimeError("Unable to extract image features from transformers model output")


def build_transformers_image_preprocess(processor) -> Compose:
    image_processor = getattr(processor, "image_processor", processor)

    size_cfg = getattr(image_processor, "size", {}) or {}
    crop_cfg = getattr(image_processor, "crop_size", {}) or {}
    image_mean = getattr(image_processor, "image_mean", [0.5, 0.5, 0.5])
    image_std = getattr(image_processor, "image_std", [0.5, 0.5, 0.5])

    if isinstance(size_cfg, dict):
        resize_h = int(size_cfg.get("height", size_cfg.get("shortest_edge", 384)))
        resize_w = int(size_cfg.get("width", resize_h))
    else:
        resize_h = int(size_cfg)
        resize_w = int(size_cfg)

    if isinstance(crop_cfg, dict):
        crop_h = int(crop_cfg.get("height", resize_h))
        crop_w = int(crop_cfg.get("width", resize_w))
    else:
        crop_h = int(crop_cfg) if crop_cfg else resize_h
        crop_w = int(crop_cfg) if crop_cfg else resize_w

    return Compose(
        [
            Resize((resize_h, resize_w), interpolation=InterpolationMode.BICUBIC),
            CenterCrop((crop_h, crop_w)),
            ToTensor(),
            Normalize(mean=image_mean, std=image_std),
        ]
    )


class Track1ImageDataset(Dataset):
    def __init__(
        self,
        image_names: list[str],
        image_paths: list[str],
        preprocess,
        *,
        strict_images: bool,
        decoder: str,
    ):
        self.image_names = image_names
        self.image_paths = image_paths
        self.preprocess = preprocess
        self.strict_images = strict_images
        self.decoder = decoder

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_name = self.image_names[index]
        image_path = self.image_paths[index]
        try:
            if self.decoder == "opencv":
                image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if image_bgr is None:
                    raise OSError(f"Failed to decode image with OpenCV: {image_path}")
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image_rgb)
            else:
                with Image.open(image_path) as image_obj:
                    image = image_obj.convert("RGB")

            pixel_values = self.preprocess(image)
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


def resolve_image_decoder(requested_decoder: str) -> str:
    if requested_decoder == "auto":
        return "opencv" if cv2 is not None else "pil"
    if requested_decoder == "opencv" and cv2 is None:
        raise RuntimeError("OpenCV decoder requested, but cv2 is not installed.")
    return requested_decoder


def resolve_model_dtype(requested_dtype: str, *, device: str) -> torch.dtype | None:
    if requested_dtype == "auto":
        if device.startswith("cuda"):
            return torch.float16
        return None
    return DTYPE_MAP[requested_dtype]


def resolve_devices(*, device: str, devices: str | None) -> list[str]:
    raw = devices if devices else device
    parsed = [entry.strip() for entry in raw.split(",") if entry.strip()]
    if not parsed:
        raise ValueError("No valid devices provided.")
    return parsed


def _infer_spatial_shape_from_preprocess(preprocess, default_hw: int = 512) -> tuple[int, int]:
    transforms = getattr(preprocess, "transforms", None)
    if not transforms:
        return default_hw, default_hw

    crop_size: tuple[int, int] | None = None
    resize_size: tuple[int, int] | None = None

    for transform in transforms:
        name = type(transform).__name__
        if name == "CenterCrop":
            size = getattr(transform, "size", None)
            if isinstance(size, tuple):
                crop_size = (int(size[0]), int(size[1]))
            elif isinstance(size, int):
                crop_size = (size, size)
        elif name == "Resize":
            size = getattr(transform, "size", None)
            if isinstance(size, tuple):
                resize_size = (int(size[0]), int(size[1]))
            elif isinstance(size, int):
                resize_size = (size, size)

    if crop_size is not None:
        return crop_size
    if resize_size is not None:
        return resize_size
    return default_hw, default_hw


def tune_loader_parallelism(
    *,
    requested_workers: int,
    requested_prefetch: int,
    batch_size: int,
    preprocess,
    max_loader_buffer_gb: float,
) -> tuple[int, int, float]:
    if requested_workers <= 0:
        return 0, requested_prefetch, 0.0

    height, width = _infer_spatial_shape_from_preprocess(preprocess)
    bytes_per_image = 3 * height * width * 4
    bytes_per_batch = max(1, batch_size * bytes_per_image)
    max_buffer_bytes = max(1, int(max_loader_buffer_gb * (1024**3)))
    max_inflight_batches = max(1, max_buffer_bytes // bytes_per_batch)

    effective_workers = max(1, requested_workers)
    effective_prefetch = max(1, requested_prefetch)

    if effective_workers * effective_prefetch > max_inflight_batches:
        effective_prefetch = min(effective_prefetch, max(1, max_inflight_batches // effective_workers))
        if effective_workers * effective_prefetch > max_inflight_batches:
            effective_workers = max(1, max_inflight_batches // effective_prefetch)

    estimated_bytes = effective_workers * effective_prefetch * bytes_per_batch
    estimated_gb = estimated_bytes / (1024**3)
    return effective_workers, effective_prefetch, estimated_gb


def resolve_attn_implementation(requested_attn: str, *, device: str) -> str | None:
    if requested_attn == "auto":
        return "sdpa" if device.startswith("cuda") else None
    if requested_attn == "none":
        return None
    return requested_attn


class CUDAPrefetcher:
    def __init__(self, loader, *, device: str, channels_last: bool):
        self.loader = loader
        self.device = device
        self.channels_last = channels_last
        self.stream = torch.cuda.Stream(device=device)
        self._loader_iter = None
        self._next_batch = None
        self._next_invalid_samples = None

    def __iter__(self):
        self._loader_iter = iter(self.loader)
        self._next_batch = None
        self._next_invalid_samples = None
        self._preload()
        return self

    def __next__(self):
        if self._next_batch is None and self._next_invalid_samples is None:
            raise StopIteration

        torch.cuda.current_stream(device=self.device).wait_stream(self.stream)
        current_batch = self._next_batch
        current_invalid_samples = self._next_invalid_samples
        self._preload()
        return current_batch, current_invalid_samples

    def _preload(self) -> None:
        try:
            packed_batch, invalid_samples = next(self._loader_iter)
        except StopIteration:
            self._next_batch = None
            self._next_invalid_samples = None
            return

        if packed_batch is None:
            self._next_batch = None
            self._next_invalid_samples = invalid_samples
            return

        with torch.cuda.stream(self.stream):
            pixel_values = packed_batch["pixel_values"].to(self.device, non_blocking=True)
            if self.channels_last:
                pixel_values = pixel_values.contiguous(memory_format=torch.channels_last)

        self._next_batch = {
            "pixel_values": pixel_values,
            "image_names": packed_batch["image_names"],
        }
        self._next_invalid_samples = invalid_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute teacher image embeddings for Track1.")
    parser.add_argument("--img-list", default="dataset/img_list.csv")
    parser.add_argument("--image-folder", default="dataset/images")
    parser.add_argument(
        "--model-id",
        default="google/siglip2-so400m-patch16-512",
        help="Teacher model identifier (OpenCLIP or Hugging Face transformers).",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "open_clip", "transformers"],
        default="auto",
        help="Teacher backend. auto chooses transformers for google/siglip* models.",
    )
    parser.add_argument("--output", default="artifacts/teacher_image_embeddings.npz")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="If >0, only encode the first N images from --img-list for quick profiling.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument(
        "--max-loader-buffer-gb",
        type=float,
        default=12.0,
        help="Cap estimated DataLoader in-flight host buffer (workers*prefetch*batch) to avoid OOM kills.",
    )
    parser.add_argument(
        "--decoder",
        choices=["auto", "pil", "opencv"],
        default="auto",
        help="Image decoder backend. auto prefers OpenCV when available.",
    )
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument(
        "--model-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Model weights dtype for transformers backend. auto uses float16 on CUDA.",
    )
    parser.add_argument(
        "--output-dtype",
        choices=["float16", "float32"],
        default="float32",
        help="Output embedding dtype saved to NPZ.",
    )
    parser.add_argument(
        "--attn-impl",
        choices=["auto", "none", "eager", "sdpa"],
        default="auto",
        help="Transformers attention implementation. auto uses sdpa on CUDA.",
    )
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Enable torch.compile for transformers backend to improve steady-state throughput.",
    )
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="default",
        help="torch.compile mode when --compile-model is enabled.",
    )
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=0,
        help="If >0, print average data-wait/compute/cpu-copy timings for the first N batches.",
    )
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
    parser.add_argument(
        "--devices",
        help="Comma-separated device list for parallel inference, e.g. cuda:0,cuda:1. Overrides --device.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    devices = resolve_devices(device=args.device, devices=args.devices)
    primary_device = devices[0]
    is_cuda = primary_device.startswith("cuda")
    backend = resolve_backend(args.model_id, args.backend)
    configure_cuda_runtime(device=primary_device, allow_tf32=not args.no_tf32)
    num_workers = resolve_num_workers(args.num_workers, primary_device)
    amp_enabled = not args.no_amp
    decoder = resolve_image_decoder(args.decoder)
    model_dtype = resolve_model_dtype(args.model_dtype, device=primary_device)
    attn_impl = resolve_attn_implementation(args.attn_impl, device=primary_device)
    image_table = load_track1_image_table(args.img_list, args.image_folder)
    if image_table.empty:
        raise ValueError(f"No images found in {args.img_list}")
    if args.max_samples > 0:
        image_table = image_table.head(args.max_samples).reset_index(drop=True)
        print(f"Max samples enabled: {len(image_table)}")

    outputs: list[torch.Tensor] = []
    image_names = image_table["image_name"].tolist()
    image_paths = image_table["image_path"].tolist()

    print(f"Image loader workers: {num_workers}")
    if is_cuda:
        print(f"AMP enabled: {amp_enabled}")
        print(f"Channels-last: {args.channels_last}")
        print(f"TF32 enabled: {not args.no_tf32}")
        print(f"Devices: {', '.join(devices)}")
    print(f"Image decoder: {decoder}")
    if backend == "transformers":
        print(f"Model dtype: {str(model_dtype).replace('torch.', '') if model_dtype is not None else 'default'}")
        print(f"Attention impl: {attn_impl if attn_impl is not None else 'default'}")
        if args.compile_model:
            print(f"torch.compile: enabled ({args.compile_mode})")
    print(f"Output dtype: {args.output_dtype}")

    kept_image_names: list[str] = []
    failed_samples: list[tuple[str, str]] = []

    if backend == "open_clip":
        base_model, _, preprocess = open_clip.create_model_and_transforms(args.model_id)
        model_by_device: dict[str, torch.nn.Module] = {}
        for device_name in devices:
            model = copy.deepcopy(base_model).to(device_name).eval()
            if args.channels_last and device_name.startswith("cuda"):
                model = model.to(memory_format=torch.channels_last)
            model_by_device[device_name] = model

        dataset = Track1ImageDataset(
            image_names,
            image_paths,
            preprocess,
            strict_images=args.strict_images,
            decoder=decoder,
        )
        loader_workers, loader_prefetch, est_loader_gb = tune_loader_parallelism(
            requested_workers=num_workers,
            requested_prefetch=args.prefetch_factor,
            batch_size=args.batch_size,
            preprocess=preprocess,
            max_loader_buffer_gb=args.max_loader_buffer_gb,
        )
        if num_workers > 0:
            print(
                f"Loader effective workers/prefetch: {loader_workers}/{loader_prefetch} "
                f"(requested {num_workers}/{args.prefetch_factor}, est in-flight {est_loader_gb:.2f} GB)"
            )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=loader_workers,
            pin_memory=is_cuda,
            persistent_workers=loader_workers > 0,
            prefetch_factor=loader_prefetch if loader_workers > 0 else None,
            collate_fn=collate_teacher_batch,
        )

        with torch.inference_mode():
            use_prefetcher = is_cuda and len(devices) == 1
            batch_iter = (
                CUDAPrefetcher(loader, device=primary_device, channels_last=args.channels_last)
                if use_prefetcher
                else loader
            )
            batch_iter = iter(batch_iter)
            total_valid_images = 0
            start_time = time.perf_counter()
            data_wait_total = 0.0
            compute_total = 0.0
            cpu_copy_total = 0.0
            profiled_batches = 0
            progress = tqdm(total=len(loader), desc="Encoding teacher images", unit="batch", dynamic_ncols=True)
            while True:
                wait_start = time.perf_counter()
                try:
                    packed_batch, invalid_samples = next(batch_iter)
                except StopIteration:
                    break
                data_wait = time.perf_counter() - wait_start
                progress.update(1)
                for sample in invalid_samples:
                    failed_samples.append((str(sample["image_name"]), str(sample["error"])))

                if packed_batch is None:
                    continue

                pixel_values = packed_batch["pixel_values"]
                sample_count = int(pixel_values.shape[0])
                chunk_sizes = [sample_count // len(devices)] * len(devices)
                for idx in range(sample_count % len(devices)):
                    chunk_sizes[idx] += 1
                pixel_chunks = list(torch.split(pixel_values, chunk_sizes, dim=0))

                compute_start = time.perf_counter()
                batch_outputs: list[torch.Tensor] = []
                for device_name, chunk in zip(devices, pixel_chunks):
                    if chunk.numel() == 0:
                        continue
                    chunk = chunk.to(device_name, non_blocking=is_cuda)
                    if args.channels_last and device_name.startswith("cuda"):
                        chunk = chunk.contiguous(memory_format=torch.channels_last)
                    autocast_context = (
                        torch.autocast(device_type="cuda", enabled=amp_enabled and device_name.startswith("cuda"))
                        if device_name.startswith("cuda")
                        else nullcontext()
                    )
                    with autocast_context:
                        features = model_by_device[device_name].encode_image(chunk)
                        features = torch.nn.functional.normalize(features, dim=-1)
                    batch_outputs.append(features)
                should_profile = args.profile_steps > 0 and profiled_batches < args.profile_steps
                if should_profile and is_cuda:
                    for device_name in devices:
                        if device_name.startswith("cuda"):
                            torch.cuda.synchronize(device=device_name)
                compute_end = time.perf_counter()

                cpu_copy_start = time.perf_counter()
                outputs.extend(batch_outputs)
                cpu_copy_end = time.perf_counter()

                kept_image_names.extend(packed_batch["image_names"])
                total_valid_images += len(packed_batch["image_names"])
                if should_profile:
                    data_wait_total += data_wait
                    compute_total += compute_end - compute_start
                    cpu_copy_total += cpu_copy_end - cpu_copy_start
                    profiled_batches += 1
            progress.close()
            elapsed = max(1e-6, time.perf_counter() - start_time)
            print(f"Encoding throughput: {total_valid_images / elapsed:.2f} images/s")
            if profiled_batches > 0:
                print(
                    "Profile avg per batch "
                    f"(first {profiled_batches}): "
                    f"data_wait={data_wait_total / profiled_batches:.4f}s, "
                    f"compute={compute_total / profiled_batches:.4f}s, "
                    f"queue={cpu_copy_total / profiled_batches:.6f}s"
                )
    else:
        from transformers import AutoModel, AutoProcessor

        load_kwargs = {"dtype": model_dtype}
        if attn_impl is not None:
            load_kwargs["attn_implementation"] = attn_impl
        base_model = AutoModel.from_pretrained(args.model_id, **load_kwargs)
        model_by_device: dict[str, torch.nn.Module] = {}
        for device_name in devices:
            model = copy.deepcopy(base_model).to(device_name).eval()
            if args.channels_last and device_name.startswith("cuda"):
                model = model.to(memory_format=torch.channels_last)
            if args.compile_model:
                model = torch.compile(model, mode=args.compile_mode)
            model_by_device[device_name] = model
        processor = AutoProcessor.from_pretrained(args.model_id, use_fast=False)
        preprocess_transformers = build_transformers_image_preprocess(processor)

        dataset = Track1ImageDataset(
            image_names,
            image_paths,
            preprocess_transformers,
            strict_images=args.strict_images,
            decoder=decoder,
        )
        loader_workers, loader_prefetch, est_loader_gb = tune_loader_parallelism(
            requested_workers=num_workers,
            requested_prefetch=args.prefetch_factor,
            batch_size=args.batch_size,
            preprocess=preprocess_transformers,
            max_loader_buffer_gb=args.max_loader_buffer_gb,
        )
        if num_workers > 0:
            print(
                f"Loader effective workers/prefetch: {loader_workers}/{loader_prefetch} "
                f"(requested {num_workers}/{args.prefetch_factor}, est in-flight {est_loader_gb:.2f} GB)"
            )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=loader_workers,
            pin_memory=is_cuda,
            persistent_workers=loader_workers > 0,
            prefetch_factor=loader_prefetch if loader_workers > 0 else None,
            collate_fn=collate_teacher_batch,
        )

        with torch.inference_mode():
            use_prefetcher = is_cuda and len(devices) == 1
            batch_iter = (
                CUDAPrefetcher(loader, device=primary_device, channels_last=args.channels_last)
                if use_prefetcher
                else loader
            )
            batch_iter = iter(batch_iter)
            total_valid_images = 0
            start_time = time.perf_counter()
            data_wait_total = 0.0
            compute_total = 0.0
            cpu_copy_total = 0.0
            profiled_batches = 0
            progress = tqdm(total=len(loader), desc="Encoding teacher images", unit="batch", dynamic_ncols=True)
            while True:
                wait_start = time.perf_counter()
                try:
                    packed_batch, invalid_samples = next(batch_iter)
                except StopIteration:
                    break
                data_wait = time.perf_counter() - wait_start
                progress.update(1)
                for sample in invalid_samples:
                    failed_samples.append((str(sample["image_name"]), str(sample["error"])))

                if packed_batch is None:
                    continue

                pixel_values = packed_batch["pixel_values"]
                sample_count = int(pixel_values.shape[0])
                chunk_sizes = [sample_count // len(devices)] * len(devices)
                for idx in range(sample_count % len(devices)):
                    chunk_sizes[idx] += 1
                pixel_chunks = list(torch.split(pixel_values, chunk_sizes, dim=0))

                compute_start = time.perf_counter()
                batch_outputs: list[torch.Tensor] = []
                for device_name, chunk in zip(devices, pixel_chunks):
                    if chunk.numel() == 0:
                        continue
                    chunk = chunk.to(device_name, non_blocking=is_cuda)
                    if args.channels_last and device_name.startswith("cuda"):
                        chunk = chunk.contiguous(memory_format=torch.channels_last)
                    model_inputs = {"pixel_values": chunk}
                    autocast_context = (
                        torch.autocast(device_type="cuda", enabled=amp_enabled and device_name.startswith("cuda"))
                        if device_name.startswith("cuda")
                        else nullcontext()
                    )
                    with autocast_context:
                        features = extract_transformers_image_features(model_by_device[device_name], model_inputs)
                        features = torch.nn.functional.normalize(features, dim=-1)
                    batch_outputs.append(features)
                should_profile = args.profile_steps > 0 and profiled_batches < args.profile_steps
                if should_profile and is_cuda:
                    for device_name in devices:
                        if device_name.startswith("cuda"):
                            torch.cuda.synchronize(device=device_name)
                compute_end = time.perf_counter()

                cpu_copy_start = time.perf_counter()
                outputs.extend(batch_outputs)
                cpu_copy_end = time.perf_counter()
                kept_image_names.extend(packed_batch["image_names"])
                total_valid_images += len(packed_batch["image_names"])
                if should_profile:
                    data_wait_total += data_wait
                    compute_total += compute_end - compute_start
                    cpu_copy_total += cpu_copy_end - cpu_copy_start
                    profiled_batches += 1
            progress.close()
            elapsed = max(1e-6, time.perf_counter() - start_time)
            print(f"Encoding throughput: {total_valid_images / elapsed:.2f} images/s")
            if profiled_batches > 0:
                print(
                    "Profile avg per batch "
                    f"(first {profiled_batches}): "
                    f"data_wait={data_wait_total / profiled_batches:.4f}s, "
                    f"compute={compute_total / profiled_batches:.4f}s, "
                    f"queue={cpu_copy_total / profiled_batches:.6f}s"
                )

    if not outputs:
        raise RuntimeError("No valid images were encoded. Check --img-list and image file integrity.")

    if is_cuda:
        for device_name in devices:
            if device_name.startswith("cuda"):
                torch.cuda.synchronize(device=device_name)
    output_chunks = [chunk.cpu() if chunk.device.type == "cuda" else chunk for chunk in outputs]
    embeddings_tensor = torch.cat(output_chunks, dim=0)
    embeddings = embeddings_tensor.numpy()
    if str(embeddings.dtype) != NUMPY_DTYPE_MAP[args.output_dtype]:
        embeddings = embeddings.astype(NUMPY_DTYPE_MAP[args.output_dtype], copy=False)
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