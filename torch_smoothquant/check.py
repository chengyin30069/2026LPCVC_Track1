from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}


def default_qdq_path() -> Path:
    return Path(__file__).resolve().parent / "exported_onnx" / "image_encoder_qdq.onnx"


def default_original_path() -> Path:
    return Path(__file__).resolve().parent / "exported_onnx" / "smoothquant_image_encoder.onnx"


def default_image_dir() -> Path | None:
    candidates = [
        ROOT_DIR / "dataset" / "images",
        ROOT_DIR / "2026LPCVC_Track1" / "dataset_sample" / "images",
        ROOT_DIR / "quant_clip_test" / "dataset" / "images",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def find_images(image_dir: Path, limit: int | None) -> list[Path]:
    images = sorted(
        path for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )
    if limit is not None:
        images = images[:limit]
    return images


def session_input_hw(session: ort.InferenceSession) -> tuple[int, int]:
    input_shape = session.get_inputs()[0].shape
    height = int(input_shape[2])
    width = int(input_shape[3])
    return height, width


def preprocess_image(image_path: Path, height: int, width: int) -> np.ndarray:
    image = Image.open(image_path).convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    image_np = np.transpose(image_np, (2, 0, 1))
    return image_np[np.newaxis, ...].astype(np.float32)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)
    return x / denom


def cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs = lhs.astype(np.float32).reshape(-1)
    rhs = rhs.astype(np.float32).reshape(-1)
    denom = (np.linalg.norm(lhs) * np.linalg.norm(rhs)) + 1e-8
    return float(np.dot(lhs, rhs) / denom)


def max_abs_error(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.max(np.abs(lhs.astype(np.float32) - rhs.astype(np.float32))))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare cosine similarity between a QDQ ONNX image encoder and the original ONNX image encoder.",
    )
    parser.add_argument(
        "--qdq-path",
        type=Path,
        default=default_qdq_path(),
        help="Path to the QDQ ONNX model.",
    )
    parser.add_argument(
        "--original-path",
        type=Path,
        default=default_original_path(),
        help="Path to the original ONNX model used before QDQ conversion.",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=None,
        help="Single image to evaluate.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=default_image_dir(),
        help="Directory of images to evaluate when --image-path is not set.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=16,
        help="Maximum number of images to compare from --image-dir.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Compare raw outputs instead of L2-normalized embeddings.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.qdq_path.is_file():
        raise FileNotFoundError(f"QDQ model not found: {args.qdq_path}")
    if not args.original_path.is_file():
        raise FileNotFoundError(f"Original model not found: {args.original_path}")

    if args.image_path is not None:
        image_paths = [args.image_path]
    else:
        if args.image_dir is None:
            raise ValueError("No image source found. Pass --image-path or --image-dir.")
        image_paths = find_images(args.image_dir, args.limit)

    if not image_paths:
        raise ValueError("No images found to compare.")

    qdq_session = ort.InferenceSession(args.qdq_path.as_posix(), providers=["CPUExecutionProvider"])
    original_session = ort.InferenceSession(args.original_path.as_posix(), providers=["CPUExecutionProvider"])

    qdq_input = qdq_session.get_inputs()[0].name
    original_input = original_session.get_inputs()[0].name
    qdq_output = qdq_session.get_outputs()[0].name
    original_output = original_session.get_outputs()[0].name

    qdq_hw = session_input_hw(qdq_session)
    original_hw = session_input_hw(original_session)
    if qdq_hw != original_hw:
        raise ValueError(f"Input shapes do not match: qdq={qdq_hw}, original={original_hw}")

    print(f"[INFO] QDQ model      : {args.qdq_path}")
    print(f"[INFO] Original model : {args.original_path}")
    print(f"[INFO] Input size     : {qdq_hw[0]}x{qdq_hw[1]}")

    cosines: list[float] = []
    errors: list[float] = []

    for image_path in image_paths:
        pixel_values = preprocess_image(image_path, *qdq_hw)

        qdq_out = qdq_session.run([qdq_output], {qdq_input: pixel_values})[0].astype(np.float32)
        original_out = original_session.run([original_output], {original_input: pixel_values})[0].astype(np.float32)

        if not args.no_normalize:
            qdq_out = l2_normalize(qdq_out)
            original_out = l2_normalize(original_out)

        cos = cosine_similarity(original_out, qdq_out)
        err = max_abs_error(original_out, qdq_out)
        cosines.append(cos)
        errors.append(err)

        print(f"{image_path.name}: cosine={cos:.6f} max_abs_err={err:.6f}")

    print(f"[INFO] Compared {len(image_paths)} image(s)")
    print(f"[INFO] Mean cosine     : {sum(cosines) / len(cosines):.6f}")
    print(f"[INFO] Min cosine      : {min(cosines):.6f}")
    print(f"[INFO] Mean max abs err: {sum(errors) / len(errors):.6f}")


if __name__ == "__main__":
    main()
