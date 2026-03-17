import argparse
import json
from pathlib import Path

import onnx
import qai_hub


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile and profile exported ONNX models on QAI Hub.")
    parser.add_argument("--onnx-dir", default="exported_onnx")
    parser.add_argument("--metadata", default="exported_onnx/export_metadata.json")
    parser.add_argument("--image-onnx", default=None, help="Override image ONNX path.")
    parser.add_argument("--text-onnx", default=None, help="Override text ONNX path.")
    parser.add_argument("--skip-text", action="store_true")
    parser.add_argument("--device", default="XR2 Gen 2 (Proxy)")
    parser.add_argument("--max-profiler-iterations", type=int, default=100)
    return parser.parse_args()


def load_export_metadata(metadata_path: Path) -> dict[str, object] | None:
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def resolve_onnx_paths(args: argparse.Namespace) -> tuple[Path, Path | None]:
    metadata = load_export_metadata(Path(args.metadata))

    if args.image_onnx:
        image_path = Path(args.image_onnx)
    elif metadata and metadata.get("image_onnx"):
        image_path = Path(str(metadata["image_onnx"]))
    else:
        image_path = Path(args.onnx_dir) / "student_image_encoder.onnx"

    text_path: Path | None
    if args.skip_text:
        text_path = None
    elif args.text_onnx:
        text_path = Path(args.text_onnx)
    elif metadata and metadata.get("text_onnx"):
        text_path = Path(str(metadata["text_onnx"]))
    else:
        text_path = Path(args.onnx_dir) / "text_encoder.onnx"

    return image_path, text_path


def load_and_validate_onnx(path: Path, label: str) -> onnx.ModelProto:
    if not path.exists():
        raise FileNotFoundError(f"{label} ONNX file not found: {path}")
    print(f"Loading {label} ONNX from {path}...")
    model = onnx.load(path)
    onnx.checker.check_model(model)
    print(f"{label} ONNX model is valid")
    return model


def compile_model(model: onnx.ModelProto, device: qai_hub.Device, input_specs: dict, options: str) -> str:
    compile_job = qai_hub.submit_compile_job(
        model=model,
        device=device,
        input_specs=input_specs,
        options=options,
    )
    compile_job.modify_sharing(add_emails=['lowpowervision@gmail.com'])
    return compile_job.job_id


def profile_compiled_job(job_id: str, device: qai_hub.Device, max_profiler_iterations: int) -> str:
    compiled_model = qai_hub.get_job(job_id).get_target_model()
    profile_job = qai_hub.submit_profile_job(
        model=compiled_model,
        device=device,
        options=f"--max_profiler_iterations {max_profiler_iterations}",
    )
    return profile_job.job_id


def ensure_qaihub_config() -> None:
    config_path = Path.home() / ".qai_hub" / "client.ini"
    if not config_path.exists():
        raise FileNotFoundError(
            "QAI Hub client config not found. Expected: "
            f"{config_path}.\n"
            "Please configure your API key first: https://aihub.qualcomm.com/get-started"
        )


def main() -> None:
    args = parse_args()
    ensure_qaihub_config()
    onnx_dir = Path(args.onnx_dir)
    if not onnx_dir.exists():
        raise FileNotFoundError(f"ONNX directory not found: {onnx_dir}. Run export_onnx.py first.")

    image_path, text_path = resolve_onnx_paths(args)
    target_device = qai_hub.Device(args.device)

    compile_options = "--target_runtime precompiled_qnn_onnx --truncate_64bit_io"

    image_model = load_and_validate_onnx(image_path, "Image")
    print("\nSubmitting image compile job...")
    image_compile_id = compile_model(
        model=image_model,
        device=target_device,
        input_specs={"image": (1, 3, 224, 224)},
        options=compile_options,
    )
    print(f"Image compile job ID: {image_compile_id}")

    text_compile_id = None
    if text_path is not None:
        text_model = load_and_validate_onnx(text_path, "Text")
        print("\nSubmitting text compile job...")
        text_compile_id = compile_model(
            model=text_model,
            device=target_device,
            input_specs={"text_input": ((1, 77), "int32")},
            options=compile_options,
        )
        print(f"Text compile job ID: {text_compile_id}")

    print("\nSubmitting profile jobs...")
    image_profile_id = profile_compiled_job(
        job_id=image_compile_id,
        device=target_device,
        max_profiler_iterations=args.max_profiler_iterations,
    )
    print(f"Image profile job ID: {image_profile_id}")

    text_profile_id = None
    if text_compile_id is not None:
        text_profile_id = profile_compiled_job(
            job_id=text_compile_id,
            device=target_device,
            max_profiler_iterations=args.max_profiler_iterations,
        )
        print(f"Text profile job ID: {text_profile_id}")

    summary = {
        "image_onnx": str(image_path),
        "text_onnx": str(text_path) if text_path is not None else None,
        "image_compile_job_id": image_compile_id,
        "text_compile_job_id": text_compile_id,
        "image_profile_job_id": image_profile_id,
        "text_profile_job_id": text_profile_id,
    }
    print("\nJob summary")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
