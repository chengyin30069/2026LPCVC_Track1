import argparse
import json
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F
from torch import nn

from utils.student_model import StudentImageModel


class StudentImageEncoderWrapper(nn.Module):
    def __init__(self, student_model: StudentImageModel):
        super().__init__()
        self.student_model = student_model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.student_model(images)


class TextEncoderWrapper(nn.Module):
    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Keep exported input dtype as int32 while converting to int64 for model internals.
        token_ids = token_ids.to(torch.int64)
        return F.normalize(self.clip_model.encode_text(token_ids), dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export student checkpoint to ONNX.")
    parser.add_argument(
        "--student-checkpoint",
        default="artifacts/student_distill_v4/best_loss_checkpoint.pt",
        help="Path to trained student checkpoint.",
    )
    parser.add_argument("--output-dir", default="exported_onnx")
    parser.add_argument("--image-onnx-name", default="student_image_encoder.onnx")
    parser.add_argument("--text-onnx-name", default="text_encoder.onnx")
    parser.add_argument("--opset-version", type=int, default=18)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--context-length", type=int, default=77)
    parser.add_argument("--vocab-size", type=int, default=49408)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--skip-text-encoder", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def export_onnx_model(
    model: nn.Module,
    dummy_input: torch.Tensor,
    output_path: Path,
    input_names: list[str],
    output_names: list[str],
    opset_version: int,
) -> None:
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        do_constant_folding=True,
        dynamic_axes=None,
        verbose=False,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,
        dynamo=True,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    checkpoint_path = Path(args.student_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Student checkpoint not found: {checkpoint_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving ONNX files to directory: {output_dir.resolve()}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    student_model_id = checkpoint.get(
        "student_model_id",
        "hf-hub:laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K",
    )
    print(f"Loading OpenCLIP model: {student_model_id}")

    clip_model, _, _ = open_clip.create_model_and_transforms(student_model_id)
    clip_model = clip_model.to(device).to(torch.float32).eval()

    student_model = StudentImageModel(clip_model).to(device).to(torch.float32).eval()
    load_result = student_model.load_state_dict(checkpoint["student_state_dict"], strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        print(
            "Checkpoint compatibility: "
            f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
        )

    image_encoder = StudentImageEncoderWrapper(student_model).eval()

    dummy_image_input = torch.rand(
        args.batch_size,
        3,
        args.image_size,
        args.image_size,
        dtype=torch.float32,
        device=device,
    )
    image_onnx_path = output_dir / args.image_onnx_name
    print(f"\nExporting student image encoder to {image_onnx_path}...")
    export_onnx_model(
        model=image_encoder,
        dummy_input=dummy_image_input,
        output_path=image_onnx_path,
        input_names=["image"],
        output_names=["embedding"],
        opset_version=args.opset_version,
    )

    text_onnx_path = None
    if not args.skip_text_encoder:
        text_encoder = TextEncoderWrapper(clip_model).eval()
        dummy_text_input = torch.randint(
            low=0,
            high=args.vocab_size,
            size=(args.batch_size, args.context_length),
            dtype=torch.int32,
            device=device,
        )
        text_onnx_path = output_dir / args.text_onnx_name
        print(f"\nExporting text encoder to {text_onnx_path}...")
        export_onnx_model(
            model=text_encoder,
            dummy_input=dummy_text_input,
            output_path=text_onnx_path,
            input_names=["text"],
            output_names=["text_embedding"],
            opset_version=args.opset_version,
        )

    metadata = {
        "student_checkpoint": str(checkpoint_path),
        "student_model_id": student_model_id,
        "image_onnx": str(image_onnx_path),
        "text_onnx": str(text_onnx_path) if text_onnx_path is not None else None,
        "opset_version": args.opset_version,
        "image_size": args.image_size,
        "context_length": args.context_length,
        "batch_size": args.batch_size,
    }
    metadata_path = output_dir / "export_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nExport complete.")
    print(f"Metadata written to: {metadata_path}")


if __name__ == "__main__":
    main()
