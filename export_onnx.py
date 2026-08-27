import argparse
import json
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F
from torch import nn

def get_clip_embedding_dim(clip_model: nn.Module) -> int:
    projection = getattr(clip_model, "text_projection", None)
    if projection is not None and hasattr(projection, "shape"):
        return int(projection.shape[-1])
    raise ValueError("Unable to infer CLIP embedding dimension from the model.")

class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.residual_scale = nn.Parameter(torch.zeros(1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.norm(self.linear(features))
        mixed = features + self.residual_scale * residual
        return F.normalize(mixed, dim=-1)

class TeacherProjector(nn.Module):
    """Projects student embeddings to teacher embedding space for cross-dimension FD loss.

    Used only during training when teacher embedding dim != student embedding dim.
    At inference time, the student uses its original embedding dim directly.
    """

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.linear = nn.Linear(student_dim, teacher_dim, bias=False)
        nn.init.kaiming_normal_(self.linear.weight, mode="fan_out", nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.linear(x), dim=-1)

class StudentImageModel(nn.Module):
    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.clip_model = clip_model
        self.embedding_dim = get_clip_embedding_dim(clip_model)
        self.projection_head = ProjectionHead(self.embedding_dim)
        self.teacher_projector: TeacherProjector | None = None

    def init_teacher_projector(self, teacher_dim: int) -> None:
        """Initialize a projector to map student embeddings to teacher embedding space."""
        if teacher_dim != self.embedding_dim:
            self.teacher_projector = TeacherProjector(self.embedding_dim, teacher_dim)

    def project_to_teacher(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project student embeddings to teacher space. Returns original if dims match."""
        if self.teacher_projector is not None:
            return self.teacher_projector(embeddings)
        return F.normalize(embeddings, dim=-1)

    def encode_backbone(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.clip_model.encode_image(pixel_values)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.projection_head(self.encode_backbone(pixel_values))


class StudentImageEncoderWrapper(nn.Module):
    def __init__(self, student_model: StudentImageModel):
        super().__init__()
        self.student_model = student_model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.student_model(images)


class ClipImageEncoderWrapper(nn.Module):
    def __init__(self, clip_model: nn.Module, *, normalize: bool = True):
        super().__init__()
        self.clip_model = clip_model
        self.normalize = normalize

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            try:
                return self.clip_model.encode_image(images, normalize=True)
            except TypeError:
                return F.normalize(self.clip_model.encode_image(images), dim=-1)
        return self.clip_model.encode_image(images)


class TextEncoderWrapper(nn.Module):
    def __init__(self, clip_model: nn.Module, *, cast_input_to_int64: bool):
        super().__init__()
        self.clip_model = clip_model
        self.cast_input_to_int64 = cast_input_to_int64

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.cast_input_to_int64:
            token_ids = token_ids.to(torch.int64)
        return F.normalize(self.clip_model.encode_text(token_ids), dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export student checkpoint to ONNX.")
    parser.add_argument(
        "--student-checkpoint",
        default="artifacts/student_checkpoint_deploy.pt",
        help="Path to trained student checkpoint.",
    )
    parser.add_argument("--output-dir", default="exported_onnx")
    parser.add_argument("--image-onnx-name", default="image_encoder.onnx")
    parser.add_argument("--text-onnx-name", default="text_encoder.onnx")
    parser.add_argument("--opset-version", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--context-length", type=int, default=77)
    parser.add_argument("--vocab-size", type=int, default=49408)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--export-original-clip-arch",
        action="store_true",
        default=True,
        help=(
            "Export image encoder with original OpenCLIP architecture (clip_model.encode_image) "
            "instead of StudentImageModel projection_head path."
        ),
    )
    parser.add_argument(
        "--no-image-normalize",
        action="store_true",
        help="Do not apply final embedding normalization in image ONNX output.",
    )
    parser.add_argument(
        "--external-data",
        action="store_true",
        help="Export ONNX weights as sidecar .onnx.data files (disabled by default).",
    )
    parser.add_argument(
        "--no-dynamo-export",
        action="store_true",
        default=True,
        help="Use legacy Torch ONNX exporter (dynamo=False) for maximum converter compatibility.",
    )
    parser.add_argument(
        "--text-input-dtype",
        choices=["int64", "int32"],
        default="int32",
        help="Exported text encoder input dtype. Legacy QAI flow expects int64.",
    )
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
    *,
    external_data: bool,
    dynamo_export: bool,
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
        dynamo=dynamo_export,
        external_data=external_data,
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

    if args.export_original_clip_arch:
        image_encoder = ClipImageEncoderWrapper(clip_model, normalize=not args.no_image_normalize).eval()
        print("Image export mode: original OpenCLIP architecture (no student projection_head).")
    else:
        image_encoder = StudentImageEncoderWrapper(student_model).eval()
        print("Image export mode: distilled StudentImageModel architecture (includes projection_head).")

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
        external_data=args.external_data,
        dynamo_export=not args.no_dynamo_export,
    )

    text_onnx_path = None
    if not args.skip_text_encoder:
        text_encoder = TextEncoderWrapper(clip_model, cast_input_to_int64=(args.text_input_dtype == "int32")).eval()
        text_dummy_dtype = torch.int64 if args.text_input_dtype == "int64" else torch.int32
        dummy_text_input = torch.randint(
            low=0,
            high=args.vocab_size,
            size=(args.batch_size, args.context_length),
            dtype=text_dummy_dtype,
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
            external_data=args.external_data,
            dynamo_export=not args.no_dynamo_export,
        )

    metadata = {
        "student_checkpoint": str(checkpoint_path),
        "student_model_id": student_model_id,
        "image_onnx": str(image_onnx_path),
        "text_onnx": str(text_onnx_path) if text_onnx_path is not None else None,
        "image_export_architecture": "openclip" if args.export_original_clip_arch else "student_with_projection_head",
        "image_normalized_output": not args.no_image_normalize,
        "external_data": args.external_data,
        "text_input_dtype": args.text_input_dtype,
        "dynamo_export": not args.no_dynamo_export,
        "opset_version": args.opset_version,
        "image_size": args.image_size,
        "context_length": args.context_length,
        "batch_size": args.batch_size,
    }
    metadata_path = output_dir / "export_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if not args.external_data:
        image_sidecar = image_onnx_path.with_suffix(image_onnx_path.suffix + ".data")
        if image_sidecar.exists():
            print(f"Warning: unexpected sidecar file generated: {image_sidecar}")
        if text_onnx_path is not None:
            text_sidecar = text_onnx_path.with_suffix(text_onnx_path.suffix + ".data")
            if text_sidecar.exists():
                print(f"Warning: unexpected sidecar file generated: {text_sidecar}")

    print("\nExport complete.")
    print(f"Metadata written to: {metadata_path}")


if __name__ == "__main__":
    main()