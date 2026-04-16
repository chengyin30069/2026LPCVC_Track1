from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import open_clip
import torch
from torch import nn

from benchmark import (
    configure_cuda_runtime,
    evaluate_coco_recall_at10,
    load_teacher_model,
)
from utils.student_model import StudentImageModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark hybrid retrieval model: student image encoder + teacher text encoder."
    )
    parser.add_argument("--coco-root", default="coco2017/val2017")
    parser.add_argument("--coco-ann", default="coco2017/annotations/captions_val2017.json")
    parser.add_argument(
        "--student-checkpoint",
        default="artifacts/student_stage2_text_align_v2/best_loss_checkpoint.pt",
    )
    parser.add_argument(
        "--teacher-model-id",
        default="hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        help="Teacher model used for text encoding.",
    )
    parser.add_argument(
        "--teacher-backend",
        choices=["auto", "open_clip", "transformers"],
        default="auto",
        help="Teacher text encoder backend.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--transformers-amp",
        action="store_true",
        help="Kept for compatibility with benchmark.py args shape.",
    )
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--debug-similarity", action="store_true")
    parser.add_argument(
        "--legacy-full-batches-only",
        action="store_true",
        help="Keep benchmark_legacy behavior by dropping incomplete final batch.",
    )
    parser.add_argument(
        "--teacher-feature-batch-size",
        type=int,
        default=32,
        help="Compatibility arg; not used by this hybrid wrapper.",
    )
    parser.add_argument(
        "--teacher-text-feature-batch-size",
        type=int,
        default=64,
        help="Compatibility arg; not used by this hybrid wrapper.",
    )
    parser.add_argument(
        "--teacher-score-mode",
        choices=["cosine", "logit"],
        default="cosine",
        help="Compatibility arg; not used by this hybrid wrapper.",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _move_to_device(token_batch: Any, device: str) -> Any:
    if isinstance(token_batch, dict):
        return {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in token_batch.items()
        }
    if hasattr(token_batch, "to"):
        return token_batch.to(device)
    return token_batch


def _coerce_feature_tensor(output_obj: Any) -> torch.Tensor:
    if isinstance(output_obj, torch.Tensor):
        if output_obj.ndim == 3:
            return output_obj[:, 0, :]
        return output_obj
    for candidate_name in ("text_embeds", "image_embeds", "pooler_output", "last_hidden_state"):
        if hasattr(output_obj, candidate_name):
            candidate = getattr(output_obj, candidate_name)
            if isinstance(candidate, torch.Tensor):
                if candidate.ndim == 3:
                    return candidate[:, 0, :]
                return candidate
    raise RuntimeError("Unable to coerce text features into a tensor for dimension probing.")


def _maybe_init_teacher_projector(student_model: StudentImageModel, state_dict: dict[str, torch.Tensor]) -> None:
    projector_key = "teacher_projector.linear.weight"
    if projector_key not in state_dict:
        return
    weight = state_dict[projector_key]
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError(f"Invalid projector weight in checkpoint: {projector_key}")
    teacher_dim, student_dim = int(weight.shape[0]), int(weight.shape[1])
    if student_dim != student_model.embedding_dim:
        raise ValueError(
            "Checkpoint projector/student embedding mismatch: "
            f"projector expects {student_dim}, student embedding_dim is {student_model.embedding_dim}."
        )
    student_model.init_teacher_projector(teacher_dim)


def load_student_image_model(checkpoint_path: Path, device: str) -> tuple[StudentImageModel, Any, str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    student_model_id = checkpoint.get("student_model_id", "hf-hub:laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K")

    clip_model, _, preprocess = open_clip.create_model_and_transforms(student_model_id)
    clip_model = clip_model.to(device).eval()

    student_model = StudentImageModel(clip_model).to(device).eval()
    state_dict = checkpoint["student_state_dict"]
    _maybe_init_teacher_projector(student_model, state_dict)
    load_result = student_model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        print(
            "Student checkpoint compatibility: "
            f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
        )
    return student_model, preprocess, student_model_id


def probe_teacher_text_dim(teacher_model: nn.Module, teacher_tokenizer, device: str) -> int:
    with torch.inference_mode():
        token_batch = teacher_tokenizer(["a photo of a dog"])
        token_batch = _move_to_device(token_batch, device)
        features = teacher_model.encode_text(token_batch)
        features = _coerce_feature_tensor(features)
    return int(features.shape[-1])


class HybridStudentImageTeacherTextModel(nn.Module):
    def __init__(
        self,
        *,
        student_model: StudentImageModel,
        teacher_text_model: nn.Module,
        project_image_to_teacher_dim: bool,
    ):
        super().__init__()
        self.student_model = student_model
        self.teacher_text_model = teacher_text_model
        self.project_image_to_teacher_dim = project_image_to_teacher_dim

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        image_features = self.student_model(pixel_values)
        if self.project_image_to_teacher_dim:
            image_features = self.student_model.project_to_teacher(image_features)
        return image_features

    def encode_text(self, token_batch) -> torch.Tensor:
        return self.teacher_text_model.encode_text(token_batch)


def main() -> None:
    args = parse_args()
    configure_cuda_runtime(args.device)

    if not Path(args.coco_root).exists():
        raise FileNotFoundError(f"COCO image root not found: {args.coco_root}")
    if not Path(args.coco_ann).exists():
        raise FileNotFoundError(f"COCO annotation file not found: {args.coco_ann}")

    checkpoint_path = Path(args.student_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Student checkpoint not found: {checkpoint_path}")

    student_model, student_preprocess, student_model_id = load_student_image_model(checkpoint_path, args.device)
    teacher_text_model, teacher_tokenizer, _ = load_teacher_model(
        args.teacher_model_id,
        args.device,
        args.teacher_backend,
    )

    student_dim = int(student_model.embedding_dim)
    teacher_text_dim = probe_teacher_text_dim(teacher_text_model, teacher_tokenizer, args.device)

    use_projection = False
    if student_dim != teacher_text_dim:
        projector = student_model.teacher_projector
        projector_out_dim = (
            int(projector.linear.weight.shape[0])
            if projector is not None and hasattr(projector, "linear")
            else None
        )
        if projector is None or projector_out_dim != teacher_text_dim:
            raise ValueError(
                "Hybrid encoder dimension mismatch: "
                f"student_image_dim={student_dim}, teacher_text_dim={teacher_text_dim}. "
                "No compatible teacher_projector was found in the student checkpoint."
            )
        use_projection = True
        print(
            "Using checkpoint teacher_projector for hybrid eval: "
            f"{student_dim} -> {teacher_text_dim}"
        )

    hybrid_model = HybridStudentImageTeacherTextModel(
        student_model=student_model,
        teacher_text_model=teacher_text_model,
        project_image_to_teacher_dim=use_projection,
    ).to(args.device).eval()

    score = evaluate_coco_recall_at10(
        model=hybrid_model,
        tokenizer=teacher_tokenizer,
        preprocess=student_preprocess,
        args=args,
        label="student-image+teacher-text",
    )

    result = {
        "student_checkpoint": str(checkpoint_path),
        "student_model_id": student_model_id,
        "teacher_model_id": args.teacher_model_id,
        "teacher_backend": args.teacher_backend,
        "student_image_dim": student_dim,
        "teacher_text_dim": teacher_text_dim,
        "used_teacher_projector": use_projection,
        "hybrid_recall_at_10": float(score),
    }
    print("\nHybrid Validation Recall@10")
    print(f"student-image + teacher-text : {score:.6f}")
    print("\nJSON summary")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
