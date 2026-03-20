from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run two-stage student distillation (image-only stage + text-alignment stage)."
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable used to launch stages.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")

    parser.add_argument("--skip-stage1", action="store_true", help="Skip stage 1 image-only distillation.")
    parser.add_argument("--skip-stage2", action="store_true", help="Skip stage 2 text-alignment training.")
    parser.add_argument("--skip-stage3", action="store_true", help="Skip stage 3 metric-push fine-tuning.")

    parser.add_argument("--stage1-img-list", default="dataset_image_only/img_list.csv")
    parser.add_argument("--stage1-image-folder", default="dataset_image_only/images")
    parser.add_argument("--stage1-teacher-embeddings", default="artifacts/teacher_image_embeddings_image_only.npz")
    parser.add_argument("--stage1-output-dir", default="artifacts/student_stage1_image_only_v2")
    parser.add_argument("--stage1-epochs", type=int, default=14)
    parser.add_argument("--stage1-batch-size", type=int, default=128)
    parser.add_argument("--stage1-grad-accumulation", type=int, default=2)

    parser.add_argument("--stage2-img-list", default="dataset/img_list.csv")
    parser.add_argument("--stage2-image-folder", default="dataset/images")
    parser.add_argument("--stage2-text-embeddings", default="artifacts/text_embeddings.npz")
    parser.add_argument("--stage2-hard-negatives", default="artifacts/hard_negatives.npz")
    parser.add_argument("--stage2-teacher-embeddings", default="artifacts/teacher_image_embeddings.npz")
    parser.add_argument("--stage2-output-dir", default="artifacts/student_stage2_text_align_v2")
    parser.add_argument("--stage2-epochs", type=int, default=16)
    parser.add_argument("--stage2-batch-size", type=int, default=96)
    parser.add_argument("--stage2-grad-accumulation", type=int, default=2)
    parser.add_argument(
        "--stage2-resume-checkpoint",
        help="Optional explicit resume checkpoint for stage 2. If omitted, uses stage1 best checkpoint when available.",
    )

    parser.add_argument("--stage3-img-list", default="dataset/img_list.csv")
    parser.add_argument("--stage3-image-folder", default="dataset/images")
    parser.add_argument("--stage3-text-embeddings", default="artifacts/text_embeddings.npz")
    parser.add_argument("--stage3-hard-negatives", default="artifacts/hard_negatives.npz")
    parser.add_argument("--stage3-teacher-embeddings", default="artifacts/teacher_image_embeddings.npz")
    parser.add_argument("--stage3-output-dir", default="artifacts/student_stage3_metric_push")
    parser.add_argument("--stage3-epochs", type=int, default=4)
    parser.add_argument("--stage3-batch-size", type=int, default=96)
    parser.add_argument("--stage3-grad-accumulation", type=int, default=2)
    parser.add_argument(
        "--stage3-resume-checkpoint",
        help="Optional explicit resume checkpoint for stage 3. If omitted, uses stage2 best checkpoint when available.",
    )
    parser.add_argument(
        "--stage3-target-recall",
        type=float,
        default=0.45,
        help="Target Validation Recall@10 for stage 3 summary output.",
    )
    parser.add_argument(
        "--stage3-enforce-target",
        action="store_true",
        help="Fail if stage 3 best Validation Recall@10 does not reach --stage3-target-recall.",
    )

    parser.add_argument("--device", default="cuda:1" if sys.platform != "darwin" else "cpu")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def ensure_exists(path: str, label: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def run_command(command: list[str], *, dry_run: bool, label: str) -> None:
    printable = " ".join(shlex.quote(part) for part in command)
    print(f"[{label}] {printable}")
    if dry_run:
        return
    subprocess.run(command, check=True)


def summarize_stage3_result(
    *,
    history_path: Path,
    target_recall: float,
    enforce_target: bool,
) -> None:
    if not history_path.exists():
        print(f"[stage3] history not found: {history_path}")
        if enforce_target:
            raise FileNotFoundError(f"stage3 history not found: {history_path}")
        return

    records = json.loads(history_path.read_text(encoding="utf-8"))
    recall_values = [
        float(record["val_recall_at_10"])
        for record in records
        if isinstance(record, dict) and "val_recall_at_10" in record
    ]
    if not recall_values:
        print("[stage3] val_recall_at_10 is missing in history; cannot evaluate target.")
        if enforce_target:
            raise RuntimeError("stage3 target enforcement requested but val_recall_at_10 is unavailable.")
        return

    best_recall = max(recall_values)
    print(
        "[stage3] Validation Recall@10 summary: "
        f"best={best_recall:.6f}, target={target_recall:.6f}, reached={best_recall >= target_recall}"
    )
    if enforce_target and best_recall < target_recall:
        raise RuntimeError(
            "stage3 target not reached: "
            f"best_val_recall_at_10={best_recall:.6f} < target={target_recall:.6f}"
        )


def build_stage1_command(args: argparse.Namespace) -> list[str]:
    output_dir = Path(args.stage1_output_dir)
    missing_log = output_dir / "missing_teacher_images.txt"
    command = [
        args.python,
        "train_student_distill.py",
        "--img-list", args.stage1_img_list,
        "--image-folder", args.stage1_image_folder,
        "--teacher-embeddings", args.stage1_teacher_embeddings,
        "--output-dir", args.stage1_output_dir,
        "--epochs", str(args.stage1_epochs),
        "--batch-size", str(args.stage1_batch_size),
        "--grad-accumulation", str(args.stage1_grad_accumulation),
        "--lr", "1e-5",
        "--weight-decay", "1e-4",
        "--contrastive-weight", "0.0",
        "--distill-weight", "0.7",
        "--distill-loss-type", "cosine",
        "--baseline-anchor-weight", "0.3",
        "--baseline-anchor-final-weight", "0.1",
        "--hard-negative-weight", "0.0",
        "--lr-scheduler", "cosine",
        "--warmup-steps", "500",
        "--min-lr", "1e-7",
        "--val-split", "0.05",
        "--best-checkpoint-metric", "val-loss",
        "--grad-clip-norm", "1.0",
        "--unfreeze-last-n-blocks", "0",
        "--missing-teacher-log", str(missing_log),
        "--device", args.device,
    ]
    if args.channels_last:
        command.append("--channels-last")
    if args.gradient_checkpointing:
        command.append("--gradient-checkpointing")
    return command


def build_stage2_command(args: argparse.Namespace, default_resume_checkpoint: str | None) -> list[str]:
    command = [
        args.python,
        "train_student_distill.py",
        "--img-list", args.stage2_img_list,
        "--image-folder", args.stage2_image_folder,
        "--text-embeddings", args.stage2_text_embeddings,
        "--hard-negatives", args.stage2_hard_negatives,
        "--teacher-embeddings", args.stage2_teacher_embeddings,
        "--output-dir", args.stage2_output_dir,
        "--epochs", str(args.stage2_epochs),
        "--batch-size", str(args.stage2_batch_size),
        "--grad-accumulation", str(args.stage2_grad_accumulation),
        "--lr", "5e-6",
        "--weight-decay", "1e-4",
        "--contrastive-weight", "1.0",
        "--distill-weight", "0.2",
        "--distill-loss-type", "cosine",
        "--baseline-anchor-weight", "0.05",
        "--baseline-anchor-final-weight", "0.0",
        "--hard-negative-weight", "0.03",
        "--num-hard-negatives", "4",
        "--lr-scheduler", "cosine",
        "--warmup-steps", "500",
        "--min-lr", "1e-7",
        "--val-split", "0.05",
        "--best-checkpoint-metric", "val-recall@10",
        "--val-recall-max-texts", "0",
        "--val-recall-image-chunk-size", "256",
        "--val-recall-text-chunk-size", "8192",
        "--grad-clip-norm", "1.0",
        "--unfreeze-last-n-blocks", "2",
        "--save-epoch-checkpoints",
        "--device", args.device,
    ]
    resume_checkpoint = args.stage2_resume_checkpoint or default_resume_checkpoint
    if resume_checkpoint:
        command.extend(["--resume-checkpoint", resume_checkpoint])

    if args.channels_last:
        command.append("--channels-last")
    if args.gradient_checkpointing:
        command.append("--gradient-checkpointing")
    return command


def build_stage3_command(args: argparse.Namespace, default_resume_checkpoint: str | None) -> list[str]:
    command = [
        args.python,
        "train_student_distill.py",
        "--img-list", args.stage3_img_list,
        "--image-folder", args.stage3_image_folder,
        "--text-embeddings", args.stage3_text_embeddings,
        "--hard-negatives", args.stage3_hard_negatives,
        "--teacher-embeddings", args.stage3_teacher_embeddings,
        "--output-dir", args.stage3_output_dir,
        "--epochs", str(args.stage3_epochs),
        "--batch-size", str(args.stage3_batch_size),
        "--grad-accumulation", str(args.stage3_grad_accumulation),
        "--lr", "2e-6",
        "--weight-decay", "1e-4",
        "--contrastive-weight", "1.0",
        "--distill-weight", "0.1",
        "--distill-loss-type", "cosine",
        "--baseline-anchor-weight", "0.02",
        "--baseline-anchor-final-weight", "0.0",
        "--hard-negative-weight", "0.06",
        "--num-hard-negatives", "8",
        "--lr-scheduler", "cosine",
        "--warmup-steps", "200",
        "--min-lr", "5e-8",
        "--val-split", "0.05",
        "--best-checkpoint-metric", "val-recall@10",
        "--val-recall-max-texts", "0",
        "--val-recall-image-chunk-size", "256",
        "--val-recall-text-chunk-size", "8192",
        "--grad-clip-norm", "1.0",
        "--unfreeze-last-n-blocks", "2",
        "--save-epoch-checkpoints",
        "--device", args.device,
    ]
    resume_checkpoint = args.stage3_resume_checkpoint or default_resume_checkpoint
    if resume_checkpoint:
        command.extend(["--resume-checkpoint", resume_checkpoint])

    if args.channels_last:
        command.append("--channels-last")
    if args.gradient_checkpointing:
        command.append("--gradient-checkpointing")
    return command


def main() -> None:
    args = parse_args()

    if args.skip_stage1 and args.skip_stage2 and args.skip_stage3:
        raise ValueError("Nothing to run: all stages are skipped.")

    default_stage2_resume_checkpoint = None
    default_stage3_resume_checkpoint = str(Path(args.stage2_output_dir) / "best_loss_checkpoint.pt")

    if not args.skip_stage1:
        ensure_exists(args.stage1_img_list, "Stage1 img-list")
        ensure_exists(args.stage1_image_folder, "Stage1 image folder")
        ensure_exists(args.stage1_teacher_embeddings, "Stage1 teacher embeddings")
        stage1_command = build_stage1_command(args)
        run_command(stage1_command, dry_run=args.dry_run, label="stage1")
        default_stage2_resume_checkpoint = str(Path(args.stage1_output_dir) / "best_loss_checkpoint.pt")

    if not args.skip_stage2:
        ensure_exists(args.stage2_img_list, "Stage2 img-list")
        ensure_exists(args.stage2_image_folder, "Stage2 image folder")
        ensure_exists(args.stage2_text_embeddings, "Stage2 text embeddings")
        ensure_exists(args.stage2_hard_negatives, "Stage2 hard negatives")
        ensure_exists(args.stage2_teacher_embeddings, "Stage2 teacher embeddings")

        if args.stage2_resume_checkpoint:
            ensure_exists(args.stage2_resume_checkpoint, "Stage2 resume checkpoint")
        elif default_stage2_resume_checkpoint and not args.dry_run:
            ensure_exists(default_stage2_resume_checkpoint, "Auto stage2 resume checkpoint")

        stage2_command = build_stage2_command(args, default_stage2_resume_checkpoint)
        run_command(stage2_command, dry_run=args.dry_run, label="stage2")
        default_stage3_resume_checkpoint = str(Path(args.stage2_output_dir) / "best_loss_checkpoint.pt")

    # if not args.skip_stage3:
    #     ensure_exists(args.stage3_img_list, "Stage3 img-list")
    #     ensure_exists(args.stage3_image_folder, "Stage3 image folder")
    #     ensure_exists(args.stage3_text_embeddings, "Stage3 text embeddings")
    #     ensure_exists(args.stage3_hard_negatives, "Stage3 hard negatives")
    #     ensure_exists(args.stage3_teacher_embeddings, "Stage3 teacher embeddings")

    #     stage3_resume_checkpoint = args.stage3_resume_checkpoint or default_stage3_resume_checkpoint
    #     if stage3_resume_checkpoint:
    #         if not args.dry_run:
    #             ensure_exists(stage3_resume_checkpoint, "Stage3 resume checkpoint")
    #     else:
    #         raise ValueError(
    #             "Stage3 requires a resume checkpoint. "
    #             "Provide --stage3-resume-checkpoint or run stage2 first to generate one."
    #         )

    #     stage3_command = build_stage3_command(args, stage3_resume_checkpoint)
    #     run_command(stage3_command, dry_run=args.dry_run, label="stage3")

    #     if not args.dry_run:
    #         summarize_stage3_result(
    #             history_path=Path(args.stage3_output_dir) / "history.json",
    #             target_recall=args.stage3_target_recall,
    #             enforce_target=args.stage3_enforce_target,
    #         )


if __name__ == "__main__":
    main()
