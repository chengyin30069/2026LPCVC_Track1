from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import open_clip
import torch

from benchmark import StudentEvalWrapper, configure_cuda_runtime, evaluate_coco_recall_at10
from utils.student_model import StudentImageModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all student checkpoints with benchmark_legacy-compatible Recall@10 "
            "(via benchmark.py evaluator), then plot Recall@10 and training loss together."
        )
    )
    parser.add_argument("--checkpoint-dir", default="artifacts/student-bigG14-CLIPKD")
    parser.add_argument("--checkpoint-glob", default="student_checkpoint_epoch_*.pt")
    parser.add_argument("--history", default=None, help="Optional history.json path (defaults to <checkpoint-dir>/history.json)")
    parser.add_argument(
        "--history-csv",
        default=None,
        help="Optional live history CSV path (defaults to <checkpoint-dir>/history.csv).",
    )
    parser.add_argument("--coco-root", default="coco2017/val2017")
    parser.add_argument("--coco-ann", default="coco2017/annotations/captions_val2017.json")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--debug-similarity", action="store_true")
    parser.add_argument(
        "--legacy-full-batches-only",
        action="store_true",
        default=True,
        help="Keep benchmark_legacy.py behavior: skip the last incomplete eval batch.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path (defaults to <checkpoint-dir>/benchmark_legacy_sweep.csv)",
    )
    parser.add_argument(
        "--output-plot",
        default=None,
        help="Output plot path (defaults to <checkpoint-dir>/benchmark_legacy_sweep.png)",
    )
    return parser.parse_args()


def parse_epoch_from_name(path: Path) -> int:
    match = re.search(r"epoch_(\d+)\.pt$", path.name)
    if not match:
        raise ValueError(f"Unable to parse epoch from checkpoint name: {path.name}")
    return int(match.group(1))


def _maybe_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"none", "nan", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_history_map(history_path: Path, history_csv_path: Path) -> dict[int, dict]:
    rows: list[dict] = []
    if history_csv_path.exists():
        with history_csv_path.open("r", encoding="utf-8", newline="") as file:
            csv_rows = list(csv.DictReader(file))
        for row in csv_rows:
            parsed = {}
            for key, value in row.items():
                parsed[key] = _maybe_float(value)
            rows.append(parsed)
    elif history_path.exists():
        rows = json.loads(history_path.read_text(encoding="utf-8"))

    history_map: dict[int, dict] = {}
    for row in rows:
        if row.get("epoch") is None:
            continue
        history_map[int(float(row["epoch"]))] = row
    return history_map


def build_eval_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        device=args.device,
        no_amp=args.no_amp,
        transformers_amp=False,
        coco_root=args.coco_root,
        coco_ann=args.coco_ann,
        max_samples=args.max_samples,
        eval_batch_size=args.eval_batch_size,
        feature_batch_size=args.feature_batch_size,
        teacher_feature_batch_size=32,
        teacher_text_feature_batch_size=64,
        teacher_score_mode="cosine",
        num_workers=args.num_workers,
        channels_last=args.channels_last,
        debug_similarity=args.debug_similarity,
        legacy_full_batches_only=args.legacy_full_batches_only,
    )


def main() -> None:
    args = parse_args()
    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    if not Path(args.coco_root).exists():
        raise FileNotFoundError(f"COCO image root not found: {args.coco_root}")
    if not Path(args.coco_ann).exists():
        raise FileNotFoundError(f"COCO annotation file not found: {args.coco_ann}")

    checkpoints = sorted(ckpt_dir.glob(args.checkpoint_glob), key=parse_epoch_from_name)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matched {args.checkpoint_glob!r} under {ckpt_dir}")

    history_path = Path(args.history) if args.history else ckpt_dir / "history.json"
    history_csv_path = Path(args.history_csv) if args.history_csv else ckpt_dir / "history.csv"
    history_map = load_history_map(history_path, history_csv_path)

    output_csv = Path(args.output_csv) if args.output_csv else ckpt_dir / "benchmark_legacy_sweep.csv"
    output_plot = Path(args.output_plot) if args.output_plot else ckpt_dir / "benchmark_legacy_sweep.png"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_plot.parent.mkdir(parents=True, exist_ok=True)

    configure_cuda_runtime(args.device)
    eval_args = build_eval_args(args)

    first_checkpoint = torch.load(checkpoints[0], map_location="cpu")
    student_model_id = first_checkpoint.get("student_model_id", "hf-hub:laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K")

    clip_model, _, preprocess = open_clip.create_model_and_transforms(student_model_id)
    tokenizer = open_clip.get_tokenizer(student_model_id)
    clip_model = clip_model.to(args.device).eval()
    student_model = StudentImageModel(clip_model).to(args.device).eval()
    wrapped = StudentEvalWrapper(clip_model, student_model).to(args.device).eval()

    tracked_loss_keys = [
        "contrastive_loss",
        "baseline_anchor_loss",
        "icl_loss",
        "clipkd_ckd_loss",
        "projected_fd_loss",
    ]
    tracked_metric_keys = ["val_recall_at_10"]
    rows: list[dict[str, float | int | str | None]] = []

    for ckpt_path in checkpoints:
        epoch = parse_epoch_from_name(ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        load_result = student_model.load_state_dict(checkpoint["student_state_dict"], strict=False)
        unexpected_keys = [key for key in load_result.unexpected_keys if not key.startswith("teacher_projector.")]
        if load_result.missing_keys or unexpected_keys:
            print(
                f"[epoch {epoch}] compatibility warning: "
                f"missing={load_result.missing_keys}, unexpected={unexpected_keys}"
            )

        score = evaluate_coco_recall_at10(
            model=wrapped,
            tokenizer=tokenizer,
            preprocess=preprocess,
            args=eval_args,
            label=f"student-epoch-{epoch:02d}",
        )

        history_row = history_map.get(epoch, {})
        train_loss = history_row.get("total_loss")
        val_loss = history_row.get("val_total_loss")

        row = {
            "epoch": epoch,
            "checkpoint": str(ckpt_path),
            "recall_at_10": float(score),
            "train_total_loss": float(train_loss) if train_loss is not None else None,
            "val_total_loss": float(val_loss) if val_loss is not None else None,
        }
        for key in tracked_loss_keys + tracked_metric_keys:
            value = history_row.get(key)
            row[key] = float(value) if value is not None else None
        rows.append(row)
        print(
            f"[{epoch:02d}] Recall@10={row['recall_at_10']:.6f} "
            f"train_loss={row['train_total_loss']} val_loss={row['val_total_loss']}"
        )

        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    rows.sort(key=lambda item: int(item["epoch"]))
    best = max(rows, key=lambda item: float(item["recall_at_10"]))

    fieldnames = ["epoch", "checkpoint", "recall_at_10", "train_total_loss", "val_total_loss"] + tracked_loss_keys + tracked_metric_keys
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    epochs = [int(row["epoch"]) for row in rows]
    recalls = [float(row["recall_at_10"]) for row in rows]
    train_losses = [row["train_total_loss"] for row in rows]
    val_losses = [row["val_total_loss"] for row in rows]
    train_val_recalls = [row.get("val_recall_at_10") for row in rows]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_acc, ax_loss) = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True, sharex=True)
    ax_acc.plot(epochs, recalls, marker="o", linewidth=2.0, label="Checkpoint Recall@10")
    val_recall_x = [x for x, y in zip(epochs, train_val_recalls) if y is not None]
    val_recall_y = [float(y) for y in train_val_recalls if y is not None]
    if val_recall_x:
        ax_acc.plot(val_recall_x, val_recall_y, marker="d", linewidth=1.5, linestyle="--", label="Train Val Recall@10")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Checkpoint Recall and Active Distillation Losses")
    ax_acc.legend(loc="best")

    train_x = [x for x, y in zip(epochs, train_losses) if y is not None]
    train_y = [float(y) for y in train_losses if y is not None]
    if train_x:
        ax_loss.plot(train_x, train_y, marker="s", linewidth=1.5, linestyle="--", label="Train total_loss")

    val_x = [x for x, y in zip(epochs, val_losses) if y is not None]
    val_y = [float(y) for y in val_losses if y is not None]
    if val_x:
        ax_loss.plot(val_x, val_y, marker="^", linewidth=1.5, linestyle=":", label="Val total_loss")

    for key in tracked_loss_keys:
        loss_x = [x for x, row in zip(epochs, rows) if row.get(key) is not None]
        loss_y = [float(row[key]) for row in rows if row.get(key) is not None]
        if loss_x:
            ax_loss.plot(loss_x, loss_y, linewidth=1.2, label=key)

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(loc="best")

    fig.savefig(output_plot, dpi=180)

    print("\nSweep finished.")
    print(f"Best epoch: {int(best['epoch'])}, Recall@10: {float(best['recall_at_10']):.6f}")
    print(f"Saved CSV : {output_csv}")
    print(f"Saved plot: {output_plot}")


if __name__ == "__main__":
    main()
