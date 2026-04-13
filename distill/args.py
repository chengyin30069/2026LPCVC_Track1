from __future__ import annotations

import argparse

import torch


def apply_clipkd_preset(args: argparse.Namespace) -> None:
    preset = getattr(args, "clipkd_preset", "none")
    if preset == "none":
        return

    # CLIP-KD paper-aligned defaults for this codebase:
    # prefer FD + ICL (+ optional CRD), and disable weaker auxiliary KD terms.
    args.learnable_temperature = True
    args.weight_decay = 0.1

    args.distill_weight = 0.0
    args.distill_final_weight = 0.0
    args.teacher_cosine_weight = 0.0
    args.teacher_cosine_final_weight = 0.0

    args.hard_negative_weight = 0.0
    args.hard_negative_final_weight = 0.0
    args.memory_bank_distill_weight = 0.0
    args.memory_bank_distill_final_weight = 0.0
    args.backbone_feature_distill_weight = 0.0
    args.backbone_feature_distill_final_weight = 0.0
    args.masked_feature_distill_weight = 0.0
    args.masked_feature_distill_final_weight = 0.0
    args.gradient_distill_weight = 0.0
    args.gradient_distill_final_weight = 0.0
    args.augmented_feature_distill_weight = 0.0
    args.augmented_feature_distill_final_weight = 0.0
    args.intermediate_distill_weight = 0.0
    args.intermediate_distill_final_weight = 0.0

    # Keep CLIP task objective and make FD/ICL contributions non-trivial.
    args.contrastive_weight = 0.08
    args.contrastive_final_weight = 0.08
    args.feature_distill_weight = 2.5
    args.feature_distill_final_weight = 2.5
    args.icl_weight = 0.08
    args.icl_final_weight = 0.04

    # Keep a small non-zero anchor floor to reduce late embedding drift.
    args.baseline_anchor_weight = 0.01
    args.baseline_anchor_final_weight = 0.003

    # CRD is optional in paper ablations; keep it only in fd-icl-crd preset.
    if preset == "fd-icl-crd":
        args.relation_distill_weight = 0.0
        args.relation_distill_final_weight = 0.0
        args.crd_weight = 2.0
        args.crd_final_weight = 2.0
    else:
        args.relation_distill_weight = 0.0
        args.relation_distill_final_weight = 0.0
        args.crd_weight = 0.0
        args.crd_final_weight = 0.0

    # Paper runs are longer (32 epochs); upgrade from default short runs.
    if args.epochs == 16:
        args.epochs = 32
    if args.warmup_steps == 600:
        args.warmup_steps = 1000


def apply_bigG14_fd_icl_preset(args: argparse.Namespace) -> None:
    """CLIP-KD recipe optimized for bigG-14 teacher with cross-dimension projected FD."""
    args.learnable_temperature = True
    args.weight_decay = 0.05

    # Disable all legacy losses
    args.distill_weight = 0.0
    args.distill_final_weight = 0.0
    args.teacher_cosine_weight = 0.0
    args.teacher_cosine_final_weight = 0.0
    args.hard_negative_weight = 0.0
    args.hard_negative_final_weight = 0.0
    args.memory_bank_distill_weight = 0.0
    args.memory_bank_distill_final_weight = 0.0
    # Disable memory-bank buffer updates when memory-bank distillation is off.
    # This removes unnecessary per-step tensor concat/copy overhead.
    args.memory_bank_size = 0
    args.backbone_feature_distill_weight = 0.0
    args.backbone_feature_distill_final_weight = 0.0
    args.masked_feature_distill_weight = 0.0
    args.masked_feature_distill_final_weight = 0.0
    args.gradient_distill_weight = 0.0
    args.gradient_distill_final_weight = 0.0
    args.augmented_feature_distill_weight = 0.0
    args.augmented_feature_distill_final_weight = 0.0
    args.intermediate_distill_weight = 0.0
    args.intermediate_distill_final_weight = 0.0
    args.feature_distill_weight = 0.0
    args.feature_distill_final_weight = 0.0
    args.relation_distill_weight = 0.0
    args.relation_distill_final_weight = 0.0
    args.crd_weight = 0.0
    args.crd_final_weight = 0.0

    # Core CLIP-KD losses for bigG-14
    args.projected_fd_weight = 3.0
    args.projected_fd_final_weight = 3.0
    args.icl_weight = 0.15
    args.icl_final_weight = 0.08
    args.contrastive_weight = 0.1
    args.contrastive_final_weight = 0.1

    # Small anchor floor to prevent drift
    args.baseline_anchor_weight = 0.005
    args.baseline_anchor_final_weight = 0.001

    # Training scaling for 1.35M dataset (1,345,289 pairs)
    args.epochs = 40
    args.batch_size = 128
    args.grad_accumulation = 8
    args.lr = 5e-6
    args.warmup_steps = 2000
    args.unfreeze_last_n_blocks = 2
    args.early_stop_patience = 6
    args.save_epoch_checkpoints = True


def apply_clipkd_upstream_bigg14_preset(args: argparse.Namespace) -> None:
    """More faithful CLIP-KD setup adapted for bigG-14 teacher distillation."""
    args.learnable_temperature = True
    args.weight_decay = 0.1

    # Disable unrelated auxiliary losses.
    args.distill_weight = 0.0
    args.distill_final_weight = 0.0
    args.teacher_cosine_weight = 0.0
    args.teacher_cosine_final_weight = 0.0
    args.hard_negative_weight = 0.0
    args.hard_negative_final_weight = 0.0
    args.memory_bank_distill_weight = 0.0
    args.memory_bank_distill_final_weight = 0.0
    args.memory_bank_size = 0
    args.backbone_feature_distill_weight = 0.0
    args.backbone_feature_distill_final_weight = 0.0
    args.masked_feature_distill_weight = 0.0
    args.masked_feature_distill_final_weight = 0.0
    args.gradient_distill_weight = 0.0
    args.gradient_distill_final_weight = 0.0
    args.augmented_feature_distill_weight = 0.0
    args.augmented_feature_distill_final_weight = 0.0
    args.intermediate_distill_weight = 0.0
    args.intermediate_distill_final_weight = 0.0
    args.feature_distill_weight = 0.0
    args.feature_distill_final_weight = 0.0
    args.relation_distill_weight = 0.0
    args.relation_distill_final_weight = 0.0
    args.crd_weight = 0.0
    args.crd_final_weight = 0.0

    # CLIP-KD-style core objectives.
    args.contrastive_weight = 1.0
    args.contrastive_final_weight = 1.0
    args.projected_fd_weight = 2000.0
    args.projected_fd_final_weight = 2000.0
    args.icl_weight = 1.0
    args.icl_final_weight = 1.0
    args.icl_loss_type = "ce"
    args.clipkd_ckd_weight = 1.0
    args.clipkd_ckd_final_weight = 1.0
    args.clipkd_cross_kd_weight = 0.0
    args.clipkd_cross_kd_final_weight = 0.0

    # Keep tiny anchor to limit catastrophic drift.
    args.baseline_anchor_weight = 0.001
    args.baseline_anchor_final_weight = 0.0

    # Training setup tuned for single 3090Ti.
    args.epochs = 24
    args.batch_size = 96
    args.grad_accumulation = 8
    args.lr = 3e-6
    args.warmup_steps = 1500
    args.unfreeze_last_n_blocks = 2
    args.unfreeze_text_last_n_blocks = 2
    args.online_student_text = True
    args.train_logit_scale = True
    args.val_split = 0.02
    args.val_recall_source = "all"
    args.val_recall_max_texts = 120000
    args.early_stop_patience = 4
    args.save_epoch_checkpoints = True


def apply_clipkd_upstream_h14_preset(args: argparse.Namespace) -> None:
    """CLIP-KD-style setup for ViT-H-14 teacher on CC3M-scale data."""
    args.learnable_temperature = True
    args.weight_decay = 0.1
    args.adam_beta1 = 0.9
    args.adam_beta2 = 0.98
    args.adam_eps = 1e-6

    # Disable unrelated auxiliary losses.
    args.distill_weight = 0.0
    args.distill_final_weight = 0.0
    args.teacher_cosine_weight = 0.0
    args.teacher_cosine_final_weight = 0.0
    args.hard_negative_weight = 0.0
    args.hard_negative_final_weight = 0.0
    args.memory_bank_distill_weight = 0.0
    args.memory_bank_distill_final_weight = 0.0
    args.memory_bank_size = 0
    args.backbone_feature_distill_weight = 0.0
    args.backbone_feature_distill_final_weight = 0.0
    args.masked_feature_distill_weight = 0.0
    args.masked_feature_distill_final_weight = 0.0
    args.gradient_distill_weight = 0.0
    args.gradient_distill_final_weight = 0.0
    args.augmented_feature_distill_weight = 0.0
    args.augmented_feature_distill_final_weight = 0.0
    args.intermediate_distill_weight = 0.0
    args.intermediate_distill_final_weight = 0.0
    args.feature_distill_weight = 0.0
    args.feature_distill_final_weight = 0.0
    args.relation_distill_weight = 0.0
    args.relation_distill_final_weight = 0.0
    args.crd_weight = 0.0
    args.crd_final_weight = 0.0

    # CLIP-KD-style core objectives with late-epoch KD decay for better baseline retention.
    args.contrastive_weight = 1.2
    args.contrastive_final_weight = 1.4
    args.projected_fd_weight = 800.0
    args.projected_fd_final_weight = 300.0
    args.icl_weight = 0.8
    args.icl_final_weight = 0.4
    args.icl_loss_type = "ce"
    args.clipkd_ckd_weight = 0.8
    args.clipkd_ckd_final_weight = 0.4
    args.clipkd_cross_kd_weight = 0.0
    args.clipkd_cross_kd_final_weight = 0.0

    # Stronger anchor helps avoid drifting below the initial student baseline.
    args.baseline_anchor_weight = 0.5
    args.baseline_anchor_final_weight = 0.2

    # Training setup tuned for single 3090Ti and no forced early-stop.
    args.epochs = 24
    args.batch_size = 1024
    args.grad_accumulation = 8
    args.lr = 2e-5
    args.warmup_steps = 9000
    args.unfreeze_last_n_blocks = 2
    args.unfreeze_text_last_n_blocks = 0
    args.online_student_text = False
    args.train_logit_scale = False
    args.val_split = 0.02
    args.val_recall_source = "all"
    args.val_recall_max_texts = 120000
    args.early_stop_patience = 0
    args.save_epoch_checkpoints = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a distilled student image model for Track1.")
    parser.add_argument("--img-list", default="dataset/img_list.csv")
    parser.add_argument("--image-folder", default="dataset/images")
    parser.add_argument(
        "--train-source",
        choices=["all", "coco", "coco2014", "coco2017", "vg"],
        default="all",
        help="Restrict training samples to specific image sources.",
    )
    parser.add_argument(
        "--text-embeddings",
        help="Optional text embedding cache. Required only when enabling text-based losses.",
    )
    parser.add_argument(
        "--teacher-text-embeddings",
        help="Optional teacher text embedding cache for interactive contrastive distillation.",
    )
    parser.add_argument("--teacher-embeddings", default="artifacts/teacher_image_embeddings.npz")
    parser.add_argument("--hard-negatives", help="Optional hard-negative NPZ file.")
    parser.add_argument(
        "--student-model-id",
        default="hf-hub:laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K",
    )
    parser.add_argument("--output-dir", default="artifacts/student_distill_v6")
    parser.add_argument(
        "--clipkd-preset",
        choices=[
            "none",
            "fd-icl",
            "fd-icl-crd",
            "bigG14-fd-icl",
            "clipkd-upstream-bigg14",
            "clipkd-upstream-h14",
        ],
        default="none",
        help=(
            "Apply CLIP-KD paper-aligned loss presets. "
            "fd-icl uses Feature Distillation + ICL; fd-icl-crd additionally enables CRD; "
            "bigG14-fd-icl uses projected FD for bigG-14 cross-dim distillation; "
            "clipkd-upstream-bigg14 approximates upstream CLIP-KD objective weights; "
            "clipkd-upstream-h14 is tuned for ViT-H-14 teacher on CC3M."
        ),
    )
    parser.add_argument("--resume-checkpoint", help="Optional checkpoint path to continue training from.")
    parser.add_argument(
        "--resume-optimizer-state",
        action="store_true",
        help="Restore optimizer/scaler states when available in the resume checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--grad-accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="AdamW beta1.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="AdamW beta2.")
    parser.add_argument("--adam-eps", type=float, default=1e-8, help="AdamW epsilon.")
    parser.add_argument("--contrastive-weight", type=float, default=0.06)
    parser.add_argument(
        "--contrastive-final-weight",
        type=float,
        help="Final contrastive weight at last epoch; defaults to --contrastive-weight.",
    )
    parser.add_argument(
        "--contrastive-loss-type",
        choices=["infonce", "sigmoid"],
        default="infonce",
        help="Contrastive objective between image and text embeddings.",
    )
    parser.add_argument("--distill-weight", type=float, default=0.86)
    parser.add_argument(
        "--distill-final-weight",
        type=float,
        help="Final distillation weight at last epoch; defaults to --distill-weight.",
    )
    parser.add_argument(
        "--distill-loss-type",
        choices=["mse", "cosine", "kl", "kl_sym", "smooth_l1"],
        default="mse",
        help="Distillation objective between student and teacher embeddings.",
    )
    parser.add_argument(
        "--distill-temperature",
        type=float,
        default=0.08,
        help="Legacy temperature (kept for backward compatibility).",
    )
    parser.add_argument(
        "--teacher-cosine-weight",
        type=float,
        default=0.06,
        help="Direct cosine loss weight between student and teacher embeddings.",
    )
    parser.add_argument(
        "--teacher-cosine-final-weight",
        type=float,
        help="Final teacher-cosine weight at last epoch; defaults to --teacher-cosine-weight.",
    )
    parser.add_argument("--baseline-anchor-weight", type=float, default=0.02)
    parser.add_argument(
        "--baseline-anchor-final-weight",
        type=float,
        default=0.0,
        help="Anchor weight at final epoch; weight is linearly decayed from baseline-anchor-weight.",
    )
    parser.add_argument("--hard-negative-weight", type=float, default=0.035)
    parser.add_argument(
        "--hard-negative-final-weight",
        type=float,
        help="Final hard-negative weight at last epoch; defaults to --hard-negative-weight.",
    )
    parser.add_argument("--hard-negative-margin", type=float, default=0.05)
    parser.add_argument(
        "--hard-negative-weighting",
        choices=["uniform", "softmax"],
        default="softmax",
        help="How to weight hard negatives inside the margin objective.",
    )
    parser.add_argument(
        "--hard-negative-softmax-temperature",
        type=float,
        default=0.04,
        help="Temperature for softmax hard-negative weighting.",
    )
    parser.add_argument("--num-hard-negatives", type=int, default=10)
    parser.add_argument(
        "--positive-pooling",
        choices=["random", "mean"],
        default="mean",
        help="How to build the positive text target when multiple positives exist for an image.",
    )
    parser.add_argument(
        "--relation-distill-weight",
        type=float,
        default=0.05,
        help="Extra relation-level distillation weight over intra-batch similarities.",
    )
    parser.add_argument(
        "--relation-distill-final-weight",
        type=float,
        help="Final relation distillation weight; defaults to --relation-distill-weight.",
    )
    parser.add_argument(
        "--relation-distill-temperature",
        type=float,
        default=0.07,
        help="Temperature for relation distillation KL objective.",
    )
    parser.add_argument(
        "--crd-weight",
        type=float,
        default=0.01,
        help="Contrastive relational distillation (CRD) weight.",
    )
    parser.add_argument(
        "--crd-final-weight",
        type=float,
        help="Final CRD weight; defaults to --crd-weight.",
    )
    parser.add_argument(
        "--icl-weight",
        type=float,
        default=0.0,
        help="Interactive contrastive distillation weight (student anchor vs teacher distributions).",
    )
    parser.add_argument(
        "--icl-final-weight",
        type=float,
        help="Final interactive contrastive weight; defaults to --icl-weight.",
    )
    parser.add_argument(
        "--icl-teacher-temperature",
        type=float,
        default=0.07,
        help="Teacher temperature used by interactive contrastive distillation.",
    )
    parser.add_argument(
        "--icl-loss-type",
        choices=["kl", "ce"],
        default="kl",
        help="ICL objective: KL-to-teacher logits (kl) or upstream CE-style cross interaction (ce).",
    )
    parser.add_argument(
        "--clipkd-ckd-weight",
        type=float,
        default=0.0,
        help="Upstream CLIP-KD CKD weight (KL between student and teacher logits).",
    )
    parser.add_argument(
        "--clipkd-ckd-final-weight",
        type=float,
        help="Final CKD weight at last epoch; defaults to --clipkd-ckd-weight.",
    )
    parser.add_argument(
        "--clipkd-cross-kd-weight",
        type=float,
        default=0.0,
        help="Upstream CLIP-KD cross-KD weight (student-to-teacher cross logits KL).",
    )
    parser.add_argument(
        "--clipkd-cross-kd-final-weight",
        type=float,
        help="Final cross-KD weight at last epoch; defaults to --clipkd-cross-kd-weight.",
    )
    parser.add_argument(
        "--memory-bank-size",
        type=int,
        default=12288,
        help="Memory bank size for cross-batch relational distillation (0 disables).",
    )
    parser.add_argument(
        "--memory-bank-distill-weight",
        type=float,
        default=0.03,
        help="Cross-batch memory-bank distillation weight.",
    )
    parser.add_argument(
        "--memory-bank-distill-final-weight",
        type=float,
        help="Final memory-bank distillation weight; defaults to --memory-bank-distill-weight.",
    )
    parser.add_argument(
        "--memory-bank-min-samples",
        type=int,
        default=512,
        help="Minimum memory-bank samples before enabling memory-bank distillation.",
    )
    parser.add_argument(
        "--memory-bank-distill-temperature",
        type=float,
        default=0.07,
        help="Temperature for cross-batch memory-bank relational distillation.",
    )
    parser.add_argument(
        "--backbone-feature-distill-weight",
        type=float,
        default=0.0,
        help="Weight for matching student and reference image backbone features.",
    )
    parser.add_argument(
        "--backbone-feature-distill-final-weight",
        type=float,
        help="Final backbone feature distillation weight; defaults to --backbone-feature-distill-weight.",
    )
    parser.add_argument(
        "--feature-distill-weight",
        type=float,
        default=0.01,
        help="Feature distillation weight between student and teacher embeddings.",
    )
    parser.add_argument(
        "--feature-distill-final-weight",
        type=float,
        help="Final feature distillation weight; defaults to --feature-distill-weight.",
    )
    parser.add_argument(
        "--masked-feature-distill-weight",
        type=float,
        default=0.01,
        help="Masked feature distillation weight.",
    )
    parser.add_argument(
        "--masked-feature-distill-final-weight",
        type=float,
        help="Final masked feature distillation weight; defaults to --masked-feature-distill-weight.",
    )
    parser.add_argument(
        "--masked-feature-keep-ratio",
        type=float,
        default=0.7,
        help="Keep ratio for masked feature distillation.",
    )
    parser.add_argument(
        "--gradient-distill-weight",
        type=float,
        default=0.005,
        help="Gradient distillation weight.",
    )
    parser.add_argument(
        "--gradient-distill-final-weight",
        type=float,
        help="Final gradient distillation weight; defaults to --gradient-distill-weight.",
    )
    parser.add_argument(
        "--augmented-feature-distill-weight",
        type=float,
        default=0.01,
        help="Augmented feature distillation weight.",
    )
    parser.add_argument(
        "--augmented-feature-distill-final-weight",
        type=float,
        help="Final augmented feature distillation weight; defaults to --augmented-feature-distill-weight.",
    )
    parser.add_argument(
        "--augmented-feature-noise-std",
        type=float,
        default=0.02,
        help="Gaussian noise std used by augmented feature distillation.",
    )
    parser.add_argument(
        "--projected-fd-weight",
        type=float,
        default=0.0,
        help="Projected feature distillation weight (student projected to teacher dim via linear layer).",
    )
    parser.add_argument(
        "--projected-fd-final-weight",
        type=float,
        help="Final projected FD weight at last epoch; defaults to --projected-fd-weight.",
    )
    parser.add_argument(
        "--intermediate-distill-weight",
        type=float,
        default=0.0,
        help="Weight for teacher-student intermediate vision block distillation.",
    )
    parser.add_argument(
        "--intermediate-distill-final-weight",
        type=float,
        help="Final intermediate distillation weight; defaults to --intermediate-distill-weight.",
    )
    parser.add_argument(
        "--intermediate-distill-num-blocks",
        type=int,
        default=2,
        help="Number of last vision blocks used for intermediate distillation.",
    )
    parser.add_argument(
        "--intermediate-teacher-model-id",
        default="google/siglip2-so400m-patch16-512",
        help="Teacher model used for intermediate block distillation.",
    )
    parser.add_argument(
        "--intermediate-distill-frequency",
        type=int,
        default=2,
        help="Compute intermediate distillation every N training steps.",
    )
    parser.add_argument(
        "--intermediate-distill-stop-epoch",
        type=int,
        default=8,
        help="Disable intermediate distillation after this epoch (0 keeps it enabled for all epochs).",
    )
    parser.add_argument(
        "--intermediate-distill-on-val",
        action="store_true",
        help="Also compute intermediate distillation during validation (slower).",
    )
    parser.add_argument(
        "--distill-teacher-temperature",
        type=float,
        default=0.12,
        help="Teacher temperature for KL-like distillation objectives.",
    )
    parser.add_argument(
        "--distill-student-temperature",
        type=float,
        default=0.07,
        help="Student temperature for KL-like distillation objectives.",
    )
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument(
        "--learnable-temperature",
        action="store_true",
        help="Learn contrastive temperature during training.",
    )
    parser.add_argument(
        "--min-temperature",
        type=float,
        default=0.02,
        help="Lower clamp for learned contrastive temperature.",
    )
    parser.add_argument(
        "--max-temperature",
        type=float,
        default=0.2,
        help="Upper clamp for learned contrastive temperature.",
    )
    parser.add_argument("--unfreeze-last-n-blocks", type=int, default=1)
    parser.add_argument("--unfreeze-text-last-n-blocks", type=int, default=0)
    parser.add_argument("--unfreeze-text-tower", action="store_true")
    parser.add_argument("--online-student-text", action="store_true")
    parser.add_argument("--train-logit-scale", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action="store_true",
        help="Enable DataLoader pin_memory explicitly (default: auto-on for CUDA).",
    )
    parser.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="Disable DataLoader pin_memory explicitly.",
    )
    parser.set_defaults(pin_memory=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Gradient clipping max norm. Set <= 0 to disable.",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "cosine"],
        default="cosine",
        help="Learning-rate scheduler type.",
    )
    parser.add_argument("--warmup-steps", type=int, default=600)
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-7,
        help="Minimum learning rate reached by cosine scheduler.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.05,
        help="Fraction of samples held out for validation metrics (0 disables).",
    )
    parser.add_argument(
        "--val-every-epochs",
        type=int,
        default=1,
        help="Evaluate validation metrics every N epochs.",
    )
    parser.add_argument(
        "--best-checkpoint-metric",
        choices=["train-loss", "val-loss", "val-recall@10"],
        default="val-recall@10",
        help="Metric used to choose best_loss_checkpoint.pt.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=3,
        help=(
            "Stop training when the best-checkpoint metric does not improve for N validation "
            "epochs. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--val-recall-image-chunk-size",
        type=int,
        default=256,
        help="Validation Recall@10 image chunk size.",
    )
    parser.add_argument(
        "--val-recall-text-chunk-size",
        type=int,
        default=8192,
        help="Validation Recall@10 text chunk size.",
    )
    parser.add_argument(
        "--val-recall-max-texts",
        type=int,
        default=300000,
        help="Cap candidate texts used by Validation Recall@10 (0 means all).",
    )
    parser.add_argument(
        "--val-recall-source",
        choices=["all", "coco", "coco2014", "coco2017", "vg"],
        default="coco",
        help="Restrict validation Recall@10 to specific image sources.",
    )
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--save-epoch-checkpoints", action="store_true")
    parser.add_argument("--seed", type=int, default=1919810)
    parser.add_argument(
        "--strict-teacher-coverage",
        action="store_true",
        help="Fail if any training image is missing in the teacher embedding cache.",
    )
    parser.add_argument(
        "--missing-teacher-log",
        help="Optional file to save image names skipped due to missing teacher embeddings.",
    )
    parser.add_argument(
        "--device",
        default="cuda:1" if torch.cuda.is_available() else "cpu",
        help="Device spec, e.g. cuda:1 or comma-separated cuda:0,cuda:1 for DataParallel.",
    )
    args = parser.parse_args()
    if getattr(args, "clipkd_preset", "none") == "bigG14-fd-icl":
        apply_bigG14_fd_icl_preset(args)
    elif getattr(args, "clipkd_preset", "none") == "clipkd-upstream-bigg14":
        apply_clipkd_upstream_bigg14_preset(args)
    elif getattr(args, "clipkd_preset", "none") == "clipkd-upstream-h14":
        apply_clipkd_upstream_h14_preset(args)
    else:
        apply_clipkd_preset(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.grad_accumulation < 1:
        raise ValueError("--grad-accumulation must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.adam_beta1 <= 0 or args.adam_beta1 >= 1:
        raise ValueError("--adam-beta1 must be in (0, 1)")
    if args.adam_beta2 <= 0 or args.adam_beta2 >= 1:
        raise ValueError("--adam-beta2 must be in (0, 1)")
    if args.adam_eps <= 0:
        raise ValueError("--adam-eps must be > 0")

    if (
        args.contrastive_weight <= 0
        and args.distill_weight <= 0
        and args.teacher_cosine_weight <= 0
        and args.baseline_anchor_weight <= 0
        and args.hard_negative_weight <= 0
        and args.relation_distill_weight <= 0
        and args.crd_weight <= 0
        and args.icl_weight <= 0
        and args.feature_distill_weight <= 0
        and args.masked_feature_distill_weight <= 0
        and args.gradient_distill_weight <= 0
        and args.augmented_feature_distill_weight <= 0
        and args.clipkd_ckd_weight <= 0
        and args.clipkd_cross_kd_weight <= 0
    ):
        raise ValueError("At least one loss weight must be > 0.")

    icl_final_weight = args.icl_weight if args.icl_final_weight is None else args.icl_final_weight
    if (args.icl_weight > 0 or icl_final_weight > 0) and not args.teacher_text_embeddings:
        raise ValueError("--teacher-text-embeddings is required when ICL distillation is enabled")
    if args.icl_teacher_temperature <= 0:
        raise ValueError("--icl-teacher-temperature must be > 0")

    clipkd_ckd_final_weight = args.clipkd_ckd_weight if args.clipkd_ckd_final_weight is None else args.clipkd_ckd_final_weight
    clipkd_cross_final_weight = (
        args.clipkd_cross_kd_weight if args.clipkd_cross_kd_final_weight is None else args.clipkd_cross_kd_final_weight
    )
    if (args.clipkd_ckd_weight > 0 or clipkd_ckd_final_weight > 0 or args.clipkd_cross_kd_weight > 0 or clipkd_cross_final_weight > 0) and not args.teacher_text_embeddings:
        raise ValueError("--teacher-text-embeddings is required when CLIP-KD logits losses are enabled")

    if args.val_split < 0 or args.val_split >= 1:
        raise ValueError("--val-split must be in [0, 1).")
    if args.val_every_epochs < 1:
        raise ValueError("--val-every-epochs must be >= 1")
    if args.lr_scheduler == "cosine" and args.lr <= 0:
        raise ValueError("--lr must be > 0 when using cosine scheduler")
    if args.val_recall_image_chunk_size < 1:
        raise ValueError("--val-recall-image-chunk-size must be >= 1")
    if args.val_recall_text_chunk_size < 1:
        raise ValueError("--val-recall-text-chunk-size must be >= 1")
    if args.val_recall_max_texts < 0:
        raise ValueError("--val-recall-max-texts must be >= 0")
    if args.early_stop_patience < 0:
        raise ValueError("--early-stop-patience must be >= 0")

    if args.min_temperature <= 0 or args.max_temperature <= 0 or args.min_temperature >= args.max_temperature:
        raise ValueError("Temperature bounds must satisfy 0 < --min-temperature < --max-temperature")

    if args.memory_bank_size < 0:
        raise ValueError("--memory-bank-size must be >= 0")
    if args.memory_bank_distill_weight > 0 and args.memory_bank_size <= 0:
        raise ValueError("--memory-bank-size must be > 0 when --memory-bank-distill-weight > 0")
    if args.memory_bank_min_samples < 1:
        raise ValueError("--memory-bank-min-samples must be >= 1")
    if args.distill_teacher_temperature <= 0 or args.distill_student_temperature <= 0:
        raise ValueError("--distill-teacher-temperature and --distill-student-temperature must be > 0")
    if args.intermediate_distill_num_blocks < 1:
        raise ValueError("--intermediate-distill-num-blocks must be >= 1")
    if args.intermediate_distill_frequency < 1:
        raise ValueError("--intermediate-distill-frequency must be >= 1")
    if args.intermediate_distill_stop_epoch < 0:
        raise ValueError("--intermediate-distill-stop-epoch must be >= 0")
    if args.masked_feature_keep_ratio <= 0 or args.masked_feature_keep_ratio > 1:
        raise ValueError("--masked-feature-keep-ratio must be in (0, 1]")
    if args.augmented_feature_noise_std < 0:
        raise ValueError("--augmented-feature-noise-std must be >= 0")
    if args.unfreeze_text_last_n_blocks < 0:
        raise ValueError("--unfreeze-text-last-n-blocks must be >= 0")
    if args.clipkd_ckd_weight < 0:
        raise ValueError("--clipkd-ckd-weight must be >= 0")
    if args.clipkd_cross_kd_weight < 0:
        raise ValueError("--clipkd-cross-kd-weight must be >= 0")
    if args.clipkd_ckd_final_weight is not None and args.clipkd_ckd_final_weight < 0:
        raise ValueError("--clipkd-ckd-final-weight must be >= 0")
    if args.clipkd_cross_kd_final_weight is not None and args.clipkd_cross_kd_final_weight < 0:
        raise ValueError("--clipkd-cross-kd-final-weight must be >= 0")
