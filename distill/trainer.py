from __future__ import annotations

import copy
import json
import math
from contextlib import nullcontext
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.student_model import (
    StudentImageModel,
    configure_trainable_student,
    count_trainable_parameters,
    get_trainable_parameters,
)
from utils.track1_utils import load_image_embedding_cache, load_text_embedding_cache

from .args import validate_args
from .data import (
    Track1DistillDataset,
    build_hard_negative_lookup,
    build_train_val_subsets,
    configure_cuda_runtime,
    matches_source_filter,
    resolve_num_workers,
)
from .features import (
    BlockFeatureRecorder,
    compute_intermediate_relation_loss,
    get_vision_blocks,
    resolve_resume_checkpoint_path,
    update_memory_bank,
)
from .losses import compute_losses, evaluate_val_recall_at_k
from .optim import build_cosine_scheduler, interpolate_weight


def is_openclip_compatible_model_id(model_id: str) -> bool:
    lowered = model_id.lower()
    return not (lowered.startswith("google/siglip") or lowered.startswith("google/siglip2"))


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_training_checkpoint(
    checkpoint_path: Path,
    *,
    student_model: nn.Module,
    text_cache: dict[str, object] | None,
    teacher_cache: dict[str, object],
    trainable_modules: list[str],
    args,
    history: list[dict[str, float]],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    best_total_loss: float | None = None,
    best_metric_name: str | None = None,
    best_metric_value: float | None = None,
    epoch: int | None = None,
    contrastive_log_temperature: torch.Tensor | None = None,
) -> None:
    payload = {
        "student_model_id": args.student_model_id,
        "student_state_dict": student_model.state_dict(),
        "text_embedding_model_name": text_cache["model_name"] if text_cache is not None else None,
        "teacher_model_name": teacher_cache["model_name"],
        "trainable_modules": trainable_modules,
        "args": vars(args),
        "history": history,
        "epoch": epoch,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if best_total_loss is not None:
        payload["best_total_loss"] = float(best_total_loss)
    if best_metric_name is not None:
        payload["best_metric_name"] = best_metric_name
    if best_metric_value is not None:
        payload["best_metric_value"] = float(best_metric_value)
    if contrastive_log_temperature is not None:
        payload["contrastive_log_temperature"] = contrastive_log_temperature.detach().cpu()
    torch.save(payload, checkpoint_path)


def run_training(args) -> None:
    validate_args(args)

    contrastive_final_weight = args.contrastive_weight if args.contrastive_final_weight is None else args.contrastive_final_weight
    distill_final_weight = args.distill_weight if args.distill_final_weight is None else args.distill_final_weight
    teacher_cosine_final_weight = args.teacher_cosine_weight if args.teacher_cosine_final_weight is None else args.teacher_cosine_final_weight
    hard_negative_final_weight = args.hard_negative_weight if args.hard_negative_final_weight is None else args.hard_negative_final_weight
    relation_distill_final_weight = args.relation_distill_weight if args.relation_distill_final_weight is None else args.relation_distill_final_weight
    crd_final_weight = args.crd_weight if args.crd_final_weight is None else args.crd_final_weight
    icl_final_weight = args.icl_weight if args.icl_final_weight is None else args.icl_final_weight
    memory_bank_distill_final_weight = (
        args.memory_bank_distill_weight if args.memory_bank_distill_final_weight is None else args.memory_bank_distill_final_weight
    )
    backbone_feature_distill_final_weight = (
        args.backbone_feature_distill_weight
        if args.backbone_feature_distill_final_weight is None
        else args.backbone_feature_distill_final_weight
    )
    feature_distill_final_weight = (
        args.feature_distill_weight if args.feature_distill_final_weight is None else args.feature_distill_final_weight
    )
    masked_feature_distill_final_weight = (
        args.masked_feature_distill_weight
        if args.masked_feature_distill_final_weight is None
        else args.masked_feature_distill_final_weight
    )
    gradient_distill_final_weight = (
        args.gradient_distill_weight if args.gradient_distill_final_weight is None else args.gradient_distill_final_weight
    )
    augmented_feature_distill_final_weight = (
        args.augmented_feature_distill_weight
        if args.augmented_feature_distill_final_weight is None
        else args.augmented_feature_distill_final_weight
    )
    intermediate_distill_final_weight = (
        args.intermediate_distill_weight if args.intermediate_distill_final_weight is None else args.intermediate_distill_final_weight
    )

    hard_negative_enabled = args.hard_negative_weight > 0 or hard_negative_final_weight > 0
    text_supervision_enabled = args.contrastive_weight > 0 or contrastive_final_weight > 0 or hard_negative_enabled
    if text_supervision_enabled and not args.text_embeddings:
        raise ValueError("Text supervision is enabled but --text-embeddings is missing.")
    if hard_negative_enabled and not args.hard_negatives:
        raise ValueError("--hard-negatives is required when hard-negative loss is enabled")
    if hard_negative_enabled and args.num_hard_negatives < 1:
        raise ValueError("--num-hard-negatives must be >= 1 when hard-negative loss is enabled")

    set_seed(args.seed)
    configure_cuda_runtime(device=args.device, allow_tf32=not args.no_tf32)
    num_workers = resolve_num_workers(args.num_workers, args.device)

    resume_payload = None
    resume_epoch = 0
    history: list[dict[str, float]] = []
    if args.resume_checkpoint:
        resume_path = resolve_resume_checkpoint_path(args.resume_checkpoint)
        resume_payload = torch.load(resume_path, map_location="cpu")
        resume_epoch = int(resume_payload.get("epoch", 0) or 0)
        history = list(resume_payload.get("history", []))
        checkpoint_model_id = resume_payload.get("student_model_id")
        if checkpoint_model_id:
            if checkpoint_model_id != args.student_model_id:
                print(
                    "Resume checkpoint model_id differs from --student-model-id; "
                    f"using checkpoint model_id: {checkpoint_model_id}"
                )
            args.student_model_id = checkpoint_model_id

    text_cache = load_text_embedding_cache(args.text_embeddings) if text_supervision_enabled else None
    teacher_text_cache = load_text_embedding_cache(args.teacher_text_embeddings) if args.teacher_text_embeddings else None
    teacher_cache = load_image_embedding_cache(args.teacher_embeddings)
    hard_negative_lookup = build_hard_negative_lookup(args.hard_negatives)

    model, preprocess_train, _ = open_clip.create_model_and_transforms(args.student_model_id)
    model = model.to(args.device)
    reference_model = StudentImageModel(copy.deepcopy(model)).to(args.device).eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad = False
    student_model = StudentImageModel(model).to(args.device)

    if args.channels_last and args.device.startswith("cuda"):
        reference_model = reference_model.to(memory_format=torch.channels_last)
        student_model = student_model.to(memory_format=torch.channels_last)

    trainable_modules = configure_trainable_student(student_model, unfreeze_last_n_blocks=args.unfreeze_last_n_blocks)

    if resume_payload is not None:
        load_result = student_model.load_state_dict(resume_payload["student_state_dict"], strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            print(
                "Resume checkpoint compatibility: "
                f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
            )
        print(f"Resuming from checkpoint: {args.resume_checkpoint} (epoch={resume_epoch})")

    if args.gradient_checkpointing and hasattr(student_model.clip_model, "set_grad_checkpointing"):
        student_model.clip_model.set_grad_checkpointing(True)

    intermediate_distill_enabled = args.intermediate_distill_weight > 0 or intermediate_distill_final_weight > 0
    teacher_intermediate_model = None
    student_block_recorder = None
    teacher_block_recorder = None
    if intermediate_distill_enabled and not is_openclip_compatible_model_id(args.intermediate_teacher_model_id):
        print(
            "Intermediate distillation disabled: "
            f"model '{args.intermediate_teacher_model_id}' is not OpenCLIP-compatible. "
            "Use an OpenCLIP model id or set intermediate distillation weight to 0."
        )
        intermediate_distill_enabled = False

    if intermediate_distill_enabled:
        try:
            teacher_intermediate_model, _, _ = open_clip.create_model_and_transforms(args.intermediate_teacher_model_id)
        except Exception as exc:  # pragma: no cover - defensive runtime fallback
            print(
                "Intermediate distillation disabled: "
                f"failed to load teacher model '{args.intermediate_teacher_model_id}' with OpenCLIP ({exc})."
            )
            intermediate_distill_enabled = False

    if intermediate_distill_enabled:
        teacher_intermediate_model = teacher_intermediate_model.to(args.device).eval()
        for parameter in teacher_intermediate_model.parameters():
            parameter.requires_grad = False

        student_blocks = get_vision_blocks(student_model.clip_model)
        teacher_blocks = get_vision_blocks(teacher_intermediate_model)
        if not student_blocks or not teacher_blocks:
            raise ValueError("Unable to find vision blocks for intermediate distillation")

        student_hooks = student_blocks[-args.intermediate_distill_num_blocks :]
        teacher_hooks = teacher_blocks[-args.intermediate_distill_num_blocks :]
        if len(student_hooks) != len(teacher_hooks):
            min_count = min(len(student_hooks), len(teacher_hooks))
            student_hooks = student_hooks[-min_count:]
            teacher_hooks = teacher_hooks[-min_count:]
            if min_count == 0:
                raise ValueError("No matching blocks available for intermediate distillation")

        student_block_recorder = BlockFeatureRecorder(student_hooks)
        teacher_block_recorder = BlockFeatureRecorder(teacher_hooks)

    text_ids = None
    text_embeddings = None
    teacher_text_ids = None
    teacher_text_embeddings = None
    if text_cache is not None:
        text_ids = torch.as_tensor(text_cache["text_ids"], dtype=torch.long)
        text_embeddings = torch.as_tensor(text_cache["embeddings"], dtype=torch.float32)
    if teacher_text_cache is not None:
        teacher_text_ids = torch.as_tensor(teacher_text_cache["text_ids"], dtype=torch.long)
        teacher_text_embeddings = torch.as_tensor(teacher_text_cache["embeddings"], dtype=torch.float32)

    if text_supervision_enabled and text_embeddings is not None:
        text_embedding_dim = int(text_embeddings.shape[1])
        if student_model.embedding_dim != text_embedding_dim:
            raise ValueError(
                "Text embedding dimension mismatch: "
                f"student_dim={student_model.embedding_dim}, text_dim={text_embedding_dim}. "
                "Regenerate text embeddings using the same backbone."
            )

    teacher_embeddings = torch.as_tensor(teacher_cache["embeddings"], dtype=torch.float32)

    dataset = Track1DistillDataset(
        img_list_path=args.img_list,
        image_folder=args.image_folder,
        preprocess=preprocess_train,
        text_ids=text_ids,
        text_embeddings=text_embeddings,
        teacher_text_ids=teacher_text_ids,
        teacher_text_embeddings=teacher_text_embeddings,
        teacher_image_names=list(teacher_cache["image_names"]),
        teacher_embeddings=teacher_embeddings,
        hard_negative_lookup=hard_negative_lookup,
        num_hard_negatives=args.num_hard_negatives if hard_negative_enabled else 0,
        strict_teacher_coverage=args.strict_teacher_coverage,
        positive_pooling=args.positive_pooling,
    )

    if args.train_source != "all":
        original_sample_count = len(dataset.samples)
        dataset.samples = [
            sample for sample in dataset.samples if matches_source_filter(str(sample.get("image_name", "")), args.train_source)
        ]
        print(
            "Applied training source filter: "
            f"source={args.train_source}, kept={len(dataset.samples)}/{original_sample_count}"
        )

    if len(dataset) == 0:
        raise ValueError("No valid training samples were constructed from the dataset.")

    if dataset.missing_teacher_images and args.missing_teacher_log:
        missing_log_path = Path(args.missing_teacher_log)
        missing_log_path.parent.mkdir(parents=True, exist_ok=True)
        missing_log_path.write_text("\n".join(dataset.missing_teacher_images) + "\n", encoding="utf-8")

    train_dataset, val_dataset = build_train_val_subsets(dataset, val_split=args.val_split, seed=args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=num_workers > 0,
        prefetch_factor=args.prefetch_factor if num_workers > 0 else None,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=args.device.startswith("cuda"),
            persistent_workers=num_workers > 0,
            prefetch_factor=args.prefetch_factor if num_workers > 0 else None,
        )

    contrastive_log_temperature = torch.nn.Parameter(
        torch.log(torch.tensor(args.temperature, dtype=torch.float32, device=args.device)),
        requires_grad=args.learnable_temperature,
    )
    if resume_payload is not None and "contrastive_log_temperature" in resume_payload:
        contrastive_log_temperature.data.copy_(resume_payload["contrastive_log_temperature"].to(args.device))

    optimizer_parameters = list(get_trainable_parameters(student_model))
    if args.learnable_temperature:
        optimizer_parameters.append(contrastive_log_temperature)

    optimizer = torch.optim.AdamW(optimizer_parameters, lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.lr_scheduler == "cosine":
        estimated_steps_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accumulation))
        total_optimizer_steps = max(1, estimated_steps_per_epoch * max(1, args.epochs))
        min_lr_scale = args.min_lr / args.lr
        scheduler = build_cosine_scheduler(
            optimizer,
            total_steps=total_optimizer_steps,
            warmup_steps=args.warmup_steps,
            min_lr_scale=min_lr_scale,
        )

    scaler_enabled = args.device.startswith("cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    if args.resume_optimizer_state and resume_payload is not None:
        if resume_payload.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        if scheduler is not None and resume_payload.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        if resume_payload.get("scaler_state_dict") is not None:
            scaler.load_state_dict(resume_payload["scaler_state_dict"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_total_loss = min((metric.get("total_loss", float("inf")) for metric in history), default=float("inf"))
    if resume_payload is not None and "best_total_loss" in resume_payload:
        best_total_loss = min(best_total_loss, float(resume_payload["best_total_loss"]))

    can_compute_val_recall = val_loader is not None and text_embeddings is not None and text_supervision_enabled

    effective_best_checkpoint_metric = args.best_checkpoint_metric
    if effective_best_checkpoint_metric == "val-recall@10" and not can_compute_val_recall:
        effective_best_checkpoint_metric = "val-loss" if val_loader is not None else "train-loss"

    if effective_best_checkpoint_metric == "val-loss":
        checkpoint_metric_name = "val_total_loss"
        checkpoint_metric_mode = "min"
    elif effective_best_checkpoint_metric == "val-recall@10":
        checkpoint_metric_name = "val_recall_at_10"
        checkpoint_metric_mode = "max"
    else:
        checkpoint_metric_name = "total_loss"
        checkpoint_metric_mode = "min"

    best_checkpoint_metric = float("inf") if checkpoint_metric_mode == "min" else -float("inf")

    print(f"Training samples: {len(train_dataset)}")
    if val_dataset is not None:
        print(f"Validation samples: {len(val_dataset)}")
    print(f"Trainable modules: {', '.join(trainable_modules)}")
    print(f"Trainable parameters: {count_trainable_parameters(student_model):,}")

    student_model.train()
    memory_bank_student = torch.empty((0, student_model.embedding_dim), device=args.device, dtype=torch.float32)
    memory_bank_teacher = torch.empty((0, teacher_embeddings.shape[1]), device=args.device, dtype=torch.float32)

    total_target_epoch = resume_epoch + args.epochs
    for epoch in range(args.epochs):
        current_epoch = resume_epoch + epoch + 1
        current_anchor_weight = interpolate_weight(
            start=args.baseline_anchor_weight,
            end=args.baseline_anchor_final_weight,
            step=epoch,
            total_steps=max(1, args.epochs),
        )
        current_contrastive_weight = interpolate_weight(
            start=args.contrastive_weight,
            end=contrastive_final_weight,
            step=epoch,
            total_steps=max(1, args.epochs),
        )
        current_distill_weight = interpolate_weight(
            start=args.distill_weight,
            end=distill_final_weight,
            step=epoch,
            total_steps=max(1, args.epochs),
        )
        current_teacher_cosine_weight = interpolate_weight(
            start=args.teacher_cosine_weight,
            end=teacher_cosine_final_weight,
            step=epoch,
            total_steps=max(1, args.epochs),
        )
        current_hard_negative_weight = interpolate_weight(
            start=args.hard_negative_weight,
            end=hard_negative_final_weight,
            step=epoch,
            total_steps=max(1, args.epochs),
        )
        current_relation_distill_weight = interpolate_weight(
            start=args.relation_distill_weight,
            end=relation_distill_final_weight,
            step=epoch,
            total_steps=max(1, args.epochs),
        )
        current_crd_weight = interpolate_weight(
            start=args.crd_weight,
            end=crd_final_weight,
            step=epoch,
            total_steps=max(1, args.epochs),
        )
        current_icl_weight = interpolate_weight(
            start=args.icl_weight,
            end=icl_final_weight,
            step=epoch,
            total_steps=max(1, args.epochs),
        )
        current_memory_bank_distill_weight = interpolate_weight(
            start=args.memory_bank_distill_weight,
            end=memory_bank_distill_final_weight,
            step=epoch,
            total_steps=max(1, args.epochs),
        )
        current_backbone_feature_distill_weight = interpolate_weight(
            start=args.backbone_feature_distill_weight,
            end=backbone_feature_distill_final_weight,
            step=epoch,
            total_steps=max(1, args.epochs),
        )
        current_feature_distill_weight = interpolate_weight(
            start=args.feature_distill_weight,
            end=feature_distill_final_weight,
            step=epoch,
            total_steps=max(1, args.epochs),
        )
        current_masked_feature_distill_weight = interpolate_weight(
            start=args.masked_feature_distill_weight,
            end=masked_feature_distill_final_weight,
            step=epoch,
            total_steps=max(1, args.epochs),
        )
        current_gradient_distill_weight = interpolate_weight(
            start=args.gradient_distill_weight,
            end=gradient_distill_final_weight,
            step=epoch,
            total_steps=max(1, args.epochs),
        )
        current_augmented_feature_distill_weight = interpolate_weight(
            start=args.augmented_feature_distill_weight,
            end=augmented_feature_distill_final_weight,
            step=epoch,
            total_steps=max(1, args.epochs),
        )
        current_intermediate_distill_weight = interpolate_weight(
            start=args.intermediate_distill_weight,
            end=intermediate_distill_final_weight,
            step=epoch,
            total_steps=max(1, args.epochs),
        )

        optimizer.zero_grad(set_to_none=True)
        running = {
            "contrastive_loss": 0.0,
            "distill_loss": 0.0,
            "teacher_cosine_loss": 0.0,
            "baseline_anchor_loss": 0.0,
            "hard_negative_loss": 0.0,
            "relation_distill_loss": 0.0,
            "crd_loss": 0.0,
            "icl_loss": 0.0,
            "memory_bank_distill_loss": 0.0,
            "backbone_feature_distill_loss": 0.0,
            "feature_distill_loss": 0.0,
            "masked_feature_distill_loss": 0.0,
            "gradient_distill_loss": 0.0,
            "augmented_feature_distill_loss": 0.0,
            "intermediate_distill_loss": 0.0,
            "total_loss": 0.0,
        }
        optimizer_steps = 0

        progress = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {current_epoch}/{total_target_epoch}", dynamic_ncols=True)

        for step, batch in enumerate(progress, start=1):
            pixel_values = batch["pixel_values"].to(args.device, non_blocking=True)
            if args.channels_last and args.device.startswith("cuda"):
                pixel_values = pixel_values.contiguous(memory_format=torch.channels_last)

            positive_embeddings = None
            if text_supervision_enabled:
                positive_embeddings = F.normalize(batch["positive_embedding"].to(args.device, non_blocking=True), dim=-1)

            teacher_targets = batch["teacher_embedding"].to(args.device, non_blocking=True)
            teacher_positive_embeddings = None
            if teacher_text_embeddings is not None:
                teacher_positive_embeddings = F.normalize(
                    batch["teacher_positive_embedding"].to(args.device, non_blocking=True),
                    dim=-1,
                )

            negative_embeddings = None
            negative_mask = None
            if current_hard_negative_weight > 0:
                negative_embeddings = F.normalize(batch["negative_embeddings"].to(args.device, non_blocking=True), dim=-1)
                negative_mask = batch["negative_mask"].to(args.device, non_blocking=True)

            autocast_context = torch.autocast(device_type="cuda", enabled=True) if args.device.startswith("cuda") else nullcontext()
            apply_intermediate_distill = (
                intermediate_distill_enabled
                and current_intermediate_distill_weight > 0
                and (step % args.intermediate_distill_frequency == 0)
                and (args.intermediate_distill_stop_epoch == 0 or current_epoch <= args.intermediate_distill_stop_epoch)
            )
            if intermediate_distill_enabled:
                student_block_recorder.clear()
                teacher_block_recorder.clear()

            with autocast_context:
                with torch.no_grad():
                    if apply_intermediate_distill:
                        teacher_intermediate_model.encode_image(pixel_values)
                    reference_backbone_features = reference_model.encode_backbone(pixel_values)
                    reference_embeddings = reference_model.projection_head(reference_backbone_features)

                student_backbone_features = student_model.encode_backbone(pixel_values)
                student_embeddings = student_model.projection_head(student_backbone_features)

                current_temperature = contrastive_log_temperature.exp().clamp(min=args.min_temperature, max=args.max_temperature)
                loss, metrics = compute_losses(
                    student_embeddings=student_embeddings,
                    reference_embeddings=reference_embeddings,
                    positive_embeddings=positive_embeddings,
                    teacher_positive_embeddings=teacher_positive_embeddings,
                    teacher_embeddings=teacher_targets,
                    negative_embeddings=negative_embeddings,
                    negative_mask=negative_mask,
                    temperature=float(current_temperature.detach()),
                    contrastive_weight=current_contrastive_weight,
                    contrastive_loss_type=args.contrastive_loss_type,
                    distill_weight=current_distill_weight,
                    distill_loss_type=args.distill_loss_type,
                    teacher_cosine_weight=current_teacher_cosine_weight,
                    baseline_anchor_weight=current_anchor_weight,
                    hard_negative_weight=current_hard_negative_weight,
                    hard_negative_margin=args.hard_negative_margin,
                    hard_negative_weighting=args.hard_negative_weighting,
                    hard_negative_softmax_temperature=args.hard_negative_softmax_temperature,
                    relation_distill_weight=current_relation_distill_weight,
                    relation_distill_temperature=args.relation_distill_temperature,
                    crd_weight=current_crd_weight,
                    icl_weight=current_icl_weight,
                    icl_teacher_temperature=args.icl_teacher_temperature,
                    memory_bank_distill_weight=current_memory_bank_distill_weight,
                    memory_bank_distill_temperature=args.memory_bank_distill_temperature,
                    memory_bank_student=memory_bank_student,
                    memory_bank_teacher=memory_bank_teacher,
                    memory_bank_min_samples=args.memory_bank_min_samples,
                    backbone_feature_distill_weight=current_backbone_feature_distill_weight,
                    student_backbone_features=student_backbone_features,
                    reference_backbone_features=reference_backbone_features,
                    distill_teacher_temperature=args.distill_teacher_temperature,
                    distill_student_temperature=args.distill_student_temperature,
                    feature_distill_weight=current_feature_distill_weight,
                    masked_feature_distill_weight=current_masked_feature_distill_weight,
                    masked_feature_keep_ratio=args.masked_feature_keep_ratio,
                    gradient_distill_weight=current_gradient_distill_weight,
                    augmented_feature_distill_weight=current_augmented_feature_distill_weight,
                    augmented_feature_noise_std=args.augmented_feature_noise_std,
                )

                intermediate_distill_loss = torch.zeros((), device=args.device)
                if apply_intermediate_distill:
                    intermediate_distill_loss = compute_intermediate_relation_loss(
                        student_block_outputs=student_block_recorder.outputs,
                        teacher_block_outputs=teacher_block_recorder.outputs,
                    )
                loss = loss + current_intermediate_distill_weight * intermediate_distill_loss
                metrics["intermediate_distill_loss"] = float(intermediate_distill_loss.detach().cpu())
                metrics["total_loss"] = float(loss.detach().cpu())
                scaled_loss = loss / args.grad_accumulation

            scaler.scale(scaled_loss).backward()

            if step % args.grad_accumulation == 0 or step == len(train_loader):
                if args.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(list(get_trainable_parameters(student_model)), max_norm=args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

            memory_bank_student = update_memory_bank(
                memory_bank_student,
                F.normalize(student_embeddings.detach().float(), dim=-1),
                max_size=args.memory_bank_size,
            )
            memory_bank_teacher = update_memory_bank(
                memory_bank_teacher,
                F.normalize(teacher_targets.detach().float(), dim=-1),
                max_size=args.memory_bank_size,
            )

            for key, value in metrics.items():
                running[key] += value

            progress.set_postfix(
                {
                    "loss": f"{metrics['total_loss']:.4f}",
                    "dist": f"{metrics['distill_loss']:.4f}",
                    "ctr": f"{metrics['contrastive_loss']:.4f}",
                    "hneg": f"{metrics['hard_negative_loss']:.4f}",
                    "rel": f"{metrics['relation_distill_loss']:.4f}",
                    "crd": f"{metrics['crd_loss']:.4f}",
                    "icl": f"{metrics['icl_loss']:.4f}",
                    "mb": f"{metrics['memory_bank_distill_loss']:.4f}",
                    "temp": f"{float(current_temperature.detach()):.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
            )

        progress.close()

        epoch_metrics = {key: value / len(train_loader) for key, value in running.items()}
        epoch_metrics["epoch"] = current_epoch
        epoch_metrics["optimizer_steps"] = optimizer_steps
        epoch_metrics["lr"] = float(optimizer.param_groups[0]["lr"])
        epoch_metrics["anchor_weight"] = float(current_anchor_weight)
        epoch_metrics["contrastive_weight"] = float(current_contrastive_weight)
        epoch_metrics["distill_weight"] = float(current_distill_weight)
        epoch_metrics["teacher_cosine_weight"] = float(current_teacher_cosine_weight)
        epoch_metrics["hard_negative_weight"] = float(current_hard_negative_weight)
        epoch_metrics["relation_distill_weight"] = float(current_relation_distill_weight)
        epoch_metrics["crd_weight"] = float(current_crd_weight)
        epoch_metrics["icl_weight"] = float(current_icl_weight)
        epoch_metrics["memory_bank_distill_weight"] = float(current_memory_bank_distill_weight)
        epoch_metrics["backbone_feature_distill_weight"] = float(current_backbone_feature_distill_weight)
        epoch_metrics["feature_distill_weight"] = float(current_feature_distill_weight)
        epoch_metrics["masked_feature_distill_weight"] = float(current_masked_feature_distill_weight)
        epoch_metrics["gradient_distill_weight"] = float(current_gradient_distill_weight)
        epoch_metrics["augmented_feature_distill_weight"] = float(current_augmented_feature_distill_weight)
        epoch_metrics["intermediate_distill_weight"] = float(current_intermediate_distill_weight)
        epoch_metrics["contrastive_temperature"] = float(
            contrastive_log_temperature.exp().clamp(min=args.min_temperature, max=args.max_temperature).detach().cpu()
        )

        should_run_validation = val_loader is not None and (
            current_epoch % args.val_every_epochs == 0 or current_epoch == total_target_epoch
        )
        if should_run_validation:
            student_model.eval()
            val_running = {key: 0.0 for key in running}
            val_recall_embeddings: list[torch.Tensor] = []
            val_recall_sample_indices: list[torch.Tensor] = []

            with torch.inference_mode():
                for val_batch in val_loader:
                    pixel_values = val_batch["pixel_values"].to(args.device, non_blocking=True)
                    if args.channels_last and args.device.startswith("cuda"):
                        pixel_values = pixel_values.contiguous(memory_format=torch.channels_last)

                    positive_embeddings = None
                    if text_supervision_enabled:
                        positive_embeddings = F.normalize(val_batch["positive_embedding"].to(args.device, non_blocking=True), dim=-1)

                    teacher_targets = val_batch["teacher_embedding"].to(args.device, non_blocking=True)
                    teacher_positive_embeddings = None
                    if teacher_text_embeddings is not None:
                        teacher_positive_embeddings = F.normalize(
                            val_batch["teacher_positive_embedding"].to(args.device, non_blocking=True),
                            dim=-1,
                        )

                    negative_embeddings = None
                    negative_mask = None
                    if current_hard_negative_weight > 0:
                        negative_embeddings = F.normalize(val_batch["negative_embeddings"].to(args.device, non_blocking=True), dim=-1)
                        negative_mask = val_batch["negative_mask"].to(args.device, non_blocking=True)

                    autocast_context = torch.autocast(device_type="cuda", enabled=True) if args.device.startswith("cuda") else nullcontext()
                    apply_intermediate_distill_val = (
                        intermediate_distill_enabled
                        and args.intermediate_distill_on_val
                        and current_intermediate_distill_weight > 0
                        and (args.intermediate_distill_stop_epoch == 0 or current_epoch <= args.intermediate_distill_stop_epoch)
                    )
                    if intermediate_distill_enabled:
                        student_block_recorder.clear()
                        teacher_block_recorder.clear()

                    with autocast_context:
                        if apply_intermediate_distill_val:
                            teacher_intermediate_model.encode_image(pixel_values)
                        reference_backbone_features = reference_model.encode_backbone(pixel_values)
                        reference_embeddings = reference_model.projection_head(reference_backbone_features)
                        student_backbone_features = student_model.encode_backbone(pixel_values)
                        student_embeddings = student_model.projection_head(student_backbone_features)

                        val_loss, val_metrics = compute_losses(
                            student_embeddings=student_embeddings,
                            reference_embeddings=reference_embeddings,
                            positive_embeddings=positive_embeddings,
                            teacher_positive_embeddings=teacher_positive_embeddings,
                            teacher_embeddings=teacher_targets,
                            negative_embeddings=negative_embeddings,
                            negative_mask=negative_mask,
                            temperature=float(
                                contrastive_log_temperature.exp().clamp(min=args.min_temperature, max=args.max_temperature).detach()
                            ),
                            contrastive_weight=current_contrastive_weight,
                            contrastive_loss_type=args.contrastive_loss_type,
                            distill_weight=current_distill_weight,
                            distill_loss_type=args.distill_loss_type,
                            teacher_cosine_weight=current_teacher_cosine_weight,
                            baseline_anchor_weight=current_anchor_weight,
                            hard_negative_weight=current_hard_negative_weight,
                            hard_negative_margin=args.hard_negative_margin,
                            hard_negative_weighting=args.hard_negative_weighting,
                            hard_negative_softmax_temperature=args.hard_negative_softmax_temperature,
                            relation_distill_weight=current_relation_distill_weight,
                            relation_distill_temperature=args.relation_distill_temperature,
                            crd_weight=current_crd_weight,
                            icl_weight=current_icl_weight,
                            icl_teacher_temperature=args.icl_teacher_temperature,
                            memory_bank_distill_weight=0.0,
                            memory_bank_distill_temperature=args.memory_bank_distill_temperature,
                            memory_bank_student=None,
                            memory_bank_teacher=None,
                            memory_bank_min_samples=args.memory_bank_min_samples,
                            backbone_feature_distill_weight=current_backbone_feature_distill_weight,
                            student_backbone_features=student_backbone_features,
                            reference_backbone_features=reference_backbone_features,
                            distill_teacher_temperature=args.distill_teacher_temperature,
                            distill_student_temperature=args.distill_student_temperature,
                            feature_distill_weight=current_feature_distill_weight,
                            masked_feature_distill_weight=current_masked_feature_distill_weight,
                            masked_feature_keep_ratio=args.masked_feature_keep_ratio,
                            gradient_distill_weight=current_gradient_distill_weight,
                            augmented_feature_distill_weight=current_augmented_feature_distill_weight,
                            augmented_feature_noise_std=args.augmented_feature_noise_std,
                        )
                        intermediate_distill_loss = torch.zeros((), device=args.device)
                        if apply_intermediate_distill_val:
                            intermediate_distill_loss = compute_intermediate_relation_loss(
                                student_block_outputs=student_block_recorder.outputs,
                                teacher_block_outputs=teacher_block_recorder.outputs,
                            )
                        val_loss = val_loss + current_intermediate_distill_weight * intermediate_distill_loss
                        val_metrics["intermediate_distill_loss"] = float(intermediate_distill_loss.detach().cpu())
                        val_metrics["total_loss"] = float(val_loss.detach().cpu())

                    if can_compute_val_recall:
                        val_recall_embeddings.append(student_embeddings.detach().float().cpu())
                        val_recall_sample_indices.append(val_batch["sample_index"].detach().long().cpu())

                    for key, value in val_metrics.items():
                        val_running[key] += value

            val_metrics_avg = {f"val_{key}": value / len(val_loader) for key, value in val_running.items()}
            epoch_metrics.update(val_metrics_avg)

            if can_compute_val_recall and val_recall_embeddings and val_recall_sample_indices:
                val_recall_at_10 = evaluate_val_recall_at_k(
                    image_embeddings=torch.cat(val_recall_embeddings, dim=0),
                    sample_indices=torch.cat(val_recall_sample_indices, dim=0),
                    dataset=dataset,
                    text_embeddings=text_embeddings,
                    top_k=10,
                    image_chunk_size=args.val_recall_image_chunk_size,
                    text_chunk_size=args.val_recall_text_chunk_size,
                    max_texts=args.val_recall_max_texts,
                    source_filter=args.val_recall_source,
                    device=args.device,
                )
                if val_recall_at_10 is not None:
                    epoch_metrics["val_recall_at_10"] = float(val_recall_at_10)

            student_model.train()

        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, sort_keys=True))

        if args.save_epoch_checkpoints:
            save_training_checkpoint(
                output_dir / f"student_checkpoint_epoch_{current_epoch:02d}.pt",
                student_model=student_model,
                text_cache=text_cache,
                teacher_cache=teacher_cache,
                trainable_modules=trainable_modules,
                args=args,
                history=history,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_total_loss=best_total_loss,
                best_metric_name=checkpoint_metric_name,
                best_metric_value=best_checkpoint_metric if not math.isinf(best_checkpoint_metric) else None,
                epoch=current_epoch,
                contrastive_log_temperature=contrastive_log_temperature,
            )

        best_total_loss = min(best_total_loss, float(epoch_metrics["total_loss"]))

        checkpoint_metric_value = epoch_metrics.get(checkpoint_metric_name)
        if checkpoint_metric_value is None and checkpoint_metric_name == "val_total_loss":
            checkpoint_metric_value = epoch_metrics["total_loss"]

        should_update_best = False
        if checkpoint_metric_value is not None and not math.isnan(float(checkpoint_metric_value)):
            if checkpoint_metric_mode == "min":
                should_update_best = float(checkpoint_metric_value) < best_checkpoint_metric
            else:
                should_update_best = float(checkpoint_metric_value) > best_checkpoint_metric

        if should_update_best:
            best_checkpoint_metric = float(checkpoint_metric_value)
            save_training_checkpoint(
                output_dir / "best_loss_checkpoint.pt",
                student_model=student_model,
                text_cache=text_cache,
                teacher_cache=teacher_cache,
                trainable_modules=trainable_modules,
                args=args,
                history=history,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_total_loss=best_total_loss,
                best_metric_name=checkpoint_metric_name,
                best_metric_value=best_checkpoint_metric,
                epoch=current_epoch,
                contrastive_log_temperature=contrastive_log_temperature,
            )

    final_epoch = int(history[-1]["epoch"]) if history else total_target_epoch
    save_training_checkpoint(
        output_dir / "student_checkpoint.pt",
        student_model=student_model,
        text_cache=text_cache,
        teacher_cache=teacher_cache,
        trainable_modules=trainable_modules,
        args=args,
        history=history,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        best_total_loss=best_total_loss,
        best_metric_name=checkpoint_metric_name,
        best_metric_value=best_checkpoint_metric if not math.isinf(best_checkpoint_metric) else None,
        epoch=final_epoch,
        contrastive_log_temperature=contrastive_log_temperature,
    )
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    if student_block_recorder is not None:
        student_block_recorder.close()
    if teacher_block_recorder is not None:
        teacher_block_recorder.close()

    print(f"Saved checkpoint to {output_dir / 'student_checkpoint.pt'}")
    print(f"Saved best checkpoint to {output_dir / 'best_loss_checkpoint.pt'}")
