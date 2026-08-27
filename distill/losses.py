from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import Track1DistillDataset


def evaluate_val_recall_at_k(
    *,
    image_embeddings: torch.Tensor,
    sample_indices: torch.Tensor,
    dataset: Track1DistillDataset,
    text_embeddings: torch.Tensor,
    top_k: int,
    image_chunk_size: int,
    text_chunk_size: int,
    max_texts: int,
    source_filter: str,
    device: str,
) -> float | None:
    if image_embeddings.numel() == 0:
        return None

    total_texts = int(text_embeddings.shape[0])
    if total_texts <= 0:
        return None
    candidate_texts = min(total_texts, max_texts) if max_texts > 0 else total_texts
    if candidate_texts <= 0:
        return None

    bounded_top_k = min(max(1, int(top_k)), candidate_texts)
    normalized_images = F.normalize(image_embeddings.float(), dim=-1)

    total_eval = 0
    total_hits = 0

    def should_keep_sample(image_name: str) -> bool:
        if source_filter == "all":
            return True
        if source_filter == "coco":
            return image_name.startswith("coco2014_") or image_name.startswith("coco2017_")
        if source_filter == "coco2014":
            return image_name.startswith("coco2014_")
        if source_filter == "coco2017":
            return image_name.startswith("coco2017_")
        if source_filter == "vg":
            return image_name.startswith("vg_")
        return True

    for image_start in range(0, normalized_images.shape[0], max(1, image_chunk_size)):
        image_end = min(image_start + max(1, image_chunk_size), normalized_images.shape[0])
        image_chunk = normalized_images[image_start:image_end].to(device, non_blocking=True)
        batch_size = image_chunk.shape[0]

        best_scores = torch.full(
            (batch_size, bounded_top_k),
            fill_value=-float("inf"),
            device=device,
            dtype=torch.float32,
        )
        best_indices = torch.full(
            (batch_size, bounded_top_k),
            fill_value=-1,
            device=device,
            dtype=torch.long,
        )

        for text_start in range(0, candidate_texts, max(1, text_chunk_size)):
            text_end = min(text_start + max(1, text_chunk_size), candidate_texts)
            text_chunk = F.normalize(
                text_embeddings[text_start:text_end].to(device, non_blocking=True),
                dim=-1,
            )
            similarity = image_chunk @ text_chunk.T
            local_top_k = min(bounded_top_k, similarity.shape[1])
            if local_top_k <= 0:
                continue

            chunk_scores, chunk_indices = torch.topk(similarity, k=local_top_k, dim=1)
            chunk_indices = chunk_indices + text_start

            merged_scores = torch.cat((best_scores, chunk_scores), dim=1)
            merged_indices = torch.cat((best_indices, chunk_indices), dim=1)
            top_scores, top_positions = torch.topk(merged_scores, k=bounded_top_k, dim=1)
            best_indices = torch.gather(merged_indices, dim=1, index=top_positions)
            best_scores = top_scores

        top_indices_cpu = best_indices.detach().cpu()
        for row in range(batch_size):
            sample_index = int(sample_indices[image_start + row].item())
            image_name = str(dataset.samples[sample_index].get("image_name", ""))
            if not should_keep_sample(image_name):
                continue
            positive_indices = dataset.samples[sample_index]["positive_indices"]
            valid_positives = [index for index in positive_indices if index < candidate_texts]
            if not valid_positives:
                continue

            total_eval += 1
            positive_set = set(valid_positives)
            predicted = top_indices_cpu[row].tolist()
            if any((idx >= 0) and (idx in positive_set) for idx in predicted):
                total_hits += 1

    if total_eval <= 0:
        return None
    return float(total_hits) / float(total_eval)


def compute_losses(
    *,
    student_embeddings: torch.Tensor,
    reference_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor | None,
    teacher_positive_embeddings: torch.Tensor | None,
    teacher_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor | None,
    negative_mask: torch.Tensor | None,
    temperature: float,
    contrastive_weight: float,
    contrastive_loss_type: str,
    distill_weight: float,
    distill_loss_type: str,
    teacher_cosine_weight: float,
    baseline_anchor_weight: float,
    hard_negative_weight: float,
    hard_negative_margin: float,
    hard_negative_weighting: str,
    hard_negative_softmax_temperature: float,
    relation_distill_weight: float,
    relation_distill_temperature: float,
    icl_weight: float,
    icl_teacher_temperature: float,
    memory_bank_distill_weight: float,
    memory_bank_distill_temperature: float,
    memory_bank_student: torch.Tensor | None,
    memory_bank_teacher: torch.Tensor | None,
    memory_bank_min_samples: int,
    backbone_feature_distill_weight: float,
    student_backbone_features: torch.Tensor | None,
    reference_backbone_features: torch.Tensor | None,
    distill_teacher_temperature: float,
    distill_student_temperature: float,
    crd_weight: float,
    feature_distill_weight: float,
    masked_feature_distill_weight: float,
    masked_feature_keep_ratio: float,
    gradient_distill_weight: float,
    augmented_feature_distill_weight: float,
    augmented_feature_noise_std: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    zero = torch.zeros((), device=student_embeddings.device)
    normalized_student = F.normalize(student_embeddings, dim=-1)

    contrastive_loss = zero
    if contrastive_weight > 0:
        if positive_embeddings is None:
            raise ValueError("positive_embeddings are required when contrastive_weight > 0")
        logits = normalized_student @ positive_embeddings.T / max(temperature, 1e-6)
        if contrastive_loss_type == "sigmoid":
            labels = torch.eye(logits.shape[0], device=student_embeddings.device, dtype=logits.dtype)
            contrastive_loss = F.binary_cross_entropy_with_logits(logits, labels)
        else:
            batch_size = student_embeddings.shape[0]
            labels = torch.arange(batch_size, device=student_embeddings.device)
            contrastive_loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

    normalized_teacher = F.normalize(teacher_embeddings, dim=-1)
    distill_loss = zero
    if distill_weight > 0:
        if distill_loss_type == "cosine":
            if normalized_student.shape[-1] == normalized_teacher.shape[-1]:
                distill_loss = 1.0 - F.cosine_similarity(normalized_student, normalized_teacher, dim=-1).mean()
            elif normalized_student.shape[0] > 1:
                batch_size = normalized_student.shape[0]
                off_diagonal = ~torch.eye(batch_size, dtype=torch.bool, device=student_embeddings.device)
                student_rel = (normalized_student @ normalized_student.T)[off_diagonal].view(batch_size, -1)
                teacher_rel = (normalized_teacher @ normalized_teacher.T)[off_diagonal].view(batch_size, -1)
                distill_loss = 1.0 - F.cosine_similarity(student_rel, teacher_rel, dim=-1).mean()
        elif distill_loss_type in {"kl", "kl_sym"} and student_embeddings.shape[0] > 1:
            student_similarity = normalized_student @ normalized_student.T
            teacher_similarity = normalized_teacher @ normalized_teacher.T
            batch_size = student_embeddings.shape[0]
            off_diagonal = ~torch.eye(batch_size, dtype=torch.bool, device=student_embeddings.device)
            student_logits = (student_similarity / distill_student_temperature)[off_diagonal].view(batch_size, -1)
            teacher_logits = (teacher_similarity / distill_teacher_temperature)[off_diagonal].view(batch_size, -1)

            if distill_loss_type == "kl":
                teacher_probs = F.softmax(teacher_logits, dim=-1)
                distill_loss = F.kl_div(
                    F.log_softmax(student_logits, dim=-1),
                    teacher_probs,
                    reduction="batchmean",
                ) * (distill_student_temperature**2)
            else:
                student_log_probs = F.log_softmax(student_logits, dim=-1)
                teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
                student_probs = student_log_probs.exp()
                teacher_probs = teacher_log_probs.exp()
                distill_loss = 0.5 * (
                    F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
                    + F.kl_div(teacher_log_probs, student_probs, reduction="batchmean")
                ) * (distill_student_temperature**2)
        elif distill_loss_type == "smooth_l1":
            if normalized_student.shape[-1] == normalized_teacher.shape[-1]:
                distill_loss = F.smooth_l1_loss(normalized_student, normalized_teacher, beta=0.1)
            elif normalized_student.shape[0] > 1:
                student_similarity = normalized_student @ normalized_student.T
                teacher_similarity = normalized_teacher @ normalized_teacher.T
                off_diagonal = ~torch.eye(student_embeddings.shape[0], dtype=torch.bool, device=student_embeddings.device)
                distill_loss = F.smooth_l1_loss(student_similarity[off_diagonal], teacher_similarity[off_diagonal], beta=0.1)
        else:
            student_similarity = normalized_student @ normalized_student.T
            teacher_similarity = normalized_teacher @ normalized_teacher.T
            if student_embeddings.shape[0] > 1:
                off_diagonal = ~torch.eye(student_embeddings.shape[0], dtype=torch.bool, device=student_embeddings.device)
                distill_loss = F.mse_loss(student_similarity[off_diagonal], teacher_similarity[off_diagonal])

    baseline_anchor_loss = zero
    if baseline_anchor_weight > 0:
        baseline_anchor_loss = 1.0 - F.cosine_similarity(student_embeddings, reference_embeddings, dim=-1).mean()

    teacher_cosine_loss = zero
    if teacher_cosine_weight > 0:
        if normalized_student.shape[-1] == normalized_teacher.shape[-1]:
            teacher_cosine_loss = 1.0 - F.cosine_similarity(normalized_student, normalized_teacher, dim=-1).mean()
        elif normalized_student.shape[0] > 1:
            batch_size = normalized_student.shape[0]
            off_diagonal = ~torch.eye(batch_size, dtype=torch.bool, device=student_embeddings.device)
            student_rel = (normalized_student @ normalized_student.T)[off_diagonal].view(batch_size, -1)
            teacher_rel = (normalized_teacher @ normalized_teacher.T)[off_diagonal].view(batch_size, -1)
            teacher_cosine_loss = 1.0 - F.cosine_similarity(student_rel, teacher_rel, dim=-1).mean()

    hard_negative_loss = zero
    if (
        hard_negative_weight > 0
        and positive_embeddings is not None
        and negative_embeddings is not None
        and negative_mask is not None
        and bool(negative_mask.any().item())
    ):
        positive_scores = (student_embeddings * positive_embeddings).sum(dim=-1, keepdim=True)
        negative_scores = torch.einsum("bd,bnd->bn", student_embeddings, negative_embeddings)
        margin_loss = F.relu(hard_negative_margin - positive_scores + negative_scores)
        valid_mask = negative_mask.float()
        if hard_negative_weighting == "softmax":
            masked_scores = negative_scores.masked_fill(~negative_mask, float("-inf"))
            normalized_weights = F.softmax(masked_scores / hard_negative_softmax_temperature, dim=1)
            normalized_weights = normalized_weights * valid_mask
            normalized_weights = normalized_weights / normalized_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            weighted_margin = margin_loss * normalized_weights
            valid_rows = (valid_mask.sum(dim=1) > 0).float()
            hard_negative_loss = weighted_margin.sum(dim=1).mul(valid_rows).sum() / valid_rows.sum().clamp_min(1.0)
        else:
            margin_loss = margin_loss * valid_mask
            hard_negative_loss = margin_loss.sum() / valid_mask.sum().clamp_min(1.0)

    relation_distill_loss = zero
    if relation_distill_weight > 0 and normalized_student.shape[0] > 1:
        batch_size = normalized_student.shape[0]
        off_diagonal = ~torch.eye(batch_size, dtype=torch.bool, device=student_embeddings.device)
        student_logits = ((normalized_student @ normalized_student.T) / relation_distill_temperature)[off_diagonal].view(
            batch_size, -1
        )
        teacher_probs = F.softmax(
            ((normalized_teacher @ normalized_teacher.T) / relation_distill_temperature)[off_diagonal].view(batch_size, -1),
            dim=-1,
        )
        relation_distill_loss = F.kl_div(F.log_softmax(student_logits, dim=-1), teacher_probs, reduction="batchmean") * (
            relation_distill_temperature**2
        )

    crd_loss = zero
    if crd_weight > 0 and normalized_student.shape[0] > 1:
        batch_size = normalized_student.shape[0]
        off_diagonal = ~torch.eye(batch_size, dtype=torch.bool, device=student_embeddings.device)
        student_logits = ((normalized_student @ normalized_student.T) / relation_distill_temperature)[off_diagonal].view(
            batch_size, -1
        )
        teacher_logits = ((normalized_teacher @ normalized_teacher.T) / relation_distill_temperature)[off_diagonal].view(
            batch_size, -1
        )
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        student_probs = student_log_probs.exp()
        teacher_probs = teacher_log_probs.exp()
        crd_loss = 0.5 * (
            F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
            + F.kl_div(teacher_log_probs, student_probs, reduction="batchmean")
        ) * (relation_distill_temperature**2)

    feature_distill_loss = zero
    if feature_distill_weight > 0:
        if normalized_student.shape[-1] == normalized_teacher.shape[-1]:
            feature_distill_loss = F.mse_loss(normalized_student, normalized_teacher)
        elif normalized_student.shape[0] > 1:
            student_similarity = normalized_student @ normalized_student.T
            teacher_similarity = normalized_teacher @ normalized_teacher.T
            off_diagonal = ~torch.eye(student_embeddings.shape[0], dtype=torch.bool, device=student_embeddings.device)
            feature_distill_loss = F.mse_loss(student_similarity[off_diagonal], teacher_similarity[off_diagonal])

    masked_feature_distill_loss = zero
    if masked_feature_distill_weight > 0:
        if normalized_student.shape[-1] == normalized_teacher.shape[-1]:
            keep_ratio = float(max(1e-3, min(1.0, masked_feature_keep_ratio)))
            mask = (torch.rand_like(normalized_student) < keep_ratio).float()
            valid_count = mask.sum(dim=-1).clamp_min(1.0)
            masked_delta = (normalized_student - normalized_teacher) * mask
            masked_feature_distill_loss = ((masked_delta.pow(2).sum(dim=-1) / valid_count).mean())
        elif normalized_student.shape[0] > 1:
            student_similarity = normalized_student @ normalized_student.T
            teacher_similarity = normalized_teacher @ normalized_teacher.T
            off_diagonal = ~torch.eye(student_embeddings.shape[0], dtype=torch.bool, device=student_embeddings.device)
            masked_feature_distill_loss = F.smooth_l1_loss(
                student_similarity[off_diagonal], teacher_similarity[off_diagonal], beta=0.1
            )

    gradient_distill_loss = zero
    if gradient_distill_weight > 0 and normalized_student.shape[0] > 1:
        if normalized_student.shape[-1] == normalized_teacher.shape[-1]:
            student_diff = normalized_student[1:] - normalized_student[:-1]
            teacher_diff = normalized_teacher[1:] - normalized_teacher[:-1]
            student_grad_like = F.normalize(student_diff, dim=-1)
            teacher_grad_like = F.normalize(teacher_diff, dim=-1)
            gradient_distill_loss = F.mse_loss(student_grad_like, teacher_grad_like)
        else:
            # Fallback for mismatched embedding dims: match gradients in relation space.
            student_relation = normalized_student @ normalized_student.T
            teacher_relation = normalized_teacher @ normalized_teacher.T
            student_relation_diff = student_relation[1:] - student_relation[:-1]
            teacher_relation_diff = teacher_relation[1:] - teacher_relation[:-1]
            gradient_distill_loss = F.mse_loss(student_relation_diff, teacher_relation_diff)

    augmented_feature_distill_loss = zero
    if augmented_feature_distill_weight > 0:
        if normalized_student.shape[-1] == normalized_teacher.shape[-1]:
            noise_std = max(0.0, float(augmented_feature_noise_std))
            if noise_std > 0:
                student_aug = F.normalize(normalized_student + torch.randn_like(normalized_student) * noise_std, dim=-1)
                teacher_aug = F.normalize(normalized_teacher + torch.randn_like(normalized_teacher) * noise_std, dim=-1)
            else:
                student_aug = normalized_student
                teacher_aug = normalized_teacher
            augmented_feature_distill_loss = F.mse_loss(student_aug, teacher_aug)
        elif normalized_student.shape[0] > 1:
            student_similarity = normalized_student @ normalized_student.T
            teacher_similarity = normalized_teacher @ normalized_teacher.T
            off_diagonal = ~torch.eye(student_embeddings.shape[0], dtype=torch.bool, device=student_embeddings.device)
            augmented_feature_distill_loss = F.mse_loss(student_similarity[off_diagonal], teacher_similarity[off_diagonal])

    icl_loss = zero
    if (
        icl_weight > 0
        and positive_embeddings is not None
        and teacher_positive_embeddings is not None
        and normalized_student.shape[0] > 1
    ):
        normalized_teacher_text = F.normalize(teacher_positive_embeddings, dim=-1)
        if normalized_teacher_text.shape[-1] != normalized_teacher.shape[-1]:
            raise ValueError("Teacher text embedding dim must match teacher image embedding dim for ICL")

        student_i2t_logits = (normalized_student @ positive_embeddings.T) / max(temperature, 1e-6)
        teacher_i2t_logits = (normalized_teacher @ normalized_teacher_text.T) / max(icl_teacher_temperature, 1e-6)
        student_t2i_logits = (positive_embeddings @ normalized_student.T) / max(temperature, 1e-6)
        teacher_t2i_logits = (normalized_teacher_text @ normalized_teacher.T) / max(icl_teacher_temperature, 1e-6)

        icl_i2t = F.kl_div(
            F.log_softmax(student_i2t_logits, dim=-1),
            F.softmax(teacher_i2t_logits, dim=-1),
            reduction="batchmean",
        )
        icl_t2i = F.kl_div(
            F.log_softmax(student_t2i_logits, dim=-1),
            F.softmax(teacher_t2i_logits, dim=-1),
            reduction="batchmean",
        )
        icl_loss = 0.5 * (icl_i2t + icl_t2i)

    memory_bank_distill_loss = zero
    if (
        memory_bank_distill_weight > 0
        and memory_bank_student is not None
        and memory_bank_teacher is not None
        and memory_bank_student.shape[0] >= memory_bank_min_samples
        and memory_bank_teacher.shape[0] >= memory_bank_min_samples
    ):
        bank_student = F.normalize(memory_bank_student, dim=-1)
        bank_teacher = F.normalize(memory_bank_teacher, dim=-1)
        student_logits = (normalized_student @ bank_student.T) / memory_bank_distill_temperature
        teacher_probs = F.softmax((normalized_teacher @ bank_teacher.T) / memory_bank_distill_temperature, dim=-1)
        memory_bank_distill_loss = F.kl_div(F.log_softmax(student_logits, dim=-1), teacher_probs, reduction="batchmean") * (
            memory_bank_distill_temperature**2
        )

    backbone_feature_distill_loss = zero
    if backbone_feature_distill_weight > 0:
        if student_backbone_features is None or reference_backbone_features is None:
            raise ValueError("Backbone feature distillation requires both student and reference backbone features")
        student_features = F.normalize(student_backbone_features, dim=-1)
        reference_features = F.normalize(reference_backbone_features, dim=-1)
        backbone_feature_distill_loss = 1.0 - F.cosine_similarity(student_features, reference_features, dim=-1).mean()

    total_loss = (
        contrastive_weight * contrastive_loss
        + distill_weight * distill_loss
        + teacher_cosine_weight * teacher_cosine_loss
        + baseline_anchor_weight * baseline_anchor_loss
        + hard_negative_weight * hard_negative_loss
        + relation_distill_weight * relation_distill_loss
        + crd_weight * crd_loss
        + icl_weight * icl_loss
        + memory_bank_distill_weight * memory_bank_distill_loss
        + backbone_feature_distill_weight * backbone_feature_distill_loss
        + feature_distill_weight * feature_distill_loss
        + masked_feature_distill_weight * masked_feature_distill_loss
        + gradient_distill_weight * gradient_distill_loss
        + augmented_feature_distill_weight * augmented_feature_distill_loss
    )

    return total_loss, {
        "contrastive_loss": float(contrastive_loss.detach().cpu()),
        "distill_loss": float(distill_loss.detach().cpu()),
        "teacher_cosine_loss": float(teacher_cosine_loss.detach().cpu()),
        "baseline_anchor_loss": float(baseline_anchor_loss.detach().cpu()),
        "hard_negative_loss": float(hard_negative_loss.detach().cpu()),
        "relation_distill_loss": float(relation_distill_loss.detach().cpu()),
        "crd_loss": float(crd_loss.detach().cpu()),
        "icl_loss": float(icl_loss.detach().cpu()),
        "memory_bank_distill_loss": float(memory_bank_distill_loss.detach().cpu()),
        "backbone_feature_distill_loss": float(backbone_feature_distill_loss.detach().cpu()),
        "feature_distill_loss": float(feature_distill_loss.detach().cpu()),
        "masked_feature_distill_loss": float(masked_feature_distill_loss.detach().cpu()),
        "gradient_distill_loss": float(gradient_distill_loss.detach().cpu()),
        "augmented_feature_distill_loss": float(augmented_feature_distill_loss.detach().cpu()),
        "total_loss": float(total_loss.detach().cpu()),
    }
