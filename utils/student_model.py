from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


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


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = True


def _get_vision_blocks(clip_model: nn.Module) -> list[nn.Module]:
    visual = clip_model.visual
    if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
        return list(visual.transformer.resblocks)
    if hasattr(visual, "trunk") and hasattr(visual.trunk, "blocks"):
        return list(visual.trunk.blocks)
    return []


def _get_text_blocks(clip_model: nn.Module) -> list[nn.Module]:
    if hasattr(clip_model, "transformer") and hasattr(clip_model.transformer, "resblocks"):
        return list(clip_model.transformer.resblocks)
    text_module = getattr(clip_model, "text", None)
    if text_module is not None and hasattr(text_module, "transformer") and hasattr(text_module.transformer, "resblocks"):
        return list(text_module.transformer.resblocks)
    return []


def configure_trainable_student(
    student_model: StudentImageModel,
    *,
    unfreeze_last_n_blocks: int = 1,
    unfreeze_text_last_n_blocks: int = 0,
    unfreeze_text_tower: bool = False,
    train_logit_scale: bool = False,
) -> list[str]:
    freeze_module(student_model.clip_model)
    unfreeze_module(student_model.projection_head)

    trainable_modules = ["projection_head"]
    blocks = _get_vision_blocks(student_model.clip_model)

    if unfreeze_last_n_blocks > 0 and blocks:
        for block in blocks[-unfreeze_last_n_blocks:]:
            unfreeze_module(block)
        trainable_modules.append(f"vision_blocks[-{unfreeze_last_n_blocks}:]")

    text_blocks = _get_text_blocks(student_model.clip_model)
    if unfreeze_text_tower:
        if hasattr(student_model.clip_model, "token_embedding") and isinstance(student_model.clip_model.token_embedding, nn.Module):
            unfreeze_module(student_model.clip_model.token_embedding)
            trainable_modules.append("text.token_embedding")
        if hasattr(student_model.clip_model, "transformer") and isinstance(student_model.clip_model.transformer, nn.Module):
            unfreeze_module(student_model.clip_model.transformer)
            trainable_modules.append("text.transformer")
        if hasattr(student_model.clip_model, "ln_final") and isinstance(student_model.clip_model.ln_final, nn.Module):
            unfreeze_module(student_model.clip_model.ln_final)
            trainable_modules.append("text.ln_final")
        if hasattr(student_model.clip_model, "text_projection"):
            text_projection = getattr(student_model.clip_model, "text_projection")
            if isinstance(text_projection, torch.nn.Parameter):
                text_projection.requires_grad = True
                trainable_modules.append("text.text_projection")
    elif unfreeze_text_last_n_blocks > 0 and text_blocks:
        for block in text_blocks[-unfreeze_text_last_n_blocks:]:
            unfreeze_module(block)
        trainable_modules.append(f"text_blocks[-{unfreeze_text_last_n_blocks}:]")
        if hasattr(student_model.clip_model, "ln_final") and isinstance(student_model.clip_model.ln_final, nn.Module):
            unfreeze_module(student_model.clip_model.ln_final)
            trainable_modules.append("text.ln_final")
        if hasattr(student_model.clip_model, "text_projection"):
            text_projection = getattr(student_model.clip_model, "text_projection")
            if isinstance(text_projection, torch.nn.Parameter):
                text_projection.requires_grad = True
                trainable_modules.append("text.text_projection")

    if train_logit_scale and hasattr(student_model.clip_model, "logit_scale"):
        logit_scale = getattr(student_model.clip_model, "logit_scale")
        if isinstance(logit_scale, torch.nn.Parameter):
            logit_scale.requires_grad = True
            trainable_modules.append("clip.logit_scale")

    visual = student_model.clip_model.visual
    for attribute_name in ("ln_post", "proj", "fc_norm", "head"):
        if hasattr(visual, attribute_name):
            attribute = getattr(visual, attribute_name)
            if isinstance(attribute, nn.Module):
                unfreeze_module(attribute)
                trainable_modules.append(f"visual.{attribute_name}")
            elif isinstance(attribute, torch.nn.Parameter):
                attribute.requires_grad = True
                trainable_modules.append(f"visual.{attribute_name}")

    if student_model.teacher_projector is not None:
        unfreeze_module(student_model.teacher_projector)
        trainable_modules.append("teacher_projector")

    return trainable_modules


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def get_trainable_parameters(module: nn.Module) -> Iterable[torch.nn.Parameter]:
    return (parameter for parameter in module.parameters() if parameter.requires_grad)
