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


class StudentImageModel(nn.Module):
    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.clip_model = clip_model
        self.embedding_dim = get_clip_embedding_dim(clip_model)
        self.projection_head = ProjectionHead(self.embedding_dim)

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


def configure_trainable_student(
    student_model: StudentImageModel,
    *,
    unfreeze_last_n_blocks: int = 1,
) -> list[str]:
    freeze_module(student_model.clip_model)
    unfreeze_module(student_model.projection_head)

    trainable_modules = ["projection_head"]
    blocks = _get_vision_blocks(student_model.clip_model)

    if unfreeze_last_n_blocks > 0 and blocks:
        for block in blocks[-unfreeze_last_n_blocks:]:
            unfreeze_module(block)
        trainable_modules.append(f"vision_blocks[-{unfreeze_last_n_blocks}:]")

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

    return trainable_modules


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def get_trainable_parameters(module: nn.Module) -> Iterable[torch.nn.Parameter]:
    return (parameter for parameter in module.parameters() if parameter.requires_grad)