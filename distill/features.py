from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class BlockFeatureRecorder:
    def __init__(self, modules: list[nn.Module]):
        self.outputs: list[torch.Tensor] = []
        self.handles = [module.register_forward_hook(self._hook) for module in modules]

    def _hook(self, _module, _inputs, output) -> None:
        tensor = output[0] if isinstance(output, (tuple, list)) else output
        if isinstance(tensor, torch.Tensor):
            self.outputs.append(tensor)

    def clear(self) -> None:
        self.outputs.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def get_vision_blocks(clip_model: nn.Module) -> list[nn.Module]:
    visual = clip_model.visual
    if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
        return list(visual.transformer.resblocks)
    if hasattr(visual, "trunk") and hasattr(visual.trunk, "blocks"):
        return list(visual.trunk.blocks)
    return []


def to_batch_feature(block_output: torch.Tensor) -> torch.Tensor:
    feature = block_output
    if feature.dim() == 4:
        feature = feature.flatten(start_dim=2).mean(dim=-1)
    elif feature.dim() == 3:
        if feature.shape[0] > feature.shape[1]:
            feature = feature.permute(1, 0, 2)
        feature = feature.mean(dim=1)
    elif feature.dim() == 2:
        pass
    else:
        raise ValueError(f"Unsupported block feature shape: {tuple(feature.shape)}")
    return feature.float()


def compute_intermediate_relation_loss(
    *,
    student_block_outputs: list[torch.Tensor],
    teacher_block_outputs: list[torch.Tensor],
) -> torch.Tensor:
    if not student_block_outputs or not teacher_block_outputs:
        raise ValueError("Intermediate distillation requires both student and teacher block outputs")

    paired_count = min(len(student_block_outputs), len(teacher_block_outputs))
    if paired_count <= 0:
        raise ValueError("No paired block outputs available for intermediate distillation")

    loss = torch.zeros((), device=student_block_outputs[0].device)
    for student_output, teacher_output in zip(
        student_block_outputs[-paired_count:],
        teacher_block_outputs[-paired_count:],
    ):
        student_feature = F.normalize(to_batch_feature(student_output), dim=-1)
        teacher_feature = F.normalize(to_batch_feature(teacher_output), dim=-1)
        student_relation = student_feature @ student_feature.T
        teacher_relation = teacher_feature @ teacher_feature.T

        batch_size = student_relation.shape[0]
        if batch_size <= 1:
            continue
        off_diagonal_mask = ~torch.eye(batch_size, dtype=torch.bool, device=student_relation.device)
        loss = loss + F.mse_loss(
            student_relation[off_diagonal_mask],
            teacher_relation[off_diagonal_mask],
        )
    return loss / float(paired_count)


def resolve_resume_checkpoint_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate

    if candidate.parent == Path("."):
        matches = sorted(Path(".").glob(f"**/{candidate.name}"))
        if len(matches) == 1:
            print(f"Resolved resume checkpoint '{raw_path}' -> '{matches[0]}'")
            return matches[0]
        if len(matches) > 1:
            display = "\n".join(str(path) for path in matches[:10])
            raise FileNotFoundError(
                f"Resume checkpoint '{raw_path}' is ambiguous. Use one of:\n{display}"
            )

    raise FileNotFoundError(f"Resume checkpoint does not exist: {candidate}")


def update_memory_bank(
    bank: torch.Tensor,
    new_embeddings: torch.Tensor,
    *,
    max_size: int,
) -> torch.Tensor:
    if max_size <= 0 or new_embeddings.numel() == 0:
        return bank
    new_bank = torch.cat((bank, new_embeddings.detach()), dim=0)
    if new_bank.shape[0] > max_size:
        new_bank = new_bank[-max_size:]
    return new_bank
