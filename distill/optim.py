from __future__ import annotations

import math

import torch


def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_scale: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    bounded_total_steps = max(1, int(total_steps))
    bounded_warmup_steps = max(0, min(int(warmup_steps), bounded_total_steps - 1))
    clamped_min_lr_scale = min(1.0, max(0.0, float(min_lr_scale)))

    def lr_lambda(step: int) -> float:
        if bounded_warmup_steps > 0 and step < bounded_warmup_steps:
            return max(1e-8, float(step + 1) / float(bounded_warmup_steps))
        if bounded_total_steps <= bounded_warmup_steps:
            return clamped_min_lr_scale

        progress = (step - bounded_warmup_steps) / float(bounded_total_steps - bounded_warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return clamped_min_lr_scale + (1.0 - clamped_min_lr_scale) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def interpolate_weight(*, start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return float(end)
    progress = min(1.0, max(0.0, step / float(total_steps - 1)))
    return float(start + (end - start) * progress)
