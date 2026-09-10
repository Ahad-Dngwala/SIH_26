"""Per-Vehicle Calibration Adapter - MIP Section 4.5.

LoRA-style low-rank adapter for a frozen nn.Linear layer. Meant to wrap
Channel A's and Channel B's final FC layer (`fc2` in both models'
current model.py) - see attach_calibration_adapters() below.

Depends on models/channel_a_velocity/model.py and
models/channel_b_velocity/model.py already existing with trained
weights - only makes sense to run once Section 4.1-4.4 have first
working versions (Section 12).
"""

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank adapter:

        y = frozen_linear(x) + scale * (x @ A^T @ B^T)

    A: (rank, in_features), B: (out_features, rank). B is zero-
    initialized so the adapter is a no-op until trained (standard LoRA
    init). Section 4.5: rank 4-8, base frozen, only the adapter trains.
    """

    def __init__(self, base_linear: nn.Linear, rank: int = 6, scale: float = 1.0):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)

    def trainable_parameter_count(self) -> int:
        return self.A.numel() + self.B.numel()


def attach_calibration_adapters(channel_a_model, channel_b_model, rank: int = 6):
    """Freeze both base models, replace each one's final FC layer
    (`fc2`) with a LoRA-wrapped version, and return the two adapter
    modules - these are what actually gets trained and saved per
    Section 4.5.

    Raises AssertionError if the combined adapter parameter count
    exceeds the Section 4.5 budget of 50k.
    """
    for p in channel_a_model.parameters():
        p.requires_grad = False
    for p in channel_b_model.parameters():
        p.requires_grad = False

    adapter_a = LoRALinear(channel_a_model.fc2, rank=rank)
    adapter_b = LoRALinear(channel_b_model.fc2, rank=rank)
    channel_a_model.fc2 = adapter_a
    channel_b_model.fc2 = adapter_b

    total_trainable = adapter_a.trainable_parameter_count() + adapter_b.trainable_parameter_count()
    assert total_trainable < 50_000, (
        f"combined adapter params ({total_trainable}) exceed the Section 4.5 budget of 50k "
        f"- reduce rank"
    )
    return adapter_a, adapter_b
