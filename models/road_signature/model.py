"""Road-Signature Segment Classifier - MIP Section 4.4.

Classifies a 2s IMU window into one of N pre-mapped road segments along
a specific corridor, from vibration signature alone. Trained per-
corridor/per-city (see module README) - n_segments is a constructor
argument, not a hardcoded constant.
"""

import torch
import torch.nn as nn


class RoadSignatureNet(nn.Module):
    """Input: (batch, 6, 200) channels-first, accel+gyro. Output:
    (batch, n_segments) logits - apply softmax at inference, not in
    forward(), so training can use nn.CrossEntropyLoss directly."""

    def __init__(self, n_segments: int):
        super().__init__()
        if n_segments is None or n_segments < 1:
            raise ValueError(
                "n_segments must be set per corridor/region before "
                "constructing this model - see config.yaml and README.md."
            )
        channels = [6, 32, 64, 128]
        layers = []
        for i in range(3):
            layers += [
                nn.Conv1d(channels[i], channels[i + 1], kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
        self.conv_stack = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, n_segments)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_stack(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)
