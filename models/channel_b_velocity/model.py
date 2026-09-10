"""Channel B - accelerometer-only velocity net (CarSpeedNet-style) -
MIP Section 4.3.

Independent, gyro-free forward speed estimate used as a cross-check
against Channel A. Deliberately less accurate by design - see the
module README before "improving" this past spec.
"""

import torch
import torch.nn as nn


class ChannelBVelocityNet(nn.Module):
    """Input: (batch, 3, 400) channels-first, accel xyz only, 4s window.
    Output: (batch, 1) forward velocity, m/s."""

    def __init__(self):
        super().__init__()
        channels = [3, 32, 64, 128, 128]
        layers = []
        for i in range(4):
            layers += [
                nn.Conv1d(channels[i], channels[i + 1], kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(channels[i + 1]),
                nn.ReLU(),
            ]
        self.conv_stack = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_stack(x)
        x = self.pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
