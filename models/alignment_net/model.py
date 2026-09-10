"""Alignment / Mount Calibration Net - MIP Section 4.1.

Estimates the phone's pitch/roll/yaw offset relative to the vehicle's
direction of travel, from a 2s / 9-channel IMU+mag window.

NOTE - spec discrepancy in Section 4.1, flagged rather than silently
resolved: the architecture line says "FC(32 to 3)" but the loss line
says "predict 4 values total: pitch, roll, sin(yaw), cos(yaw)" for the
wraparound-safe angle trick to work. This implementation follows the
loss line (4 outputs), since that is the only way sin/cos(yaw) is
representable, and reconstructs yaw via atan2 in `to_angles()`. Get the
MIP itself corrected - don't let this comment be the only record.
"""

import torch
import torch.nn as nn


class AlignmentNet(nn.Module):
    """Input: (batch, 9, 200) channels-first. Output: (batch, 4) raw
    [pitch, roll, sin(yaw), cos(yaw)]."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(9, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)  # robust to exact length after strided convs
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 4)  # see module docstring re: 3 vs 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

    @staticmethod
    def to_angles(raw_output: torch.Tensor) -> torch.Tensor:
        """(batch, 4) raw output -> (batch, 3) [pitch, roll, yaw] radians."""
        pitch, roll, sin_yaw, cos_yaw = raw_output.unbind(dim=-1)
        yaw = torch.atan2(sin_yaw, cos_yaw)
        return torch.stack([pitch, roll, yaw], dim=-1)
