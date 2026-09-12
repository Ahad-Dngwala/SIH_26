"""Channel A - NHC-constrained bias-correction network - MIP Section 4.2.

Corrects accelerometer/gyro bias and scale error via a TCN, producing a
forward-velocity estimate for the window. NHC (no lateral slip, no
vertical velocity) is enforced downstream in the UKF process model
(Section 5), not in this network - keep this a pure regressor.
"""

import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    """Trim the extra right-side padding from symmetric Conv1d padding
    so the convolution stays causal (output at t depends only on
    inputs <= t) - the standard TCN "chomp" (Bai et al.)."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size] if self.chomp_size > 0 else x


class TCNBlock(nn.Module):
    """One dilated causal conv block with a residual connection, per
    Section 4.2 ("4 dilated causal conv blocks ... residual connections
    between blocks")."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class ChannelAVelocityNet(nn.Module):
    """Input: (batch, 6, 200) channels-first. Output: (batch, 1) forward
    velocity, m/s, for the window's LAST timestep.

    FIX (was GlobalAvgPool1d): the label from
    data/scripts/03_window.py::build_channel_a is the ground-truth
    velocity at the window's end, not an average over the window. Each
    of the 200 causal positions out of the TCN stack sees a different
    amount of context (position 0 has seen almost nothing, position
    199 has seen the block stack's full receptive field), so averaging
    all of them together blends "barely-informed early estimate" with
    "fully-informed end-of-window estimate" into one smeared feature
    vector that doesn't correspond to any single point in time - and
    definitely not to the end-of-window instant the label is asking
    for. Reading out position -1 directly (the only position whose
    causal receptive field ends exactly at the window's last sample)
    matches the label's actual timing.

    Note: with this block's dilation schedule (1/2/4/8, kernel 3), the
    causal receptive field is only 2*(1+2+4+8)+1 = 31 samples (~0.31s
    at 100Hz) - much smaller than the 200-sample window. Taking the
    last timestep means the network only ever conditions directly on
    the most recent ~0.3s; it does not integrate the full 2s window at
    any single position. That's probably fine for a point-in-time
    velocity read-out (recent IMU dynamics dominate), but if the intent
    is for the network to use the full window's context, the dilation
    schedule needs widening - that would deviate from Section 4.2's
    "4 dilated causal conv blocks" spec, so flag it back to whoever
    owns the MIP rather than changing it here.
    """

    def __init__(self):
        super().__init__()
        channels = [6, 32, 64, 64, 128]
        dilations = [1, 2, 4, 8]
        self.blocks = nn.ModuleList(
            [
                TCNBlock(channels[i], channels[i + 1], kernel_size=3, dilation=dilations[i], dropout=0.2)
                for i in range(4)
            ]
        )
        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = x[:, :, -1]  # causal feature at the window's last timestep, matches the label's timing
        x = self.relu(self.fc1(x))
        return self.fc2(x)
