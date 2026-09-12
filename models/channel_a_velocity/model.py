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

    FIX #1 (was GlobalAvgPool1d): the label from
    data/scripts/03_window.py::build_channel_a is the ground-truth
    velocity at the window's end, not an average over the window.
    Reading out position -1 directly (the only position whose causal
    receptive field ends exactly at the window's last sample) matches
    the label's actual timing - GlobalAvgPool blended that with far
    less-informed earlier positions.

    FIX #2 (dilations 1/2/4/8 -> 8/16/32/64, same doubling pattern,
    8x the base): FIX #1 alone made results WORSE in practice (see the
    metrics.csv that prompted this) - with dilations 1/2/4/8 the causal
    receptive field feeding position -1 is only 2*(1+2+4+8)+1 = 31
    samples (~0.31s @ 100Hz), so the last-timestep readout was only
    ever seeing the most recent 0.3s of the 2s window, discarding the
    rest. Widening dilation costs nothing MIP Section 4's "small on
    purpose... do not scale up without re-checking the latency budget"
    warning cares about: same kernel_size (3 taps per layer), same
    channel widths, same param count, same multiply-add count per
    output position - dilation only changes the spacing between the
    existing 3 taps, not how many there are. New RF = 2*(8+16+32+64)+1
    = 241 samples, comfortably covering the full 200-sample window, so
    position -1 now genuinely conditions on the whole window instead of
    a sliver of it.

    This deviates from MIP Section 4.2's literal "dilations 1/2/4/8" -
    document that back to the MIP rather than treating this file as the
    silent source of truth.
    """

    def __init__(
        self,
        in_channels: int = 6,
        channels: list[int] = (32, 64, 64, 128),
        dilations: list[int] = (8, 16, 32, 64),
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        all_channels = [in_channels, *channels]
        self.blocks = nn.ModuleList(
            [
                TCNBlock(all_channels[i], all_channels[i + 1], kernel_size=kernel_size, dilation=dilations[i], dropout=dropout)
                for i in range(len(dilations))
            ]
        )
        self.fc1 = nn.Linear(all_channels[-1], 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = x[:, :, -1]  # causal feature at the window's last timestep, matches the label's timing
        x = self.relu(self.fc1(x))
        return self.fc2(x)
