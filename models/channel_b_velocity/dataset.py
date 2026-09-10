"""Dataset for Channel B - MIP Section 4.3 / 3.3.

Label (Section 3.3): forward velocity scalar per window, from the V
(vehicle ground truth) stream - same label source as Channel A, but
windowed at 4s instead of 2s (Section 3.2 step 3).
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from models.common.normalization import load_stats


class ChannelBVelocityDataset(Dataset):
    def __init__(self, processed_dir: str | Path, split: str, norm_stats_path: str | Path):
        """
        Args:
            processed_dir: data/processed/channel_b_velocity/.
            split: "train" | "val" | "test".
            norm_stats_path: this model's norm_stats.json.
        """
        self.processed_dir = Path(processed_dir)
        self.split = split
        self.norm_stats = load_stats(norm_stats_path)
        # TODO(Layer 1): load real windows/labels for `split` once
        # Section 3's output format exists.
        raise NotImplementedError(
            "Wire this up to the real Section 3 output format once "
            "data/processed/ exists - see data/processed/README.md."
        )

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (window, label):
            window: (3, 400) float32, channels-first, normalized.
            label: (1,) float32 - forward velocity, m/s.
        """
        raise NotImplementedError
