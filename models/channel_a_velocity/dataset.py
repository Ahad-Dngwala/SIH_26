"""Dataset for Channel A - MIP Section 4.2 / 3.3.

Label (Section 3.3): forward velocity scalar per window, taken from the
V (vehicle ground truth) stream. Augmentation per Section 3.4 includes
the Unsynchronised split as a noisy-label source specifically for
Channel A/B robustness - handled upstream in the Section 3 pipeline,
not here.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from models.common.normalization import load_stats


class ChannelAVelocityDataset(Dataset):
    def __init__(self, processed_dir: str | Path, split: str, norm_stats_path: str | Path):
        """
        Args:
            processed_dir: data/processed/channel_a_velocity/ (or
                wherever Section 3's pipeline writes windows for this
                model).
            split: "train" | "val" | "test".
            norm_stats_path: this model's norm_stats.json.
        """
        self.processed_dir = Path(processed_dir)
        self.split = split
        self.norm_stats = load_stats(norm_stats_path)
        # TODO(Layer 1): load real windows/labels for `split` once
        # Section 3's output format exists. Keep the
        # (window_len, channels) -> (channels, window_len) transpose
        # here in __getitem__.
        raise NotImplementedError(
            "Wire this up to the real Section 3 output format once "
            "data/processed/ exists - see data/processed/README.md."
        )

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (window, label):
            window: (6, 200) float32, channels-first, normalized.
            label: (1,) float32 - forward velocity, m/s.
        """
        raise NotImplementedError
