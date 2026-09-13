"""Dataset for Channel A - MIP Section 4.2 / 3.3.

Label (Section 3.3): forward velocity scalar per window, taken from the
V (vehicle ground truth) stream. Augmentation per Section 3.4 includes
the Unsynchronised split as a noisy-label source specifically for
Channel A/B robustness - handled upstream in the Section 3 pipeline,
not here.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from models.common.normalization import apply, load_stats


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

        split_dir = self.processed_dir / split
        windows_path = split_dir / "windows.npy"
        labels_path = split_dir / "labels.npy"
        if not windows_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                f"{windows_path} / {labels_path} not found - run "
                "data/scripts/03_window.py --model channel_a_velocity "
                "first (see data/processed/README.md)."
            )

        # (N, 200, 6) raw; normalized and transposed to (6, 200) per-item
        # below, not here, per this file's own docstring.
        self.windows = np.load(windows_path)
        self.labels = np.load(labels_path)
        if len(self.windows) != len(self.labels):
            raise ValueError(
                f"windows/labels length mismatch in {split_dir}: "
                f"{len(self.windows)} vs {len(self.labels)}"
            )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (window, label):
            window: (6, 200) float32, channels-first, normalized.
            label: (1,) float32 - forward velocity, m/s.
        """
        window = apply(self.windows[idx : idx + 1], self.norm_stats)[0]  # (200, 6), normalized
        window = torch.from_numpy(window).float().transpose(0, 1)  # -> (6, 200)
        label = torch.from_numpy(self.labels[idx]).float()
        return window, label
