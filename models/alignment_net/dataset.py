"""Dataset for the alignment net - MIP Section 4.1 / 3.3.

Labels (Section 3.3): pitch/roll/yaw offset. If IO-VNBD does not give
this directly, derive a pseudo-label from the gravity vector during
stationary/near-constant-velocity segments plus GNSS course-over-ground
as the yaw reference - that derivation is Layer 1's job in the Section
3 pipeline, not stubbed here, since it depends on exactly what ends up
in data/processed/.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from models.common.normalization import apply, load_stats


class AlignmentNetDataset(Dataset):
    def __init__(self, processed_dir: str | Path, split: str, norm_stats_path: str | Path):
        """
        Args:
            processed_dir: data/processed/alignment_net/ (or wherever
                Section 3's pipeline writes windows for this model).
            split: "train" | "val" | "test".
            norm_stats_path: this model's norm_stats.json (Section 3.2
                step 4).
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
                "data/scripts/03_window.py --model alignment_net first "
                "(see data/processed/README.md)."
            )

        # (N, 200, 9) raw, matching 03_window.py's on-disk notation.
        # Normalized and transposed to (9, 200) per-item below, not here,
        # per this file's own docstring.
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
            window: (9, 200) float32, channels-first, normalized with
                self.norm_stats.
            label: (4,) float32 - [pitch, roll, sin(yaw), cos(yaw)].
        """
        window = apply(self.windows[idx : idx + 1], self.norm_stats)[0]  # (200, 9), normalized
        window = torch.from_numpy(window).float().transpose(0, 1)  # -> (9, 200)
        label = torch.from_numpy(self.labels[idx]).float()
        return window, label
