"""Dataset for the alignment net - MIP Section 4.1 / 3.3.

Labels (Section 3.3): pitch/roll/yaw offset. If IO-VNBD does not give
this directly, derive a pseudo-label from the gravity vector during
stationary/near-constant-velocity segments plus GNSS course-over-ground
as the yaw reference - that derivation is Layer 1's job in the Section
3 pipeline, not stubbed here, since it depends on exactly what ends up
in data/processed/.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from models.common.normalization import load_stats


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
        # TODO(Layer 1): load the actual windows/labels for `split` from
        # `processed_dir` once Section 3's output file format is decided.
        # Do the (window_len, channels) -> (channels, window_len)
        # transpose here in __getitem__, not upstream in processed/, so
        # the on-disk format matches Section 3's own (200, 9) notation.
        raise NotImplementedError(
            "Wire this up to the real Section 3 output format once "
            "data/processed/ exists - see data/processed/README.md."
        )

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (window, label):
            window: (9, 200) float32, channels-first, normalized with
                self.norm_stats.
            label: (4,) float32 - [pitch, roll, sin(yaw), cos(yaw)].
        """
        raise NotImplementedError
