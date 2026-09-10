"""Dataset for the road-signature classifier - MIP Section 4.4 / 3.3.

Label (Section 3.3): segment ID, derived by dividing each corridor's
route into fixed-length arc segments (e.g. 50-100m each) using GNSS
ground truth positions, then labeling every window with the segment its
GNSS timestamp falls into. That segmentation is Layer 1's job in the
Section 3 pipeline, not stubbed here.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from models.common.normalization import load_stats


class RoadSignatureDataset(Dataset):
    def __init__(self, processed_dir: str | Path, split: str, norm_stats_path: str | Path, corridor: str):
        """
        Args:
            processed_dir: data/processed/road_signature/<corridor>/ -
                this model is per-corridor, so processed data is keyed
                by corridor, not global like the other four models.
            split: "train" | "val" | "test".
            norm_stats_path: this model's norm_stats.json (likely also
                per-corridor).
            corridor: which corridor/region's signature pack this is
                (e.g. "ahmedabad_sg_highway").
        """
        self.processed_dir = Path(processed_dir)
        self.split = split
        self.corridor = corridor
        self.norm_stats = load_stats(norm_stats_path)
        # TODO(Layer 1): load real windows/segment-ID labels for
        # `split` once Section 3's per-corridor segmentation exists.
        raise NotImplementedError(
            "Wire this up to the real Section 3 output format once "
            "data/processed/ exists - see data/processed/README.md."
        )

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (window, label):
            window: (6, 200) float32, channels-first, normalized.
            label: scalar long tensor - segment ID (class index).
        """
        raise NotImplementedError
