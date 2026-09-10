"""Dataset for the calibration adapter - MIP Section 4.5 / 3.3.

Unlike the other four models, this is NOT a train/val/test split over
the whole corpus - it is a single short (10-30s) calibration-drive
session for one specific vehicle. Label (Section 3.3): same windowing
as Channel A/B, but labels drawn only from this session's own
GNSS-derived velocity, not the wider IO-VNBD ground truth.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset


class CalibrationDriveDataset(Dataset):
    def __init__(self, session_path: str | Path, norm_stats_path: str | Path, window_seconds: float = 2.0):
        """
        Args:
            session_path: path to one raw calibration-drive recording
                (raw IMU + GNSS for 10-30s), NOT data/processed/ - this
                is windowed on the fly from a single short session, per
                Section 4.5's "on-device" framing, rather than
                pre-windowed like the other four models.
            norm_stats_path: the *base* Channel A/B model's
                norm_stats.json - the calibration session must be
                normalized the same way the base model was trained, not
                with its own separately-computed stats.
            window_seconds: 2.0 for Channel A's window convention, pass
                4.0 if calibrating Channel B instead.
        """
        self.session_path = Path(session_path)
        self.window_seconds = window_seconds
        # TODO(Layer 1): load the short session, window it (Section 3.2
        # step 3 conventions), normalize with the base model's own
        # norm_stats (not session-local stats), pair with GNSS-derived
        # velocity as the label.
        raise NotImplementedError("Wire this up once a real calibration-drive recording format exists.")

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (window, label) - same shapes as the base model's own
        dataset (Channel A: (6,200)/(1,), Channel B: (3,400)/(1,))."""
        raise NotImplementedError
