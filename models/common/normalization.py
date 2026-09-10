"""Per-channel z-score stats, shared across every model.

Section 3.2 step 4: normalization stats are computed on the TRAIN
split only, saved once per model as norm_stats.json, then applied
as-is to val/test - and later baked into the on-device preprocessing
code, so the file format here has to stay simple enough to port to
Kotlin/C++ without a JSON library doing anything clever.
"""

import json
from pathlib import Path

import numpy as np


def compute_stats(train_windows: np.ndarray) -> dict:
    """Compute per-channel mean/std from TRAIN-split windows only.

    Args:
        train_windows: array shaped (n_windows, window_len, n_channels).

    Returns:
        {"mean": [...], "std": [...]} with one entry per channel.
    """
    mean = train_windows.mean(axis=(0, 1))
    std = train_windows.std(axis=(0, 1))
    # TODO(Layer 1): decide the floor value for near-zero std channels
    # before this ships - dividing by ~0 on a stationary channel will
    # blow up normalized values. 1e-6 is a placeholder, not a tuned choice.
    std = np.where(std < 1e-6, 1e-6, std)
    return {"mean": mean.tolist(), "std": std.tolist()}


def save_stats(stats: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(stats, indent=2))


def load_stats(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def apply(windows: np.ndarray, stats: dict) -> np.ndarray:
    """Apply saved mean/std to any split (val/test/inference)."""
    mean = np.asarray(stats["mean"])
    std = np.asarray(stats["std"])
    return (windows - mean) / std
