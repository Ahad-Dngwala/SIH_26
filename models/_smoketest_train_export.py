"""Not part of the pipeline - a one-off smoke test that runs synthetic
windows through each Phase-1 model's real dataset.py -> train.py ->
export.py path (1 epoch, tiny data) to validate the mechanics without
needing real IO-VNBD data/processed/. Mirrors
data/scripts/_smoketest_synthetic.py's role for the data pipeline.
Delete once a real training run against data/processed/ is verified,
or keep as a regression test - your call.

Run from repo root: python models/_smoketest_train_export.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.common.normalization import compute_stats, save_stats  # noqa: E402

ROOT = Path("/tmp/models_smoketest")
N_TRAIN, N_VAL = 64, 16


def make_split(out_dir: Path, n: int, window_len: int, n_channels: int, label_dim: int, rng: np.random.Generator):
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = rng.normal(size=(n, window_len, n_channels)).astype(np.float32)
    labels = rng.normal(size=(n, label_dim)).astype(np.float32)
    if label_dim == 4:
        # alignment_net labels: [pitch, roll, sin(yaw), cos(yaw)] - sin/cos must
        # actually lie on the unit circle for to_angles() round-tripping to make sense.
        yaw = rng.uniform(-np.pi, np.pi, size=n).astype(np.float32)
        labels[:, 2] = np.sin(yaw)
        labels[:, 3] = np.cos(yaw)
    np.save(out_dir / "windows.npy", windows)
    np.save(out_dir / "labels.npy", labels)
    (out_dir / "meta.json").write_text('{"session_ids": []}')


def run_one(model_name: str, window_len: int, n_channels: int, label_dim: int, dataset_cls, module_cls, export_main):
    print(f"\n=== {model_name} ===")
    rng = np.random.default_rng(0)
    processed_dir = ROOT / model_name
    for split, n in [("train", N_TRAIN), ("val", N_VAL), ("test", N_VAL)]:
        make_split(processed_dir / split, n, window_len, n_channels, label_dim, rng)

    stats = compute_stats(np.load(processed_dir / "train" / "windows.npy"))
    stats_path = processed_dir / "norm_stats.json"
    save_stats(stats, stats_path)

    train_ds = dataset_cls(processed_dir, "train", stats_path)
    val_ds = dataset_cls(processed_dir, "val", stats_path)
    assert len(train_ds) == N_TRAIN
    w, l = train_ds[0]
    assert w.shape == (n_channels, window_len), w.shape
    assert l.shape == (label_dim,), l.shape
    print(f"dataset OK: {len(train_ds)} train / {len(val_ds)} val windows, item shapes {tuple(w.shape)} / {tuple(l.shape)}")

    # Real training loop (1 epoch, no Lightning Trainer needed for the smoke test -
    # exercises the same LightningModule.training_step/validation_step code paths).
    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)

    cfg_path = Path(f"models/{model_name}/config.yaml")
    cfg = yaml.safe_load(cfg_path.open())
    module = module_cls(cfg)
    opt = module.configure_optimizers()
    optimizer = opt["optimizer"] if isinstance(opt, dict) else opt[0][0]

    module.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = module.training_step((x, y), 0)
        loss.backward()
        optimizer.step()
    module.eval()
    with torch.no_grad():
        for x, y in val_loader:
            module.validation_step((x, y), 0)
    print("1-epoch train/val step OK, no NaN/crash")

    ckpt_path = processed_dir / "smoke.ckpt"
    torch.save({"state_dict": module.state_dict()}, ckpt_path)

    export_dir = processed_dir / "exported"
    export_dir.mkdir(exist_ok=True)
    import os

    old_cwd = os.getcwd()
    try:
        os.chdir(ROOT.parent)  # export.py reads data/processed/<model>/... relative to cwd
        (Path(f"data/processed/{model_name}")).parent.mkdir(parents=True, exist_ok=True)
        if not Path(f"data/processed/{model_name}").exists():
            os.symlink(processed_dir.resolve(), f"data/processed/{model_name}")
        export_main(str(ckpt_path), str(export_dir))
    finally:
        os.chdir(old_cwd)

    fp32 = list(export_dir.glob("*_fp32.onnx"))
    assert fp32, "no fp32 onnx written"
    session = ort.InferenceSession(str(fp32[0]))
    dummy = np.random.randn(1, n_channels, window_len).astype(np.float32)
    out = session.run(None, {session.get_inputs()[0].name: dummy})[0]
    assert out.shape == (1, label_dim)
    print(f"export + ONNX inference OK: {fp32[0].name} -> output shape {out.shape}")


def main():
    if ROOT.exists():
        shutil.rmtree(ROOT)

    from models.alignment_net.dataset import AlignmentNetDataset
    from models.alignment_net.export import main as align_export_main
    from models.alignment_net.train import AlignmentNetModule
    from models.channel_a_velocity.dataset import ChannelAVelocityDataset
    from models.channel_a_velocity.export import main as chan_a_export_main
    from models.channel_a_velocity.train import ChannelAModule

    run_one("alignment_net", 200, 9, 4, AlignmentNetDataset, AlignmentNetModule, align_export_main)
    run_one("channel_a_velocity", 200, 6, 1, ChannelAVelocityDataset, ChannelAModule, chan_a_export_main)

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
