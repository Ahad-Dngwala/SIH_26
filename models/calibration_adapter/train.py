"""Python-side validation of the calibration-adapter mechanism - MIP
Section 4.5. NOT the on-device implementation - see module README for
why this is a simulation, not what ships. Proves the LoRA insertion
converges on a real calibration session before Layer 3 ports the same
idea into Section 7.4's on-device flow.

Usage (from repo root, inside the `ml` container):
    python models/calibration_adapter/train.py --config models/calibration_adapter/config.yaml \
        --channel-a-checkpoint <path> --channel-b-checkpoint <path> --session <path>
"""

import argparse

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from models.calibration_adapter.dataset import CalibrationDriveDataset
from models.calibration_adapter.model import attach_calibration_adapters
from models.channel_a_velocity.model import ChannelAVelocityNet
from models.channel_b_velocity.model import ChannelBVelocityNet


def main(config_path: str, channel_a_checkpoint: str, channel_b_checkpoint: str, session_path: str):
    cfg = yaml.safe_load(open(config_path))

    channel_a = ChannelAVelocityNet()
    channel_a.load_state_dict(torch.load(channel_a_checkpoint, map_location="cpu"))
    channel_b = ChannelBVelocityNet()
    channel_b.load_state_dict(torch.load(channel_b_checkpoint, map_location="cpu"))

    adapter_a, adapter_b = attach_calibration_adapters(channel_a, channel_b, rank=cfg["adapter"]["rank"])

    # TODO(Layer 1): point norm_stats_path at the base models' own
    # stats, not session-local ones - see dataset.py docstring.
    ds_a = CalibrationDriveDataset(session_path, "data/processed/channel_a_velocity/norm_stats.json", window_seconds=2.0)
    ds_b = CalibrationDriveDataset(session_path, "data/processed/channel_b_velocity/norm_stats.json", window_seconds=4.0)
    loader_a = DataLoader(ds_a, batch_size=cfg["train"]["batch_size"], shuffle=True)
    loader_b = DataLoader(ds_b, batch_size=cfg["train"]["batch_size"], shuffle=True)

    loss_fn = nn.MSELoss()
    # Only the adapter params require grad (attach_calibration_adapters
    # already froze the base models).
    optimizer = torch.optim.SGD(
        list(adapter_a.parameters()) + list(adapter_b.parameters()),
        lr=cfg["optimizer"]["lr"],
    )

    channel_a.train()
    channel_b.train()
    for epoch in range(cfg["train"]["epochs"]):
        for (xa, ya), (xb, yb) in zip(loader_a, loader_b):
            optimizer.zero_grad()
            loss = loss_fn(channel_a(xa), ya) + loss_fn(channel_b(xb), yb)
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch}: loss {loss.item():.4f}")

    # TODO(Layer 1): save via export.py, not a raw state_dict here.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="models/calibration_adapter/config.yaml")
    parser.add_argument("--channel-a-checkpoint", required=True)
    parser.add_argument("--channel-b-checkpoint", required=True)
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    main(args.config, args.channel_a_checkpoint, args.channel_b_checkpoint, args.session)
