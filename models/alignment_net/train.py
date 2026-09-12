"""Training entry point for the alignment net - MIP Section 4.1.

Usage (from repo root, inside the `ml` container):
    python models/alignment_net/train.py --config models/alignment_net/config.yaml
"""

import argparse

import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from models.alignment_net.dataset import AlignmentNetDataset
from models.alignment_net.model import AlignmentNet


class AlignmentNetModule(pl.LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model = AlignmentNet()
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self.model(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        self.log("val_loss", loss)

        # Section 4.1's real target metric: mean angular error in degrees,
        # not the raw MSE loss. pitch/roll are plain angles; yaw comes
        # back through atan2(sin, cos) so the diff below is wraparound-safe
        # for all three.
        pred_angles = torch.rad2deg(self.model.to_angles(pred))
        true_angles = torch.rad2deg(self.model.to_angles(y))
        diff = (pred_angles - true_angles + 180.0) % 360.0 - 180.0
        mean_angular_error_deg = diff.abs().mean()
        self.log("val_mean_angular_error_deg", mean_angular_error_deg)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams["optimizer"]["lr"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams["train"]["epochs"])
        return {"optimizer": opt, "lr_scheduler": sched}


def main(config_path: str):
    cfg = yaml.safe_load(open(config_path))

    stats_path = "data/processed/alignment_net/norm_stats.json"
    train_ds = AlignmentNetDataset("data/processed/alignment_net", "train", stats_path)
    val_ds = AlignmentNetDataset("data/processed/alignment_net", "val", stats_path)

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"])

    module = AlignmentNetModule(cfg)
    trainer = pl.Trainer(
        max_epochs=cfg["train"]["epochs"],
        callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", patience=cfg["train"]["early_stopping_patience"])],
    )
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="models/alignment_net/config.yaml")
    args = parser.parse_args()
    main(args.config)
