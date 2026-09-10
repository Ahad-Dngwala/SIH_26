"""Training entry point for Channel B - MIP Section 4.3.

Usage (from repo root, inside the `ml` container):
    python models/channel_b_velocity/train.py --config models/channel_b_velocity/config.yaml
"""

import argparse

import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from models.channel_b_velocity.dataset import ChannelBVelocityDataset
from models.channel_b_velocity.model import ChannelBVelocityNet


class ChannelBModule(pl.LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model = ChannelBVelocityNet()
        self.loss_fn = nn.L1Loss()  # MAE

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
        rmse = torch.sqrt(torch.mean((pred - y) ** 2))  # the real Section 4.3 target metric
        self.log("val_loss", loss)
        self.log("val_rmse_mps", rmse)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams["optimizer"]["lr"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams["train"]["epochs"])
        return {"optimizer": opt, "lr_scheduler": sched}


def main(config_path: str):
    cfg = yaml.safe_load(open(config_path))

    stats_path = "data/processed/channel_b_velocity/norm_stats.json"
    train_ds = ChannelBVelocityDataset("data/processed/channel_b_velocity", "train", stats_path)
    val_ds = ChannelBVelocityDataset("data/processed/channel_b_velocity", "val", stats_path)

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"])

    module = ChannelBModule(cfg)
    trainer = pl.Trainer(
        max_epochs=cfg["train"]["epochs"],
        callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", patience=cfg["train"]["early_stopping_patience"])],
    )
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="models/channel_b_velocity/config.yaml")
    args = parser.parse_args()
    main(args.config)
