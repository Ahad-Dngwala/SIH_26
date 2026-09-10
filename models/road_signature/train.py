"""Training entry point for the road-signature classifier - MIP Section 4.4.

Trained per-corridor - pass --corridor and --n-segments explicitly
rather than assuming a single global model.

Usage (from repo root, inside the `ml` container):
    python models/road_signature/train.py --config models/road_signature/config.yaml \
        --corridor ahmedabad_sg_highway --n-segments 40
"""

import argparse

import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import DataLoader

from models.road_signature.dataset import RoadSignatureDataset
from models.road_signature.model import RoadSignatureNet


class RoadSignatureModule(pl.LightningModule):
    def __init__(self, cfg: dict, n_segments: int):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model = RoadSignatureNet(n_segments=n_segments)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg["loss"]["label_smoothing"])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self.model(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=-1)
        accuracy = (preds == y).float().mean()
        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy)
        # Section 4.7: accuracy alone is not sufficient - macro recall/F1
        # need the full-epoch prediction set, not a per-batch average, so
        # this is only a rough per-batch signal. TODO(Layer 1): accumulate
        # predictions across the val epoch and compute real macro
        # recall/F1 (sklearn.metrics.recall_score/f1_score with
        # average="macro") in on_validation_epoch_end before trusting a
        # checkpoint.

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["optimizer"]["lr"])


def main(config_path: str, corridor: str, n_segments: int):
    cfg = yaml.safe_load(open(config_path))

    stats_path = f"data/processed/road_signature/{corridor}/norm_stats.json"
    train_ds = RoadSignatureDataset(f"data/processed/road_signature/{corridor}", "train", stats_path, corridor)
    val_ds = RoadSignatureDataset(f"data/processed/road_signature/{corridor}", "val", stats_path, corridor)

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"])

    module = RoadSignatureModule(cfg, n_segments=n_segments)
    trainer = pl.Trainer(
        max_epochs=cfg["train"]["epochs"],
        callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", patience=cfg["train"]["early_stopping_patience"])],
    )
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="models/road_signature/config.yaml")
    parser.add_argument("--corridor", required=True, help="e.g. ahmedabad_sg_highway")
    parser.add_argument("--n-segments", type=int, required=True)
    args = parser.parse_args()
    main(args.config, args.corridor, args.n_segments)
