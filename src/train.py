"""
Training pipeline for deepfake detection models.

Supports both a PyTorch Lightning module and a standalone pure-PyTorch
training loop so the project works without Lightning installed.

Usage (Lightning):
    python src/train.py --config configs/config.yaml

Usage (pure PyTorch):
    python src/train.py --config configs/config.yaml --no_lightning
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False

from src.data.dataset import DeepfakeDataset
from src.data.preprocess import get_train_transforms, get_val_transforms
from src.models.classifier import build_model


# ---------------------------------------------------------------------------
# PyTorch Lightning module
# ---------------------------------------------------------------------------

class DeepfakeLightningModule(pl.LightningModule if LIGHTNING_AVAILABLE else object):
    """LightningModule wrapping a :class:`~src.models.classifier.DeepfakeDetector`.

    Args:
        model_name: Backbone identifier passed to :func:`~src.models.classifier.build_model`.
        pretrained: Whether to use pretrained backbone weights.
        use_attention: Whether to attach CBAM attention.
        hidden_dim: Hidden dimension in the classification head.
        dropout: Dropout probability.
        lr: Initial learning rate.
        weight_decay: L2 regularisation coefficient.
        epochs: Total training epochs (used for LR scheduler period).
    """

    def __init__(
        self,
        model_name: str = "mobilenetv3",
        pretrained: bool = True,
        use_attention: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        epochs: int = 20,
    ):
        super().__init__()
        if LIGHTNING_AVAILABLE:
            self.save_hyperparameters()
        self.model = build_model(
            name=model_name,
            pretrained=pretrained,
            use_attention=use_attention,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs

    # ------------------------------------------------------------------ #
    # forward / step helpers                                               #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(1)

    def _shared_step(self, batch, stage: str):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels.float())
        preds = (torch.sigmoid(logits) >= 0.5).long()
        acc = (preds == labels).float().mean()
        self.log(f"{stage}/loss", loss, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/acc", acc, on_epoch=True, prog_bar=True)
        return loss, logits.detach().cpu(), labels.detach().cpu()

    # ------------------------------------------------------------------ #
    # Lightning hooks                                                      #
    # ------------------------------------------------------------------ #

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self._shared_step(batch, "val")
        return {"loss": loss, "logits": logits, "labels": labels}

    def validation_epoch_end(self, outputs):
        all_logits = torch.cat([o["logits"] for o in outputs])
        all_labels = torch.cat([o["labels"] for o in outputs])
        probs = torch.sigmoid(all_logits).numpy()
        preds = (probs >= 0.5).astype(int)
        labels_np = all_labels.numpy()

        f1 = f1_score(labels_np, preds, zero_division=0)
        try:
            auc = roc_auc_score(labels_np, probs)
        except ValueError:
            auc = float("nan")

        self.log("val/f1", f1, prog_bar=True)
        self.log("val/auc", auc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-6
        )
        return [optimizer], [scheduler]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_weighted_sampler(dataset: "DeepfakeDataset") -> WeightedRandomSampler:
    """Build a :class:`WeightedRandomSampler` that balances real vs fake samples."""
    counts = dataset.get_class_counts()
    n_real, n_fake = counts["real"], counts["fake"]
    weights = [
        1.0 / n_real if lbl == 0 else 1.0 / n_fake
        for _, lbl in dataset.samples
    ]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ---------------------------------------------------------------------------
# Pure-PyTorch training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Run one training epoch and return loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().to(device)
        optimizer.zero_grad()
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += images.size(0)

    return {"loss": total_loss / total, "acc": correct / total}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on *loader* and return loss, accuracy, F1 and AUC."""
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        labels_dev = labels.float().to(device)
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels_dev)
        total_loss += loss.item() * images.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    probs = torch.sigmoid(all_logits).numpy()
    preds = (probs >= 0.5).astype(int)
    labels_np = all_labels.numpy()

    acc = (preds == labels_np).mean()
    f1 = f1_score(labels_np, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels_np, probs)
    except ValueError:
        auc = float("nan")

    return {
        "loss": total_loss / len(all_labels),
        "acc": float(acc),
        "f1": float(f1),
        "auc": float(auc),
    }


def run_pure_pytorch(cfg: dict) -> None:
    """Full training loop using plain PyTorch (no Lightning required)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = cfg.get("image_size", 224)

    train_ds = DeepfakeDataset(
        root=cfg["data"]["train_dir"],
        transform=get_train_transforms(image_size),
    )
    val_ds = DeepfakeDataset(
        root=cfg["data"]["val_dir"],
        transform=get_val_transforms(image_size),
    )

    # Weighted sampler to handle class imbalance
    sampler = _make_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 32),
        sampler=sampler,
        num_workers=cfg.get("num_workers", 4),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
    )

    model_cfg = cfg.get("model", {})
    model = build_model(
        name=model_cfg.get("name", "mobilenetv3"),
        pretrained=model_cfg.get("pretrained", True),
        use_attention=model_cfg.get("use_attention", True),
        hidden_dim=model_cfg.get("hidden_dim", 256),
        dropout=model_cfg.get("dropout", 0.3),
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.get("epochs", 20), eta_min=1e-6
    )
    criterion = nn.BCEWithLogitsLoss()

    # TensorBoard writer (optional)
    try:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = cfg.get("log_dir", "runs/experiment")
        writer = SummaryWriter(log_dir=log_dir)
    except ImportError:
        writer = None

    best_val_f1 = 0.0
    ckpt_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_name = model_cfg.get("name", "mobilenetv3")

    for epoch in range(1, cfg.get("epochs", 20) + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f} "
            f"f1={val_metrics['f1']:.4f} auc={val_metrics['auc']:.4f}"
        )

        if writer:
            for k, v in train_metrics.items():
                writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            ckpt_path = ckpt_dir / f"{model_name}_best.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Saved best model to {ckpt_path}")

    if writer:
        writer.close()


def run_lightning(cfg: dict) -> None:
    """Training loop using PyTorch Lightning."""
    if not LIGHTNING_AVAILABLE:
        raise ImportError(
            "pytorch-lightning is not installed. "
            "Run: pip install pytorch-lightning  or use --no_lightning flag."
        )

    image_size = cfg.get("image_size", 224)
    train_ds = DeepfakeDataset(
        root=cfg["data"]["train_dir"],
        transform=get_train_transforms(image_size),
    )
    val_ds = DeepfakeDataset(
        root=cfg["data"]["val_dir"],
        transform=get_val_transforms(image_size),
    )

    sampler = _make_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 32),
        sampler=sampler,
        num_workers=cfg.get("num_workers", 4),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
    )

    model_cfg = cfg.get("model", {})
    lit_model = DeepfakeLightningModule(
        model_name=model_cfg.get("name", "mobilenetv3"),
        pretrained=model_cfg.get("pretrained", True),
        use_attention=model_cfg.get("use_attention", True),
        hidden_dim=model_cfg.get("hidden_dim", 256),
        dropout=model_cfg.get("dropout", 0.3),
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 1e-4),
        epochs=cfg.get("epochs", 20),
    )

    log_dir = cfg.get("log_dir", "runs")
    logger = TensorBoardLogger(save_dir=log_dir, name=model_cfg.get("name", "model"))
    ckpt_dir = cfg.get("checkpoint_dir", "checkpoints")

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"{model_cfg.get('name', 'model')}_{{epoch:02d}}_{{val/f1:.4f}}",
            monitor="val/f1",
            mode="max",
            save_top_k=1,
        ),
        EarlyStopping(monitor="val/f1", patience=5, mode="max"),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.get("epochs", 20),
        logger=logger,
        callbacks=callbacks,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
    )
    trainer.fit(lit_model, train_loader, val_loader)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train deepfake detection model")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--no_lightning", action="store_true",
        help="Use pure PyTorch training loop instead of Lightning",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    with open(args.config, "r") as fh:
        config = yaml.safe_load(fh)

    if args.no_lightning or not LIGHTNING_AVAILABLE:
        run_pure_pytorch(config)
    else:
        run_lightning(config)
