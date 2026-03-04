"""
Robustness evaluation script for deepfake detection models.

For each model checkpoint and each compression level (CRF), evaluates
Accuracy, F1-score and ROC-AUC on the test set and plots the degradation
curves as a function of CRF level.

Usage:
    python src/evaluate.py --config configs/config.yaml
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader

from src.data.dataset import DeepfakeDataset
from src.data.preprocess import crf_to_jpeg_quality, get_compression_transforms, get_val_transforms
from src.models.classifier import build_model


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Compute accuracy, F1 and AUC for *model* on *loader*.

    Returns:
        Dict with keys ``"acc"``, ``"f1"``, ``"auc"``.
    """
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images).squeeze(1).cpu()
        all_logits.append(logits)
        all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    probs = torch.sigmoid(all_logits).numpy()
    preds = (probs >= 0.5).astype(int)
    labels_np = all_labels.numpy()

    acc = float((preds == labels_np).mean())
    f1 = float(f1_score(labels_np, preds, zero_division=0))
    try:
        auc = float(roc_auc_score(labels_np, probs))
    except ValueError:
        auc = float("nan")

    return {"acc": acc, "f1": f1, "auc": auc}


# ---------------------------------------------------------------------------
# Robustness evaluation across CRF levels
# ---------------------------------------------------------------------------

def evaluate_robustness(
    model: nn.Module,
    test_root: str,
    crf_levels: List[int],
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    device: Optional[torch.device] = None,
) -> Dict[int, Dict[str, float]]:
    """Evaluate *model* on the test set at each CRF compression level.

    For each CRF value, we simulate the corresponding JPEG quality degradation
    using albumentations (see :func:`~src.data.preprocess.crf_to_jpeg_quality`).

    Args:
        model: Trained deepfake detection model.
        test_root: Root directory of the test dataset (``real/`` and ``fake/``
            subdirectories).
        crf_levels: List of CRF values to evaluate (e.g. ``[0, 23, 32, 40]``).
        batch_size: DataLoader batch size.
        num_workers: DataLoader workers.
        image_size: Input image resolution.
        device: Torch device.

    Returns:
        Dict mapping each CRF level to its metric dictionary.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results: Dict[int, Dict[str, float]] = {}

    for crf in crf_levels:
        if crf == 0:
            # No compression — use plain val transforms
            transform = get_val_transforms(image_size)
        else:
            jpeg_quality = crf_to_jpeg_quality(crf)
            transform = get_compression_transforms(image_size, jpeg_quality)

        dataset = DeepfakeDataset(root=test_root, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        metrics = evaluate_model(model, loader, device)
        results[crf] = metrics
        print(
            f"  CRF={crf:2d} (JPEG~{crf_to_jpeg_quality(crf) if crf != 0 else 100}) | "
            f"acc={metrics['acc']:.4f}  f1={metrics['f1']:.4f}  auc={metrics['auc']:.4f}"
        )

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_robustness(
    results_per_model: Dict[str, Dict[int, Dict[str, float]]],
    crf_levels: List[int],
    output_dir: str = "results",
    metrics: List[str] = ("acc", "f1", "auc"),
) -> None:
    """Plot metric degradation curves for all models across CRF levels.

    Creates one figure per metric saved as a PNG file in *output_dir*.

    Args:
        results_per_model: Dict mapping model name → CRF results dict.
        crf_levels: Ordered list of CRF levels used as x-axis.
        output_dir: Directory where plots are saved.
        metrics: Metric names to plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    metric_labels = {"acc": "Accuracy", "f1": "F1-score", "auc": "ROC-AUC"}
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        for idx, (model_name, crf_results) in enumerate(results_per_model.items()):
            values = [crf_results[crf][metric] for crf in crf_levels]
            ax.plot(
                crf_levels,
                values,
                marker="o",
                label=model_name,
                color=colors[idx % len(colors)],
            )

        ax.set_xlabel("CRF Level (H.264)", fontsize=12)
        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=12)
        ax.set_title(
            f"Deepfake Detection Robustness — {metric_labels.get(metric, metric)}",
            fontsize=13,
        )
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        # Invert x-axis so CRF=0 (no compression, best quality) is on the
        # right side; performance typically degrades as CRF increases toward
        # the left (more compression, worse quality).
        ax.invert_xaxis()

        out_path = os.path.join(output_dir, f"robustness_{metric}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {out_path}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate deepfake detection model robustness across CRF levels"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default=None,
        help="Override checkpoint directory from config",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Directory to save plots and JSON results",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    with open(args.config, "r") as fh:
        cfg = yaml.safe_load(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = cfg.get("image_size", 224)
    batch_size = cfg.get("batch_size", 32)
    num_workers = cfg.get("num_workers", 4)
    test_dir = cfg["data"]["test_dir"]
    crf_levels: List[int] = cfg.get("crf_levels", [0, 23, 32, 40])
    ckpt_dir = Path(args.checkpoint_dir or cfg.get("checkpoint_dir", "checkpoints"))
    output_dir = args.output_dir

    model_names: List[str] = cfg.get("eval_models", ["mobilenetv3", "efficientnet_b0", "ghostnet"])
    model_cfg = cfg.get("model", {})

    results_per_model: Dict[str, Dict[int, Dict[str, float]]] = {}

    for model_name in model_names:
        print(f"\n=== Evaluating: {model_name} ===")
        ckpt_path = ckpt_dir / f"{model_name}_best.pth"
        if not ckpt_path.exists():
            print(f"  Checkpoint not found: {ckpt_path} — skipping.")
            continue

        model = build_model(
            name=model_name,
            pretrained=False,
            use_attention=model_cfg.get("use_attention", True),
            hidden_dim=model_cfg.get("hidden_dim", 256),
            dropout=model_cfg.get("dropout", 0.3),
        ).to(device)
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)

        results_per_model[model_name] = evaluate_robustness(
            model=model,
            test_root=test_dir,
            crf_levels=crf_levels,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            device=device,
        )

    if not results_per_model:
        print("No models evaluated — nothing to plot.")
    else:
        # Save raw results as JSON
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "robustness_results.json")
        # Convert int keys to strings for JSON serialisation
        serialisable = {
            m: {str(crf): v for crf, v in res.items()}
            for m, res in results_per_model.items()
        }
        with open(json_path, "w") as fh:
            json.dump(serialisable, fh, indent=2)
        print(f"\nSaved JSON results: {json_path}")

        plot_robustness(results_per_model, crf_levels, output_dir=output_dir)
