"""
Download or initialize a pre-trained deepfake detection model.

This script creates a checkpoint with pre-trained ImageNet weights that can
be used for demonstration purposes. For best accuracy, train the model on
a deepfake dataset using src/train.py.

Usage:
    python download_model.py --model mobilenetv3
    python download_model.py --model efficientnet_b0
    python download_model.py --model ghostnet
"""

import argparse
import os
from pathlib import Path

import torch
import yaml

from src.models.classifier import build_model


def download_model(model_name: str = "mobilenetv3", output_dir: str = "checkpoints"):
    """
    Download and initialize a pre-trained model with ImageNet weights.
    
    Args:
        model_name: Name of the model (mobilenetv3, efficientnet_b0, ghostnet)
        output_dir: Directory to save the checkpoint
    """
    print("=" * 70)
    print("DEEPFAKE DETECTION MODEL INITIALIZATION")
    print("=" * 70)
    print()
    
    # Load config if exists
    config_path = Path("configs/config.yaml")
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        model_cfg = config.get("model", {})
    else:
        model_cfg = {
            'name': model_name,
            'pretrained': True,
            'use_attention': True,
            'hidden_dim': 256,
            'dropout': 0.3
        }
    
    # Override with command line model name
    model_cfg['name'] = model_name
    
    print(f"Model: {model_cfg['name']}")
    print(f"Use Attention: {model_cfg.get('use_attention', True)}")
    print(f"Hidden Dim: {model_cfg.get('hidden_dim', 256)}")
    print(f"Dropout: {model_cfg.get('dropout', 0.3)}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build model with pretrained ImageNet weights
    print("Building model with pre-trained ImageNet weights...")
    try:
        model = build_model(
            name=model_cfg['name'],
            pretrained=True,  # Always use pretrained
            use_attention=model_cfg.get('use_attention', True),
            hidden_dim=model_cfg.get('hidden_dim', 256),
            dropout=model_cfg.get('dropout', 0.3)
        )
        print("✓ Model built successfully!")
    except Exception as e:
        print(f"❌ Error building model: {e}")
        raise
    
    # Save checkpoint
    checkpoint_path = Path(output_dir) / f"{model_cfg['name']}_best.pth"
    
    print(f"\nSaving checkpoint to: {checkpoint_path}")
    
    # Create checkpoint dictionary
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_name': model_cfg['name'],
        'use_attention': model_cfg.get('use_attention', True),
        'hidden_dim': model_cfg.get('hidden_dim', 256),
        'dropout': model_cfg.get('dropout', 0.3),
        'pretrained_only': True,  # Flag indicating this is ImageNet pretrained only
        'note': 'This checkpoint contains ImageNet pre-trained weights. For better accuracy, train on a deepfake dataset.'
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Checkpoint saved!")
    
    print()
    print("=" * 70)
    print("MODEL READY FOR INFERENCE")
    print("=" * 70)
    print()
    print("⚠️  IMPORTANT NOTE:")
    print("   This model uses only ImageNet pre-trained weights.")
    print("   For accurate deepfake detection, you should train the model")
    print("   on a deepfake dataset using:")
    print()
    print("   python src/train.py --config configs/config.yaml")
    print()
    print("   You can still use this model for demonstration purposes.")
    print("=" * 70)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download or initialize a pre-trained deepfake detection model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenetv3",
        choices=["mobilenetv3", "efficientnet_b0", "ghostnet"],
        help="Model architecture to download (default: mobilenetv3)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Directory to save the checkpoint (default: checkpoints)"
    )
    
    args = parser.parse_args()
    
    download_model(args.model, args.output_dir)


if __name__ == "__main__":
    main()
