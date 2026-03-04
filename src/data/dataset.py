"""
PyTorch Dataset classes for deepfake detection.

Supports loading pre-extracted face crops from a directory layout where
each class is a subdirectory (``real/`` and ``fake/``).
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset


class DeepfakeDataset(Dataset):
    """Dataset of face-crop images for binary deepfake classification.

    Expected directory layout::

        root/
          real/
            img_0001.png
            img_0002.png
            ...
          fake/
            img_0001.png
            ...

    Args:
        root: Root directory containing ``real`` and ``fake`` subdirectories.
        transform: Albumentations (or any callable) transform applied to each
            image array (H×W×C, uint8, BGR).
        extensions: Image file extensions to include.
    """

    LABEL_MAP: Dict[str, int] = {"real": 0, "fake": 1}

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ):
        self.root = Path(root)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        for class_name, label in self.LABEL_MAP.items():
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                continue
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in extensions:
                    self.samples.append((img_path, label))

        if not self.samples:
            raise FileNotFoundError(
                f"No images found under {root}. "
                "Expected subdirectories named 'real' and 'fake'."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            raise IOError(f"Cannot read image: {img_path}")
        # Convert BGR → RGB for albumentations / torchvision consistency
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label

    def get_class_counts(self) -> Dict[str, int]:
        """Return the number of samples per class."""
        counts: Dict[str, int] = {"real": 0, "fake": 0}
        reverse_map = {v: k for k, v in self.LABEL_MAP.items()}
        for _, label in self.samples:
            counts[reverse_map[label]] += 1
        return counts
