"""
Lightweight deepfake-detection classifier models.

Provides a unified factory function that loads a pretrained backbone
(MobileNetV3-Large, EfficientNet-B0, or GhostNet via *timm*), optionally
attaches a CBAM attention block before the classifier head, and replaces the
head with a binary classification layer.
"""

from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

from .attention import CBAM

# ---------------------------------------------------------------------------
# Supported backbone names
# ---------------------------------------------------------------------------

SUPPORTED_MODELS = ("mobilenetv3", "efficientnet_b0", "ghostnet")


# ---------------------------------------------------------------------------
# Custom classification head
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    """Fully-connected classification head for binary deepfake detection.

    Args:
        in_features: Number of input features from the backbone.
        hidden_dim: Size of the intermediate hidden layer.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------

class DeepfakeDetector(nn.Module):
    """Wrapper that combines a pretrained backbone with an optional attention
    block and a custom binary classification head.

    Args:
        backbone: Feature-extraction backbone (all layers except the final
            classifier).
        attention: Optional attention module applied to the last feature map.
            If provided, the backbone must expose feature maps (not just a
            flat vector).
        head: Binary classification head.
        use_attention: Whether to run the attention module in ``forward``.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        attention: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.attention = attention
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if self.attention is not None:
            features = self.attention(features)
            features = features.mean(dim=[2, 3])
        return self.head(features)


# ---------------------------------------------------------------------------
# Factory helpers per backbone
# ---------------------------------------------------------------------------

def _build_mobilenetv3(
    pretrained: bool,
    use_attention: bool,
    hidden_dim: int,
    dropout: float,
) -> DeepfakeDetector:
    weights = tv_models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    base = tv_models.mobilenet_v3_large(weights=weights)

    if use_attention:
        # Last conv feature map has 960 channels before adaptive avg pool
        attention = CBAM(in_channels=960)
        # Keep features module (strips adaptive pool + classifier)
        backbone = base.features
        in_features = 960
    else:
        attention = None
        # Use backbone up to (but not including) the final classifier
        backbone = nn.Sequential(base.features, base.avgpool, nn.Flatten())
        in_features = base.classifier[0].in_features

    head = ClassificationHead(in_features, hidden_dim, dropout)
    return DeepfakeDetector(backbone, head, attention)


def _build_efficientnet_b0(
    pretrained: bool,
    use_attention: bool,
    hidden_dim: int,
    dropout: float,
) -> DeepfakeDetector:
    weights = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    base = tv_models.efficientnet_b0(weights=weights)

    if use_attention:
        # Last feature map: 1280 channels (before AdaptiveAvgPool2d)
        attention = CBAM(in_channels=1280)
        backbone = base.features
        in_features = 1280
    else:
        attention = None
        backbone = nn.Sequential(base.features, base.avgpool, nn.Flatten())
        in_features = base.classifier[1].in_features

    head = ClassificationHead(in_features, hidden_dim, dropout)
    return DeepfakeDetector(backbone, head, attention)


def _build_ghostnet(
    pretrained: bool,
    use_attention: bool,
    hidden_dim: int,
    dropout: float,
) -> DeepfakeDetector:
    if not TIMM_AVAILABLE:
        raise ImportError(
            "timm is required for GhostNet. Install it with: pip install timm"
        )
    base = timm.create_model("ghostnet_100", pretrained=pretrained, num_classes=0)

    # Probe actual output dimensionality with a dummy forward pass because
    # base.num_features may not match the pooled-feature output size for all
    # timm versions of GhostNet.
    with torch.no_grad():
        _dummy = torch.zeros(1, 3, 224, 224)
        _feat = base(_dummy)
        in_features = _feat.shape[1]

    if use_attention:
        attention = CBAM(in_channels=in_features)
        # Strip the last 2 children (global_pool and classifier head) so the
        # backbone returns a 4-D feature map that the CBAM block can process.
        backbone = nn.Sequential(*list(base.children())[:-2])
    else:
        attention = None
        backbone = base  # timm model with num_classes=0 returns pooled features

    head = ClassificationHead(in_features, hidden_dim, dropout)
    return DeepfakeDetector(backbone, head, attention)


# ---------------------------------------------------------------------------
# Public factory function
# ---------------------------------------------------------------------------

def build_model(
    name: str,
    pretrained: bool = True,
    use_attention: bool = True,
    hidden_dim: int = 256,
    dropout: float = 0.3,
) -> DeepfakeDetector:
    """Build a deepfake detection model.

    Args:
        name: Backbone name — one of ``"mobilenetv3"``, ``"efficientnet_b0"``,
              or ``"ghostnet"``.
        pretrained: Whether to load ImageNet-pretrained backbone weights.
        use_attention: Whether to attach a CBAM attention module.
        hidden_dim: Hidden layer size in the classification head.
        dropout: Dropout rate in the classification head.

    Returns:
        :class:`DeepfakeDetector` instance ready for training.

    Raises:
        ValueError: If *name* is not a supported backbone.
    """
    name = name.lower().strip()
    builders = {
        "mobilenetv3": _build_mobilenetv3,
        "efficientnet_b0": _build_efficientnet_b0,
        "ghostnet": _build_ghostnet,
    }
    if name not in builders:
        raise ValueError(
            f"Unknown model: {name!r}. Choose from {list(builders.keys())}."
        )
    return builders[name](pretrained, use_attention, hidden_dim, dropout)
