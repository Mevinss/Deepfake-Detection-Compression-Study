"""
Lightweight attention blocks for deepfake detection models.

Provides Channel Attention (SE-style) and a combined CBAM-style block that
can be inserted into existing CNN backbones with minimal overhead.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention block.

    Args:
        in_channels: Number of input feature-map channels.
        reduction_ratio: Channel reduction ratio for the bottleneck FC layers.
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        mid_channels = max(1, in_channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        scale = torch.sigmoid(avg_out + max_out)
        return x * scale


class SpatialAttention(nn.Module):
    """Spatial attention block using channel-wise average and max pooling.

    Args:
        kernel_size: Convolution kernel size (3 or 7 recommended).
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        scale = torch.sigmoid(self.conv(concat))
        return x * scale


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM).

    Applies channel attention followed by spatial attention.

    Args:
        in_channels: Number of input feature-map channels.
        reduction_ratio: Channel reduction ratio for :class:`ChannelAttention`.
        spatial_kernel_size: Kernel size for :class:`SpatialAttention`.
    """

    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16,
        spatial_kernel_size: int = 7,
    ):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x
