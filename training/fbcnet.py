from __future__ import annotations

import torch
from torch import nn


class FBCNet(nn.Module):
    """Compact filter-bank spatial network for fixed-size motor-imagery epochs."""

    def __init__(self, channels: int, samples: int, classes: int = 2, bands: int = 4, spatial_filters: int = 8, temporal_kernel: int = 65, stride: int = 16, dropout: float = 0.5) -> None:
        super().__init__()
        self.stride = stride
        self.filter_bank = nn.Conv2d(1, bands, (1, temporal_kernel), padding=(0, temporal_kernel // 2), bias=False)
        self.spatial = nn.Conv2d(bands, bands * spatial_filters, (channels, 1), groups=bands, bias=False)
        self.batch_norm = nn.BatchNorm2d(bands * spatial_filters)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(bands * spatial_filters * (samples // stride), classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.batch_norm(self.spatial(self.filter_bank(inputs)))
        usable = (features.shape[-1] // self.stride) * self.stride
        # Log-variance: standard FBCSP feature — linearises the power feature space.
        features = features[..., :usable].reshape(features.shape[0], features.shape[1], -1, self.stride).var(dim=-1).clamp(min=1e-6).log()
        return self.classifier(self.dropout(features.flatten(1)))
