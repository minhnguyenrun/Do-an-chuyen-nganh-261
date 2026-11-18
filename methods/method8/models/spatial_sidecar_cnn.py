"""Spatial Layout Sidecar CNN for Method 8.

Trích xuất đặc trưng đa tỷ lệ từ bản đồ mật độ bố cục (2D Layout Heatmap 22 kênh)
được tạo hoàn toàn từ dự đoán của Base RT-DETRv2 (không dùng GT).
Hòa trộn trực tiếp vào các tầng đặc trưng P3, P4, P5 của RT-DETRv2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialLayoutSidecarCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 22,  # 11 Book Prior + 11 Page Prior (từ Base predictions)
        out_dim: int = 256,     # Khớp với d_model / encoder_input_proj của RT-DETR (256)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim

        # Level 0 (P3: 80x80)
        self.stage0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

        # Level 1 (P4: 40x40)
        self.stage1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

        # Level 2 (P5: 20x20)
        self.stage2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: Layout Heatmap tensor [Batch, 22, 80, 80]
        Returns:
            list of 3 feature maps:
              - S0: [Batch, 256, 80, 80]
              - S1: [Batch, 256, 40, 40]
              - S2: [Batch, 256, 20, 20]
        """
        s0 = self.stage0(x)
        s1 = self.stage1(s0)
        s2 = self.stage2(s1)
        return [s0, s1, s2]
