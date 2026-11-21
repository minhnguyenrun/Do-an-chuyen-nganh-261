"""Stage 01: YOLOv26-DocLayNet (100% GPU loss, Frozen Backbone) for 11 DocLayNet classes."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.cfg import get_cfg

from methods.method12.common import ROOT_DIR

DEFAULT_YOLO_WEIGHTS = ROOT_DIR / "methods" / "method9" / "weights" / "yolo26_doclaynet_pretrained.pt"


def gpu_photometric_augment(pixs: torch.Tensor) -> torch.Tensor:
    """Fast GPU-only brightness/contrast jitter for [B, 3, H, W] in [0, 1]."""
    bs = pixs.shape[0]
    device = pixs.device
    alpha = torch.empty((bs, 1, 1, 1), device=device, dtype=pixs.dtype).uniform_(0.88, 1.12)
    beta = torch.empty((bs, 1, 1, 1), device=device, dtype=pixs.dtype).uniform_(-0.06, 0.06)
    return (pixs * alpha + beta).clamp_(0.0, 1.0)


class Stage01YOLODetector(nn.Module):
    """YOLOv26-DocLayNet with frozen backbone (layers 0..9) for fast 100%-GPU fine-tuning."""

    def __init__(
        self,
        pretrained_weights: Optional[str] = None,
        freeze_backbone: bool = True,
        backbone_end_idx: int = 10,
    ):
        super().__init__()
        w_path = str(pretrained_weights or DEFAULT_YOLO_WEIGHTS)
        yolo_obj = YOLO(w_path)
        self.det_model = yolo_obj.model
        self.freeze_backbone = freeze_backbone
        self.backbone_end_idx = backbone_end_idx

        self.det_model.requires_grad_(True)
        if freeze_backbone:
            for idx in range(min(backbone_end_idx, len(self.det_model.model))):
                for p in self.det_model.model[idx].parameters():
                    p.requires_grad = False
                self.det_model.model[idx].eval()

        self.criterion = None
        self._crit_device = None

    def _ensure_criterion(self, device: torch.device):
        if self.criterion is None or self._crit_device != device:
            self.det_model.to(device)
            self.criterion = self.det_model.init_criterion()
            cfg = get_cfg()
            if hasattr(self.criterion, "one2many"):
                self.criterion.one2many.hyp = cfg
            if hasattr(self.criterion, "one2one"):
                self.criterion.one2one.hyp = cfg
            if hasattr(self.criterion, "hyp"):
                self.criterion.hyp = cfg
            self._crit_device = device

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            for idx in range(min(self.backbone_end_idx, len(self.det_model.model))):
                self.det_model.model[idx].eval()
        return self

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def forward(
        self,
        pixel_values: torch.Tensor,
        targets: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        preds = self.det_model(pixel_values)
        if targets is not None:
            self._ensure_criterion(pixel_values.device)
            loss_tensor, loss_items = self.criterion(preds, targets)
            loss = loss_tensor.sum()
            return {"preds": preds, "loss": loss, "loss_items": loss_items}
        return {"preds": preds, "loss": None}
