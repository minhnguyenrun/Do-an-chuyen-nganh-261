"""Stage 01: Lightweight Frozen-Backbone Detector (DFine / RT-DETRv2) for 11 DocLayNet classes."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForObjectDetection

from methods.method12.common import NUM_DLN_CLASSES


class Stage01LightDetector(nn.Module):
    """Lightweight Object Detector with frozen backbone for Stage 01 baseline.

    Supports both `docling-project/docling-layout-egret-medium` (DFine-m, 19.5M,
    pretrained on 150k documents) and `PekingU/rtdetr_v2_r18vd` (RT-DETRv2-R18vd, ~20M).
    """

    def __init__(
        self,
        model_id: str = "docling-project/docling-layout-egret-medium",
        freeze_backbone: bool = True,
        num_classes: int = NUM_DLN_CLASSES,
    ):
        super().__init__()
        self.model_id = model_id
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone

        try:
            self.detector = AutoModelForObjectDetection.from_pretrained(
                model_id,
                local_files_only=True,
                ignore_mismatched_sizes=True,
            )
        except Exception:
            self.detector = AutoModelForObjectDetection.from_pretrained(
                model_id,
                ignore_mismatched_sizes=True,
            )

        self.config = self.detector.config

        if freeze_backbone and hasattr(self.detector.model, "backbone"):
            for param in self.detector.model.backbone.parameters():
                param.requires_grad = False
            self.detector.model.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone and hasattr(self.detector.model, "backbone"):
            self.detector.model.backbone.eval()
        return self

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor | dict | None]:
        outputs = self.detector(pixel_values=pixel_values, labels=labels)
        return {
            "logits": outputs.logits,
            "pred_boxes": outputs.pred_boxes,
            "loss": outputs.loss if labels is not None else None,
            "loss_dict": outputs.loss_dict if labels is not None else None,
        }
