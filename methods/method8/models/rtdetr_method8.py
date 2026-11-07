"""Method 8: Spatial Sidecar Heatmap RT-DETR for Book Document Layout Analysis.

Cải tiến của Method 8:
- Nạp ảnh siêu tốc từ SSD đã pre-resize 640x640.
- Bản đồ nhiệt 2D Layout Heatmap 22 kênh được sinh từ dự đoán của Base RT-DETRv2 (không leak GT).
- Zero-Gated Multi-Scale Feature Injection vào P3, P4, P5 của RT-DETRv2.
- Hungarian Matching Loss + Class-Weighted Varifocal Loss.
"""

from __future__ import annotations

import sys
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RTDetrV2ForObjectDetection

from methods.method8.models.spatial_sidecar_cnn import SpatialLayoutSidecarCNN
from methods.method5.models.class_weighted_loss import ClassWeightedRTDetrLoss, compute_class_weighted_rtdetr_loss


class SpatialSidecarRTDetrMethod8(nn.Module):
    def __init__(
        self,
        pretrained_model_path: str | None = None,
        num_classes: int = 17,
        heatmap_channels: int = 22,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes

        # 1. Pretrained RT-DETRv2 sạch
        model_name = pretrained_model_path or "docling-project/docling-layout-heron"
        print(f"[Method8] Nạp pretrained RT-DETRv2 từ: {model_name}")
        self.rtdetr = RTDetrV2ForObjectDetection.from_pretrained(model_name)
        self.config = self.rtdetr.config
        d_model = self.config.d_model  # 256

        # 2. Spatial Layout Sidecar CNN (22 -> 256 ở 80x80, 40x40, 20x20)
        self.sidecar_cnn = SpatialLayoutSidecarCNN(
            in_channels=heatmap_channels,
            out_dim=d_model,
        )

        # 3. Learnable Fusion Gates (khởi tạo 0.0 để tại step 0 mô hình giống 100% gốc)
        self.fusion_gates = nn.Parameter(torch.zeros(3))

        # Lưu trữ tạm đặc trưng sidecar trong forward pass
        self._current_spatial_feats: list[torch.Tensor] | None = None

        # 4. Đăng ký Pre-hook trên Hybrid Encoder
        def encoder_pre_hook(module, inputs):
            if self._current_spatial_feats is not None:
                proj_feats = inputs[0]
                fused = []
                for i, (p, s) in enumerate(zip(proj_feats, self._current_spatial_feats)):
                    fused.append(p + self.fusion_gates[i] * s)
                return (fused,)
            return inputs

        self.rtdetr.model.encoder.register_forward_pre_hook(encoder_pre_hook)

        # 5. Class-Weighted Loss (Varifocal + L1 + GIoU)
        self.criterion = ClassWeightedRTDetrLoss(self.config)

        # 6. Đóng băng Backbone
        if freeze_backbone:
            self._freeze_backbone()
        else:
            print("[Method8] MỞ TOÀN BỘ BACKBONE & DETR để fine-tuning.")

    def _freeze_backbone(self):
        for param in self.rtdetr.model.backbone.parameters():
            param.requires_grad = False
        print("[Method8] ĐÃ ĐÓNG BĂNG CNN BACKBONE. Chỉ huấn luyện Sidecar CNN + Hybrid Encoder + Decoder.")

    def forward(
        self,
        pixel_values: torch.Tensor,
        layout_heatmaps: torch.Tensor,
        labels: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor]:
        device = pixel_values.device

        # 1. Trích xuất đặc trưng Sidecar từ 2D Layout Heatmap
        self._current_spatial_feats = self.sidecar_cnn(layout_heatmaps)

        # 2. Forward RT-DETRv2 (Pre-hook sẽ tự động hòa trộn vào P3, P4, P5)
        outputs = self.rtdetr.model(pixel_values, labels=labels)

        # Xóa cache tạm
        self._current_spatial_feats = None

        denoising_meta_values = outputs.denoising_meta_values if self.training else None
        outputs_class = outputs.intermediate_logits
        outputs_coord = outputs.intermediate_reference_points

        pred_logits = outputs_class[:, -1]
        pred_boxes = outputs_coord[:, -1]

        result = {
            "logits": pred_logits,
            "pred_boxes": pred_boxes,
            "fusion_gates": self.fusion_gates.detach().cpu(),
            "loss": None,
            "loss_dict": None,
        }

        # 3. Tính Hungarian Matching Loss khi Training
        if labels is not None:
            loss, loss_dict = compute_class_weighted_rtdetr_loss(
                criterion=self.criterion,
                logits=pred_logits,
                labels=labels,
                device=device,
                pred_boxes=pred_boxes,
                config=self.rtdetr.config,
                outputs_class=outputs_class,
                outputs_coord=outputs_coord,
                enc_topk_logits=outputs.enc_topk_logits,
                enc_topk_bboxes=outputs.enc_topk_bboxes,
                denoising_meta_values=denoising_meta_values,
            )
            result["loss"] = loss
            result["loss_dict"] = loss_dict

        return result
