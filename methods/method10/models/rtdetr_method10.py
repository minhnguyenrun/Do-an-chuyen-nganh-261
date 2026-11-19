"""Method 10: Frozen-Backbone Feature Book-Context Fusion for RT-DETRv2.

Kiến trúc:
1. Đóng băng hoàn toàn mạng CNN Backbone của RT-DETRv2 (requires_grad = False).
2. Book Context là feature map đa tỷ lệ (P3: 80x80, P4: 40x40, P5: 20x20) được trích xuất
   từ chính Backbone đã đóng băng của các trang trong cùng cuốn sách, gom lại (trung bình có
   loại trừ chính trang mục tiêu - Zero-Leakage Leave-One-Out).
3. Hòa trộn (Zero-Gated Context Fusion) giữa Book Context Adapter và feature map chính của ảnh.
4. Fine-tuning toàn bộ nhánh Transformer (Hybrid Encoder + Decoder) và Context Fusion Adapter.
"""

from __future__ import annotations

import copy
import sys
from typing import List, Optional

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
from transformers import RTDetrV2ForObjectDetection

from methods.method10.models.class_weighted_loss import (
    ClassWeightedRTDetrLoss,
    compute_class_weighted_rtdetr_loss,
)


class ContextFusionAdapter(nn.Module):
    """Mạng adapter nhẹ tinh chỉnh Book Context Feature Maps đa tỷ lệ trước khi fusion."""

    def __init__(self, d_model: int = 256, num_scales: int = 3):
        super().__init__()
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(32, d_model),
                nn.SiLU(inplace=True),
                nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
            )
            for _ in range(num_scales)
        ])

    def forward(self, context_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self.adapters[i](f) for i, f in enumerate(context_feats)]


class BackboneContextRTDetrMethod10(nn.Module):
    def __init__(
        self,
        pretrained_model_path: str | None = None,
        num_classes: int = 17,
    ):
        super().__init__()
        self.num_classes = num_classes
        model_name = pretrained_model_path or "docling-project/docling-layout-heron"
        print(f"[Method10] Nạp pretrained RT-DETRv2 từ: {model_name}")
        try:
            self.rtdetr = RTDetrV2ForObjectDetection.from_pretrained(model_name, local_files_only=True)
        except Exception:
            self.rtdetr = RTDetrV2ForObjectDetection.from_pretrained(model_name)

        self.config = self.rtdetr.config
        d_model = self.config.d_model  # 256

        # 1. Đóng băng hoàn toàn Backbone RT-DETR
        for param in self.rtdetr.model.backbone.parameters():
            param.requires_grad = False
        self.rtdetr.model.backbone.eval()

        # Bản sao đóng băng của encoder_input_proj dùng để trừ Leave-One-Out đồng nhất với cache
        self.frozen_ctx_proj = copy.deepcopy(self.rtdetr.model.encoder_input_proj)
        for param in self.frozen_ctx_proj.parameters():
            param.requires_grad = False
        self.frozen_ctx_proj.eval()

        # 2. Context Fusion Adapter & Learnable Zero-Gates
        self.context_adapter = ContextFusionAdapter(d_model=d_model, num_scales=3)
        self.fusion_gates = nn.Parameter(torch.zeros(3))

        # Biến tạm lưu trữ trong một lần forward
        self._captured_bb_feats: Optional[List[torch.Tensor]] = None
        self._current_book_means: Optional[List[torch.Tensor]] = None
        self._current_book_counts: Optional[torch.Tensor] = None

        # 3. Hook bắt output của Backbone đã đóng băng (không cần chạy lại backbone 2 lần)
        def backbone_hook(module, inputs, outputs):
            if self._current_book_means is not None:
                self._captured_bb_feats = [
                    (f[0] if isinstance(f, (tuple, list)) else f).detach()
                    for f in outputs
                ]

        self.rtdetr.model.backbone.register_forward_hook(backbone_hook)

        # 4. Pre-hook hòa trộn Book Context vào feature map chính trước khi vào Transformer Encoder
        def encoder_pre_hook(module, inputs):
            if self._current_book_means is not None and self._captured_bb_feats is not None:
                proj_feats = inputs[0]  # [P3, P4, P5] của ảnh hiện tại
                counts = self._current_book_counts.view(-1, 1, 1, 1).to(proj_feats[0].dtype)
                valid_mask = (counts > 1.0).to(proj_feats[0].dtype)
                denom = torch.clamp(counts - 1.0, min=1.0)

                # Tính feature map đóng băng của trang hiện tại để trừ Leave-One-Out (Zero Leakage)
                # Thực hiện phép trừ trên float32 để tránh sai số bfloat16 khi sách có nhiều trang
                counts_f32 = counts.float()
                valid_f32 = valid_mask.float()
                denom_f32 = denom.float()
                with torch.no_grad():
                    page_ctx_feats = [
                        self.frozen_ctx_proj[i](self._captured_bb_feats[i]).float()
                        for i in range(len(proj_feats))
                    ]

                # Book Context = (book_mean * count - page_feat) / (count - 1)
                loo_contexts = []
                for i in range(len(proj_feats)):
                    book_mean_i = self._current_book_means[i].float()
                    ctx_i = (((book_mean_i * counts_f32 - page_ctx_feats[i]) / denom_f32) * valid_f32).to(proj_feats[i].dtype)
                    loo_contexts.append(ctx_i)

                adapted_contexts = self.context_adapter(loo_contexts)
                fused = [
                    proj_feats[i] + self.fusion_gates[i] * adapted_contexts[i]
                    for i in range(len(proj_feats))
                ]
                return (fused,)
            return inputs

        self.rtdetr.model.encoder.register_forward_pre_hook(encoder_pre_hook)

        # 5. Class-Weighted Loss (Varifocal + L1 + GIoU)
        self.criterion = ClassWeightedRTDetrLoss(self.config)
        print("[Method10] Đã đóng băng Backbone. Chỉ fine-tune Context Adapter + Projection + Transformer Encoder & Decoder.")

    def train(self, mode: bool = True):
        """Giữ backbone và frozen_ctx_proj luôn ở chế độ eval() ngay cả khi model.train()."""
        super().train(mode)
        self.rtdetr.model.backbone.eval()
        self.frozen_ctx_proj.eval()
        return self

    def get_param_groups(self, lr: float = 5e-5, gate_lr: float = 5e-4, weight_decay: float = 1e-4):
        """Tách fusion_gates (không weight_decay, LR cao hơn) khỏi phần Transformer & Adapter."""
        gate_params = [self.fusion_gates]
        other_params = [
            p for n, p in self.named_parameters()
            if p.requires_grad and n != "fusion_gates"
        ]
        return [
            {"params": other_params, "lr": lr, "weight_decay": weight_decay},
            {"params": gate_params, "lr": gate_lr, "weight_decay": 0.0},
        ]

    @torch.no_grad()
    def extract_frozen_feats(self, pixel_values: torch.Tensor) -> List[torch.Tensor]:
        """Trích xuất feature map đa tỷ lệ [256, 80, 80], [256, 40, 40], [256, 20, 20] từ backbone đóng băng."""
        self.rtdetr.model.backbone.eval()
        self.frozen_ctx_proj.eval()
        bs, _, h, w = pixel_values.shape
        pixel_mask = torch.ones((bs, h, w), dtype=torch.bool, device=pixel_values.device)
        bb_out = self.rtdetr.model.backbone(pixel_values, pixel_mask)
        feats = [f[0] if isinstance(f, (tuple, list)) else f for f in bb_out]
        return [self.frozen_ctx_proj[i](feats[i]) for i in range(len(feats))]

    def forward(
        self,
        pixel_values: torch.Tensor,
        book_means: List[torch.Tensor],
        book_counts: torch.Tensor,
        labels: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor]:
        device = pixel_values.device
        self._current_book_means = book_means
        self._current_book_counts = book_counts.to(device)
        self._captured_bb_feats = None

        outputs = self.rtdetr.model(pixel_values, labels=labels)

        self._current_book_means = None
        self._current_book_counts = None
        self._captured_bb_feats = None

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
