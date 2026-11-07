"""Method 5: 64-Page Hierarchical RT-DETR with Dual-Stage (Early & Late) Book & OCR Semantic Fusion.

Cải tiến toàn diện:
1. Nhìn nhận lên đến 64 trang (Hierarchical 64-Page Book Encoder).
2. Early Fusion (Macro Context): Hòa trộn tóm tắt 64 trang sách và thống kê OCR trang hiện tại thành Page Prior.
3. Late Decision Fusion: Tinh chỉnh quyết định phân loại và nắn chỉnh BBox bằng Cross-Attention giữa Box Features và 64 trang sách.
4. KHÔNG can thiệp/nhồi nhét vào 300 Queries bên trong Decoder RT-DETR, giữ trọn vẹn khả năng định vị vật thể của Base Model.
5. Class-Weighted Varifocal Loss (CW-VFL): Cấp trọng số cao cho các lớp hiếm (Title, Caption, Footnote, Section-header, Header/Footer) để triệt tiêu hiện tượng Catastrophic Forgetting.
6. Hỗ trợ Unfreeze / Full Fine-tuning với Differential Learning Rate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import RTDetrV2ForObjectDetection

from methods.method5.models.hierarchical_book_encoder import Hierarchical64PageBookEncoder
from methods.method5.models.rtdetr_fusion import DualStageOCRFusionModule
from methods.method5.models.class_weighted_loss import ClassWeightedRTDetrLoss, compute_class_weighted_rtdetr_loss


class BookLayoutRTDetrMethod5(nn.Module):
    def __init__(
        self,
        pretrained_model_path: str | None = None,
        num_classes: int = 17,
        num_summary_tokens: int = 8,
        max_pages: int = 64,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.max_pages = max_pages

        # 1. Pretrained RT-DETRv2 sạch
        model_name = pretrained_model_path or "docling-project/docling-layout-heron"
        print(f"[Method5] Nạp pretrained RT-DETRv2 từ: {model_name}")
        self.rtdetr = RTDetrV2ForObjectDetection.from_pretrained(model_name)
        self.config = self.rtdetr.config
        d_model = self.config.d_model  # 256

        # 2. Hierarchical 64-Page Book Encoder (Intra-Page -> Inter-Page)
        self.book_encoder = Hierarchical64PageBookEncoder(
            d_model=d_model,
            nhead=self.config.decoder_attention_heads,
            dim_feedforward=self.config.decoder_ffn_dim,
            dropout=0.1,
            num_summary_tokens=num_summary_tokens,
            num_classes=num_classes,
            max_pages=max_pages,
        )

        # 3. Dual-Stage Book & OCR Semantic Fusion Module (Early & Late)
        self.fusion = DualStageOCRFusionModule(
            d_model=d_model,
            nhead=self.config.decoder_attention_heads,
            dim_feedforward=self.config.decoder_ffn_dim,
            dropout=0.1,
        )

        # 4. Refinement Heads (Zero-initialized để bảo toàn 100% năng lực gốc ban đầu)
        self.refined_class_head = nn.Linear(d_model, num_classes)
        with torch.no_grad():
            self.refined_class_head.weight.zero_()
            self.refined_class_head.bias.zero_()

        self.refined_box_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, 4),
            nn.Tanh(),
        )
        with torch.no_grad():
            self.refined_box_head[-2].weight.zero_()
            self.refined_box_head[-2].bias.zero_()

        # 5. Class-Weighted Varifocal Loss (Bù trừ mất cân bằng Long-Tail)
        self.criterion = ClassWeightedRTDetrLoss(self.config)

        # 6. Đóng băng hoặc mở khóa RT-DETR
        if freeze_backbone:
            self._freeze_rtdetr()
        else:
            print("[Method5] MỞ TOÀN BỘ RT-DETR (Không đóng băng) + Sử dụng Class-Weighted VFL & Differential LR.")

    def _freeze_rtdetr(self):
        for param in self.rtdetr.parameters():
            param.requires_grad = False
        print("[Method5] ĐÃ ĐÓNG BĂNG 100% RT-DETRv2 (Backbone + Encoder + Decoder + Heads gốc).")
        print("[Method5] CHỈ MỞ TRAIN: 64-Page Book Encoder + Dual-Stage OCR Fusion + Refined Heads.")

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: list[dict[str, torch.Tensor]] | None = None,
        context_elements: list[list[dict[str, torch.Tensor]]] | None = None,
    ) -> dict[str, torch.Tensor]:
        device = pixel_values.device
        bs = pixel_values.size(0)

        # 1. Trích xuất ngữ cảnh 64 trang & Page OCR Context
        e_book_list = []
        page_ocr_list = []

        if context_elements is not None:
            for sample_ctx in context_elements:
                eb, p_ocr = self.book_encoder(sample_ctx, device=device)
                e_book_list.append(eb)
                page_ocr_list.append(p_ocr)
            e_book = torch.cat(e_book_list, dim=0)  # [batch, K, d_model]
            page_ocr_context = torch.cat(page_ocr_list, dim=0)  # [batch, d_model]
        else:
            e_book = self.book_encoder.book_summary_queries.expand(bs, -1, -1)
            page_ocr_context = torch.zeros(bs, self.config.d_model, device=device)

        # 2. Forward RT-DETRv2 Core (Giữ nguyên vẹn 300 queries sạch)
        outputs = self.rtdetr.model(pixel_values, labels=labels)

        denoising_meta_values = outputs.denoising_meta_values if self.training else None
        outputs_class = outputs.intermediate_logits  # [batch, num_layers, 300, 17]
        outputs_coord = outputs.intermediate_reference_points  # [batch, num_layers, 300, 4]
        predicted_corners = outputs.intermediate_predicted_corners
        initial_reference_points = outputs.initial_reference_points

        base_logits = outputs_class[:, -1]
        base_boxes = outputs_coord[:, -1]
        queries_hidden = outputs.last_hidden_state  # [batch, 300, d_model]

        # 3. Late Decision Fusion: Tinh chỉnh quyết định phân loại dựa trên Book Context (64 trang) & OCR Prior
        fused_hidden = self.fusion(
            queries_hidden=queries_hidden,
            e_book=e_book,
            page_ocr_context=page_ocr_context,
        )

        # 4. Refinement Deltas
        delta_logits = self.refined_class_head(fused_hidden)
        delta_boxes = self.refined_box_head(fused_hidden) * 0.1

        fused_logits = base_logits + 0.5 * delta_logits
        fused_boxes = (base_boxes + delta_boxes).clamp(0.0, 1.0)

        result = {
            "logits": fused_logits,
            "pred_boxes": fused_boxes,
            "base_logits": base_logits,
            "base_boxes": base_boxes,
            "e_book": e_book,
            "loss": None,
            "loss_dict": None,
        }

        # 5. Class-Weighted Hungarian Matcher Loss khi Training
        if labels is not None:
            outputs_class_fused = outputs_class.clone()
            outputs_class_fused[:, -1] = fused_logits

            outputs_coord_fused = outputs_coord.clone()
            outputs_coord_fused[:, -1] = fused_boxes

            loss, loss_dict = compute_class_weighted_rtdetr_loss(
                criterion=self.criterion,
                logits=fused_logits,
                labels=labels,
                device=device,
                pred_boxes=fused_boxes,
                config=self.rtdetr.config,
                outputs_class=outputs_class_fused,
                outputs_coord=outputs_coord_fused,
                enc_topk_logits=outputs.enc_topk_logits,
                enc_topk_bboxes=outputs.enc_topk_bboxes,
                denoising_meta_values=denoising_meta_values,
            )
            result["loss"] = loss
            result["loss_dict"] = loss_dict

        return result
