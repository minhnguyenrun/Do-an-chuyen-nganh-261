"""Context-Aware RT-DETRv2 Network with Book Layout Fusion.

Tích hợp RT-DETRv2 Pretrained cùng với:
1. BookLayoutSemanticEncoder (trích xuất E_book từ layout các trang sách)
2. BookLayoutCrossAttentionFusion (hòa trộn E_book vào 300 queries của RT-DETR)
3. Fine-tuned Classification & Box Regression Head
"""

from __future__ import annotations

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForObjectDetection

from .book_layout_encoder import BookLayoutSemanticEncoder
from .rtdetr_fusion import BookLayoutCrossAttentionFusion


class BookLayoutRTDetr(nn.Module):
    """
    Mô hình hoàn chỉnh kết hợp RT-DETRv2 và Book Layout Semantic Fusion.
    
    Pipeline:
      Ảnh trang i -> RT-DETRv2 Backbone & Encoder & Base Decoder
                  -> 300 Hidden States
                  -> Book Layout Cross-Attention Fusion (với E_book)
                  -> Classification Head & Box Regression Head
    """

    def __init__(
        self,
        pretrained_model_path: str | Path | None = None,
        num_classes: int = 17,
        num_summary_tokens: int = 8,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_summary_tokens = num_summary_tokens

        # 1. Tìm đường dẫn checkpoint RT-DETRv2 của docling
        if pretrained_model_path is None:
            hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
            snapshots = list(hf_cache.glob("models--docling-project--docling-layout-heron/snapshots/*"))
            if snapshots:
                pretrained_model_path = str(snapshots[0])
            else:
                pretrained_model_path = "docling-project/docling-layout-heron"

        print(f"[BookLayoutRTDetr] Loading pretrained RT-DETRv2 from: {pretrained_model_path}")
        self.rtdetr = AutoModelForObjectDetection.from_pretrained(str(pretrained_model_path))

        # Lưu lại id2label và label2id
        self.config = self.rtdetr.config
        self.id2label = getattr(self.config, "id2label", {})
        self.label2id = getattr(self.config, "label2id", {})

        # 2. Cầu nối sinh E_book (BookLayoutSemanticEncoder)
        d_model = self.config.d_model
        self.book_encoder = BookLayoutSemanticEncoder(
            d_model=d_model,
            num_classes=num_classes,
            num_summary_tokens=num_summary_tokens,
            encoder_layers=2,
            nhead=self.config.decoder_attention_heads,
        )

        # 3. Fusion Module hòa trộn E_book vào Object Queries
        self.fusion = BookLayoutCrossAttentionFusion(
            d_model=d_model,
            nhead=self.config.decoder_attention_heads,
            dim_feedforward=self.config.decoder_ffn_dim,
            dropout=0.1,
        )

        # 4. Refinement Heads: tinh chỉnh nhãn và box dựa trên queries đã hòa trộn
        # Khởi tạo Linear với trọng số zero để ban đầu mô hình thừa hưởng 100% dự đoán tốt của Base
        self.refined_class_head = nn.Linear(d_model, num_classes)
        with torch.no_grad():
            self.refined_class_head.weight.zero_()
            self.refined_class_head.bias.zero_()

        # Box delta head: học độ lệch nắn chỉnh bounding box (dcx, dcy, dw, dh)
        self.refined_box_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, 4),
            nn.Tanh(),  # Giới hạn điều chỉnh nhỏ
        )
        with torch.no_grad():
            self.refined_box_head[-2].weight.zero_()
            self.refined_box_head[-2].bias.zero_()

        # 5. Đóng băng các tầng nếu cần
        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Đóng băng Backbone CNN/HGNet và Hybrid Encoder để tối ưu VRAM và tốc độ."""
        for param in self.rtdetr.model.backbone.parameters():
            param.requires_grad = False
        for param in self.rtdetr.model.encoder.parameters():
            param.requires_grad = False
        print("[BookLayoutRTDetr] Backbone & Hybrid Encoder FROZEN. Only Decoder & Fusion are trainable.")

    def forward(
        self,
        pixel_values: torch.Tensor,
        e_book: torch.Tensor | None = None,
        window_pages_elements: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass:
        Args:
            pixel_values: [batch, 3, H, W]
            e_book: [batch, K, d_model] (nếu đã tính trước)
            window_pages_elements: list các trang nếu muốn tính E_book trực tiếp
        Returns:
            dict containing:
              - 'logits': [batch, 300, num_classes] (Logits đã hòa trộn ngữ cảnh sách)
              - 'pred_boxes': [batch, 300, 4] (BBox đã được nắn chỉnh)
              - 'base_logits': [batch, 300, num_classes] (Dự đoán gốc của RT-DETR)
              - 'base_boxes': [batch, 300, 4] (BBox gốc)
              - 'e_book': [batch, K, d_model]
        """
        # 1. Nếu chưa có e_book thì tính từ window_pages_elements
        if e_book is None:
            if window_pages_elements is not None:
                e_book = self.book_encoder(window_pages_elements, device=pixel_values.device)
            else:
                # Fallback: dùng learned summary tokens
                bs = pixel_values.size(0)
                e_book = self.book_encoder.book_summary_tokens.expand(bs, -1, -1)

        # 2. Forward qua RT-DETRv2 gốc để lấy hidden states của 300 queries
        outputs = self.rtdetr(pixel_values=pixel_values)
        base_logits = outputs.logits  # [batch, 300, num_classes]
        base_boxes = outputs.pred_boxes  # [batch, 300, 4] (cx, cy, w, h normalized)
        queries_hidden = outputs.last_hidden_state  # [batch, 300, d_model]

        # 3. FUSION: Object Queries hòa trộn với E_book của toàn cuốn sách
        fused_hidden = self.fusion(queries_hidden, e_book)  # [batch, 300, d_model]

        # 4. Dự đoán mới sau khi có ngữ cảnh
        refined_logits_delta = self.refined_class_head(fused_hidden)
        # Kết hợp logits gốc + delta từ ngữ cảnh sách
        refined_logits = base_logits + 0.5 * refined_logits_delta

        box_deltas = self.refined_box_head(fused_hidden) * 0.1  # Giới hạn biên độ nắn chỉnh +-10%
        refined_boxes = (base_boxes + box_deltas).clamp(0.0, 1.0)

        return {
            "logits": refined_logits,
            "pred_boxes": refined_boxes,
            "base_logits": base_logits,
            "base_boxes": base_boxes,
            "e_book": e_book,
            "fused_hidden": fused_hidden,
        }
