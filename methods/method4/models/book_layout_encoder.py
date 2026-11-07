"""Book Layout Semantic Encoder (C_book) for RT-DETRv2.

Rút trích và nén ngữ nghĩa bố cục toàn cuốn sách thành ma trận tokens E_book in R^(K x D).
Đóng vai trò tương tự như Text/OCR Tokens trong bài toán Multimodal Fusion OCR,
nhưng đại diện cho nhịp điệu bố cục và phong cách layout xuyên suốt cuốn sách.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BookLayoutSemanticEncoder(nn.Module):
    """
    Mã hóa danh sách các đặc trưng trang (P_1, P_2, ..., P_N) trong cùng 1 cuốn sách
    thành một tập hợp K tokens ngữ nghĩa sách cố định (E_book).
    
    Đầu vào mỗi trang p:
      - bbox_coords: [num_boxes, 4] (cx, cy, w, h normalized)
      - class_priors: [num_boxes, num_classes] (xác suất hoặc one-hot nhãn)
      - visual_embeds: [num_boxes, d_model] (đặc trưng thị giác từ RT-DETR queries)
      
    Đầu ra:
      - E_book: [batch_size, num_summary_tokens, d_model]
    """

    def __init__(
        self,
        d_model: int = 256,
        num_classes: int = 17,
        num_summary_tokens: int = 8,
        encoder_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.num_summary_tokens = num_summary_tokens

        # 1. Project tọa độ hình học (cx, cy, w, h, area, aspect_ratio) -> d_model // 4
        self.geo_proj = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, d_model // 4),
        )

        # 2. Project class distribution -> d_model // 4
        self.cls_proj = nn.Sequential(
            nn.Linear(num_classes, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, d_model // 4),
        )

        # 3. Fusion Linear: kết hợp Visual (d_model) + Geo (d_model//4) + Cls (d_model//4) -> d_model
        in_dim = d_model + (d_model // 4) * 2
        self.element_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # 4. Học K tokens tóm tắt sách (Learned Book Summary Tokens)
        self.book_summary_tokens = nn.Parameter(
            torch.randn(1, num_summary_tokens, d_model) * 0.02
        )

        # 5. Transformer Encoder để giao tiếp giữa các trang trong sách và summary tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        window_pages_elements: list[dict[str, torch.Tensor]],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Args:
            window_pages_elements: Danh sách các trang trong cửa sổ sách (Window of Pages).
                Mỗi trang là 1 dict chứa:
                  - 'bboxes': Tensor [M_i, 4] (cx, cy, w, h)
                  - 'class_logits': Tensor [M_i, num_classes]
                  - 'visual_feats': Tensor [M_i, d_model]
        Returns:
            E_book: Tensor shape [1, num_summary_tokens, d_model]
        """
        target_device = device or self.book_summary_tokens.device
        if not window_pages_elements:
            # Fallback nếu không có trang ngữ cảnh
            bs = 1
            return self.book_summary_tokens.expand(bs, -1, -1)

        page_tokens = []
        for page in window_pages_elements:
            boxes = page.get("bboxes")
            cls_logits = page.get("class_logits")
            vis_feats = page.get("visual_feats")

            if boxes is None or boxes.shape[0] == 0:
                continue

            # Đảm bảo tensor nằm trên target_device
            boxes = boxes.to(target_device)
            # Trích xuất 6 thuộc tính hình học: cx, cy, w, h, area, aspect
            w = boxes[:, 2].clamp(min=1e-4)
            h = boxes[:, 3].clamp(min=1e-4)
            area = (w * h).unsqueeze(1)
            aspect = (w / h).unsqueeze(1)
            geo_features = torch.cat([boxes, area, aspect], dim=1)  # [M_i, 6]
            geo_embed = self.geo_proj(geo_features)

            if cls_logits is None:
                cls_probs = torch.zeros(boxes.shape[0], self.num_classes, device=target_device)
            else:
                cls_probs = F.softmax(cls_logits.to(target_device), dim=-1)
            cls_embed = self.cls_proj(cls_probs)

            if vis_feats is None:
                vis_embed = torch.zeros(boxes.shape[0], self.d_model, device=target_device)
            else:
                vis_embed = vis_feats.to(target_device)

            # Hòa trộn đặc trưng từng phần tử layout
            elem_cat = torch.cat([vis_embed, geo_embed, cls_embed], dim=-1)
            elem_tokens = self.element_proj(elem_cat)  # [M_i, d_model]
            page_tokens.append(elem_tokens)

        if not page_tokens:
            return self.book_summary_tokens

        # Ghép tất cả các phần tử layout của toàn bộ cửa sổ sách
        all_book_elements = torch.cat(page_tokens, dim=0).unsqueeze(0)  # [1, Total_M, d_model]

        # Ghép cùng với K summary tokens
        bs = all_book_elements.size(0)
        summary_tokens = self.book_summary_tokens.expand(bs, -1, -1)
        seq_input = torch.cat([summary_tokens, all_book_elements], dim=1)  # [1, K + Total_M, d_model]

        # Giới hạn độ dài tối đa để an toàn bộ nhớ (ví dụ tối đa 1200 tokens)
        if seq_input.size(1) > 1200:
            seq_input = torch.cat([seq_input[:, :self.num_summary_tokens, :], seq_input[:, self.num_summary_tokens:1200, :]], dim=1)

        encoded = self.transformer(seq_input)
        # Chỉ lấy ra K summary tokens đầu tiên (chính là E_book)
        e_book = self.out_norm(encoded[:, :self.num_summary_tokens, :])  # [1, K, d_model]
        return e_book
