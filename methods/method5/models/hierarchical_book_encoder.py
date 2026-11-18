"""Hierarchical Book Layout Semantic Encoder for Method 5 (Supports up to 64 Pages).

Kiến trúc phân cấp 2 tầng (Hierarchical Two-Level Encoder):
- Tầng 1 (Intra-Page Encoder): Nén M vùng của mỗi trang thành 1 Page Summary Token đại diện trang p_i.
- Tầng 2 (Inter-Page Book Encoder): Học quan hệ mạch chảy bố cục xuyên suốt 64 trang sách
  qua Transformer Encoder liên trang, cô đọng thành K=8 Global Book Summary Tokens E_book.

Tích hợp Early OCR Fusion: Hòa trộn đặc trưng Text/OCR ngay từ cấp độ phần tử vùng.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Hierarchical64PageBookEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_summary_tokens: int = 8,
        num_classes: int = 17,
        max_pages: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_summary_tokens = num_summary_tokens
        self.max_pages = max_pages

        # 1. Early Fusion: Kết hợp Geometry (6-d) + Class Distribution (17-d) + Text/OCR Features (8-d)
        self.geom_proj = nn.Linear(6, 64)
        self.class_proj = nn.Linear(num_classes, 64)
        self.ocr_proj = nn.Linear(8, 64)  # 8-dim OCR stats: [char_count, word_count, aspect_ratio, density, ...]

        self.region_fusion_mlp = nn.Sequential(
            nn.Linear(64 * 3, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # 2. Vị trí tương đối của trang trong cuốn sách (Sinusoidal + Learnable Embedding)
        self.page_pos_embed = nn.Embedding(max_pages + 1, d_model)

        # 3. Tầng 1: Intra-Page Encoder (Nén các vùng trong trang thành 1 Page Summary Token)
        self.page_query_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        page_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.page_transformer = nn.TransformerEncoder(page_encoder_layer, num_layers=1)
        self.page_norm = nn.LayerNorm(d_model)

        # 4. Tầng 2: Inter-Page Book Encoder (Quan hệ giữa các trang trong chuỗi 64 trang)
        self.book_summary_queries = nn.Parameter(torch.randn(1, num_summary_tokens, d_model) * 0.02)
        book_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.book_transformer = nn.TransformerEncoder(book_encoder_layer, num_layers=2)
        self.book_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        window_pages_elements: list[dict[str, torch.Tensor]],
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            window_pages_elements: danh sách tối đa 64 trang trong cửa sổ sách.
                Mỗi trang chứa:
                  - 'boxes': [M, 4] normalized [cx, cy, w, h]
                  - 'cls_logits': [M, 17]
                  - 'ocr_stats': [M, 8] đặc trưng OCR vùng (char_count, word_count, etc.)
                  - 'page_rel_idx': vị trí trang trong cuốn sách (0 .. 63)
        Returns:
            E_book: [1, num_summary_tokens, d_model] - Ngữ cảnh toàn cuốn sách 64 trang.
            page_ocr_context: [1, d_model] - Vector đại diện OCR của trang mục tiêu hiện tại.
        """
        if device is None:
            device = self.book_summary_queries.device

        page_representations = []
        target_page_ocr = None

        N_pages = min(len(window_pages_elements), self.max_pages)

        for p_idx in range(N_pages):
            page_dict = window_pages_elements[p_idx]
            boxes = page_dict.get("boxes")
            if boxes is None or len(boxes) == 0:
                # Trang rỗng: gán learned embedding
                empty_rep = self.page_pos_embed(torch.tensor(p_idx, device=device).clamp(0, self.max_pages)).unsqueeze(0).unsqueeze(0)
                page_representations.append(empty_rep)
                continue

            boxes = boxes.to(device)
            cls_logits = page_dict["cls_logits"].to(device)

            # OCR stats
            ocr_stats = page_dict.get("ocr_stats")
            if ocr_stats is None:
                ocr_stats = torch.zeros((len(boxes), 8), device=device, dtype=torch.float32)
            else:
                ocr_stats = ocr_stats.to(device)

            # Geometry 6-d
            w = boxes[:, 2].clamp(min=1e-4)
            h = boxes[:, 3].clamp(min=1e-4)
            aspect_ratio = (w / h).unsqueeze(1).clamp(0.01, 100.0)
            area = (w * h).unsqueeze(1).clamp(0.0, 1.0)
            geom_6d = torch.cat([boxes, aspect_ratio, area], dim=1)

            # Early Fusion: Kết hợp 3 nguồn thông tin
            g_emb = self.geom_proj(geom_6d)
            c_emb = self.class_proj(cls_logits)
            o_emb = self.ocr_proj(ocr_stats)

            fused_input = torch.cat([g_emb, c_emb, o_emb], dim=1)
            elem_tokens = self.region_fusion_mlp(fused_input).unsqueeze(0)  # [1, M, d_model]

            # Tầng 1: Intra-Page Transformer
            p_pos = self.page_pos_embed(torch.tensor(p_idx, device=device).clamp(0, self.max_pages)).unsqueeze(0).unsqueeze(0)
            q_token = self.page_query_token.to(device) + p_pos  # [1, 1, d_model]
            page_seq = torch.cat([q_token, elem_tokens], dim=1)  # [1, 1 + M, d_model]

            p_enc = self.page_transformer(page_seq)
            page_rep = self.page_norm(p_enc[:, :1, :])  # [1, 1, d_model]
            page_representations.append(page_rep)

            # Lưu đại diện trang mục tiêu (ở giữa cửa sổ)
            if p_idx == N_pages // 2 or target_page_ocr is None:
                target_page_ocr = page_rep.squeeze(1)  # [1, d_model]

        if len(page_representations) == 0:
            e_book = self.book_summary_queries.to(device)
            target_page_ocr = torch.zeros(1, self.d_model, device=device)
            return e_book, target_page_ocr

        # Tầng 2: Inter-Page Transformer trên chuỗi N trang
        page_chain = torch.cat(page_representations, dim=1)  # [1, N_pages, d_model]
        book_queries = self.book_summary_queries.to(device)  # [1, K, d_model]

        full_chain = torch.cat([book_queries, page_chain], dim=1)  # [1, K + N_pages, d_model]
        book_enc = self.book_transformer(full_chain)
        e_book = self.book_norm(book_enc[:, :self.num_summary_tokens, :])  # [1, K, d_model]

        return e_book, target_page_ocr
