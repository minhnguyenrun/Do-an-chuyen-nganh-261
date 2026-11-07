"""Dual-Stage (Early & Late) Book & OCR Semantic Fusion Module for Method 5.

Thiết kế tối ưu theo phân tích khoa học:
1. KHÔNG can thiệp/nhồi nhét thông tin sách vào 300 Object Queries bên trong RT-DETR Decoder.
   300 Queries của Decoder được giữ hoàn toàn thuần khiết để định vị vật thể chuẩn xác.
2. Early Fusion (Macro Context Projection):
   Hòa trộn đặc trưng ngữ cảnh 64 trang sách (E_book) và thống kê OCR trang (page_ocr_context)
   thành vector Page Prior Context đại diện cho bố cục vĩ mô của trang.
3. Late Decision Fusion:
   Ở tầng phân loại quyết định cuối cùng, kết hợp đặc trưng từng hộp (Box Feature)
   với Page Prior Context và so khớp qua Cross-Attention với 64 trang sách để nắn chỉnh
   độ tin cậy phân loại (Delta Logits) và vị trí vi mô (Delta BBox).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualStageOCRFusionModule(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # 1. Early Fusion: Tổng hợp thông tin sách (64 trang) và OCR trang thành Page Prior
        self.early_fusion_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.early_norm = nn.LayerNorm(d_model)

        # 2. Late Fusion: Cross-Attention giữa Box Features và 64 trang sách (E_book)
        self.late_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.late_gate = nn.Parameter(torch.zeros(1))
        self.late_norm1 = nn.LayerNorm(d_model)

        # FFN Late Refinement
        self.late_ffn = nn.Sequential(
            nn.Linear(d_model * 2, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.late_norm2 = nn.LayerNorm(d_model)

    def extract_early_prior(self, e_book: torch.Tensor, page_ocr_context: torch.Tensor) -> torch.Tensor:
        """
        Early Stage: Kết hợp tóm tắt 64 trang sách và OCR trang hiện tại.
        Args:
            e_book: [batch, K, d_model]
            page_ocr_context: [batch, d_model]
        Returns:
            page_prior: [batch, d_model]
        """
        book_summary = e_book.mean(dim=1)  # [batch, d_model]
        combined = torch.cat([book_summary, page_ocr_context], dim=-1)  # [batch, 2 * d_model]
        page_prior = self.early_norm(self.early_fusion_mlp(combined))
        return page_prior

    def forward(
        self,
        queries_hidden: torch.Tensor,
        e_book: torch.Tensor,
        page_ocr_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Late Decision Stage: Nắn chỉnh các đặc trưng hộp dựa trên Page Prior và E_book.
        Args:
            queries_hidden: [batch, 300, d_model] (Đầu ra sạch từ RT-DETR Decoder)
            e_book: [batch, K, d_model] (Ngữ cảnh 64 trang sách)
            page_ocr_context: [batch, d_model]
        Returns:
            refined_features: [batch, 300, d_model]
        """
        bs, num_q, _ = queries_hidden.shape

        # 1. Early Prior
        page_prior = self.extract_early_prior(e_book, page_ocr_context)  # [batch, d_model]

        # 2. Late Cross-Attention: Từng hộp so chiếu với cấu trúc 64 trang sách
        attn_out, _ = self.late_cross_attn(
            query=queries_hidden,
            key=e_book,
            value=e_book,
        )
        gate = torch.tanh(self.late_gate)
        x = self.late_norm1(queries_hidden + gate * attn_out)

        # 3. Kết hợp Page Prior vào từng hộp để ra quyết định phân loại
        prior_expanded = page_prior.unsqueeze(1).expand(-1, num_q, -1)  # [batch, 300, d_model]
        combined_feat = torch.cat([x, prior_expanded], dim=-1)  # [batch, 300, 2 * d_model]

        refined_features = self.late_norm2(x + self.late_ffn(combined_feat))
        return refined_features
