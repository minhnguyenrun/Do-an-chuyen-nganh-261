"""Cross-Page Layout Fusion Module for RT-DETRv2.

Hòa trộn ngữ nghĩa bố cục toàn cuốn sách (E_book) vào các Object Queries của RT-DETRv2.
Tương tự như cách Fusion OCR đưa Text Tokens vào Visual Tokens,
module này dùng Multi-Head Cross-Attention để Object Queries hỏi thông tin bố cục toàn cục từ E_book.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BookLayoutCrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion Layer:
      - Query (Q): Object Queries / Hidden States của RT-DETR [batch, 300, d_model]
      - Key (K) & Value (V): E_book [batch, K, d_model]
    
    Đầu ra:
      - Hidden States đã được hòa trộn ngữ nghĩa sách (Context-Enhanced Queries).
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Gated Linear Unit (Cổng kiểm soát dòng thông tin ngữ cảnh)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

        # FFN block
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        e_book: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, num_queries (300), d_model]
            e_book: [batch, num_summary_tokens (K), d_model]
        Returns:
            fused_hidden_states: [batch, num_queries (300), d_model]
        """
        # Đảm bảo batch_size khớp nhau (nếu e_book có batch=1 thì expand theo batch_size)
        if e_book.size(0) != hidden_states.size(0):
            e_book = e_book.expand(hidden_states.size(0), -1, -1)

        # 1. Multihead Cross-Attention: Queries hỏi E_book
        attn_out, _ = self.cross_attn(
            query=hidden_states,
            key=e_book,
            value=e_book,
        )
        attn_out = self.dropout1(attn_out)

        # 2. Gated Residual: van điều tiết mức độ ảnh hưởng của ngữ cảnh sách
        gate_coeff = self.gate(torch.cat([hidden_states, attn_out], dim=-1))
        fused = hidden_states + gate_coeff * attn_out
        fused = self.norm1(fused)

        # 3. Feedforward network
        ffn_out = self.linear2(self.dropout2(F.relu(self.linear1(fused))))
        fused = fused + self.dropout3(ffn_out)
        fused = self.norm2(fused)

        return fused
