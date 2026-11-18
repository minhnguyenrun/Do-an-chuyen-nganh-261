"""Book Layout Semantic Encoder for Method 5.

Trích xuất đặc trưng ngữ nghĩa bố cục toàn cuốn sách (Cross-Page Layout Semantics)
từ chuỗi các trang liên tiếp (Geometry, Class Distribution, Positional Info)
qua Transformer Encoder để sinh ra biểu diễn toàn cục E_book in R^(B x K x d_model).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BookLayoutSemanticEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_summary_tokens: int = 8,
        num_classes: int = 17,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_summary_tokens = num_summary_tokens

        # 1. Geometry Projection (cx, cy, w, h, aspect_ratio, area) -> 6-dim
        self.geom_proj = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, d_model),
        )

        # 2. Class Distribution Projection (one-hot or soft distribution over num_classes)
        self.class_proj = nn.Linear(num_classes, d_model)

        # 3. Relative Page Position Embedding: vị trí tương đối của trang trong cửa sổ sách
        self.page_pos_embed = nn.Embedding(32, d_model)

        # 4. Learned Summary Tokens (E_book) để nén thông tin toàn cuốn sách thành K tokens
        self.book_summary_tokens = nn.Parameter(torch.randn(1, num_summary_tokens, d_model) * 0.02)

        # 5. Transformer Encoder để tự chú ý qua lại giữa các phần tử bố cục của chuỗi trang
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        window_pages_elements: list[dict[str, torch.Tensor]],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Args:
            window_pages_elements: danh sách các trang trong cửa sổ sách của mỗi sample.
                Mỗi phần tử trang là dict chứa:
                  - 'boxes': [M, 4] dạng [cx, cy, w, h] chuẩn hóa
                  - 'cls_logits': [M, num_classes] phân bố nhãn
                  - 'page_rel_idx': int hoặc tensor (vị trí tương đối 0, 1, 2, ...)
            device: thiết bị tính toán (CUDA/CPU)
        Returns:
            E_book: Tensor kích thước [1, num_summary_tokens, d_model]
        """
        all_element_embeddings = []

        if device is None:
            device = self.book_summary_tokens.device

        for p_idx, page_dict in enumerate(window_pages_elements):
            if "boxes" not in page_dict or len(page_dict["boxes"]) == 0:
                continue

            boxes = page_dict["boxes"].to(device)
            cls_logits = page_dict["cls_logits"].to(device)
            page_rel_idx = page_dict.get("page_rel_idx", p_idx)
            if not isinstance(page_rel_idx, torch.Tensor):
                page_rel_idx = torch.tensor(page_rel_idx, device=device, dtype=torch.long)
            else:
                page_rel_idx = page_rel_idx.to(device)

            # Tính geometry features bổ sung: aspect_ratio, area
            w = boxes[:, 2].clamp(min=1e-4)
            h = boxes[:, 3].clamp(min=1e-4)
            aspect_ratio = (w / h).unsqueeze(1).clamp(0.01, 100.0)
            area = (w * h).unsqueeze(1).clamp(0.0, 1.0)
            geom_6d = torch.cat([boxes, aspect_ratio, area], dim=1)

            geom_feat = self.geom_proj(geom_6d)
            class_feat = self.class_proj(cls_logits)
            pos_feat = self.page_pos_embed(page_rel_idx.clamp(0, 31)).unsqueeze(0).expand(geom_feat.size(0), -1)

            elem_feat = geom_feat + class_feat + pos_feat
            all_element_embeddings.append(elem_feat)

        if len(all_element_embeddings) == 0:
            return self.book_summary_tokens

        sequence = torch.cat(all_element_embeddings, dim=0).unsqueeze(0)  # [1, Total_M, d_model]
        summary_tokens = self.book_summary_tokens.to(device)  # [1, K, d_model]

        # Ghép summary tokens vào đầu chuỗi: [1, K + Total_M, d_model]
        full_seq = torch.cat([summary_tokens, sequence], dim=1)

        # Forward qua Transformer Encoder
        encoded = self.transformer_encoder(full_seq)
        encoded = self.norm(encoded)

        # Lấy K tokens đầu làm vector đặc trưng E_book
        e_book = encoded[:, :self.num_summary_tokens, :]  # [1, K, d_model]
        return e_book
