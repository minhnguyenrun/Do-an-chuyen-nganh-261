"""Book-Level Context Encoder and Context Fusion Modules (Method 3).

Triển khai kiến trúc phân cấp kết nối Pipeline 1 (Docling Scout) và Pipeline 2 (DINO + M2Doc):
1. RegionRepresentationMLP:
       r_{ij} = MLP([f_{ij}; e_{ij}; p_{ij}; s_{ij}])
2. PagePositionalEncoding:
       z_{ij} = r_{ij} + e_{page}(i)
3. HierarchicalBookEncoder:
       Box features {z_{ij}} -> Page Encoder -> Page Representations {R_i} -> Book Encoder -> C_{book}
4. ContextFusionModule:
       Q_i = W_Q F_i, K = W_K C_{book}, V = W_V C_{book}
       F_i' = Attention(Q_i, K, V)
       \\widetilde{F}_i = F_i + \\alpha F_i'
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionRepresentationMLP(nn.Module):
    """Mục 3.1: Xây dựng biểu diễn vector r_{ij} cho mỗi vùng j trên trang i.

    Công thức:
        r_{ij} = MLP([f_{ij}; e_{ij}; p_{ij}; s_{ij}])
    """

    def __init__(
        self,
        num_classes: int = 11,
        embed_dim: int = 256,
        visual_in_dim: int = 0,
        label_dim: int = 64,
        pos_dim: int = 128,
        score_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.visual_in_dim = visual_in_dim

        # e_{ij}: Embedding nhãn dự đoán (thêm 1 slot cho padding/unknown)
        self.label_embed = nn.Embedding(num_classes + 1, label_dim)

        # p_{ij}: Embedding tọa độ bounding box chuẩn hóa [x_min, y_min, x_max, y_max, w, h, aspect, area]
        self.pos_mlp = nn.Sequential(
            nn.Linear(8, pos_dim),
            nn.LayerNorm(pos_dim),
            nn.GELU(),
            nn.Linear(pos_dim, pos_dim),
            nn.LayerNorm(pos_dim),
        )

        # s_{ij}: Embedding confidence score
        self.score_mlp = nn.Sequential(
            nn.Linear(1, score_dim),
            nn.LayerNorm(score_dim),
            nn.GELU(),
        )

        # f_{ij}: Nhánh đặc trưng hình ảnh/đa phương thức tùy chọn
        self.vis_proj = nn.Linear(visual_in_dim, embed_dim // 2) if visual_in_dim > 0 else None
        vis_out = (embed_dim // 2) if visual_in_dim > 0 else 0

        total_in = vis_out + label_dim + pos_dim + score_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(total_in, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(
        self,
        spatial_features: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        visual_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            spatial_features: (B_pages, M, 8)
            labels: (B_pages, M) in [0..num_classes]
            scores: (B_pages, M) in [0, 1]
            visual_features: Optional (B_pages, M, visual_in_dim)

        Returns:
            r: (B_pages, M, embed_dim)
        """
        e_ij = self.label_embed(labels.clamp(0, self.num_classes))
        p_ij = self.pos_mlp(spatial_features)
        s_ij = self.score_mlp(scores.unsqueeze(-1))

        parts = [e_ij, p_ij, s_ij]
        if self.vis_proj is not None and visual_features is not None:
            f_ij = self.vis_proj(visual_features)
            parts.insert(0, f_ij)

        concat_vec = torch.cat(parts, dim=-1)
        return self.fusion_mlp(concat_vec)


class PagePositionalEncoding(nn.Module):
    """Mục 3.2: Mã hóa vị trí trang e_{page}(i).

    Kết hợp 3 thành phần:
    1. Sinusoidal encoding của chỉ số trang tuyệt đối i.
    2. Tỷ lệ vị trí tương đối trong cuốn sách (i / N) để xử lý sách có độ dài bất kỳ.
    3. Embedding tính chẵn/lẻ của trang (Recto/Verso parity - rất quan trọng vì sách in
       thường đặt lề và Header/Page-number đối xứng giữa trang chẵn và trang lẻ).
    """

    def __init__(self, embed_dim: int = 256, max_pages: int = 2048):
        super().__init__()
        self.embed_dim = embed_dim

        # Sinusoidal table
        pe = torch.zeros(max_pages, embed_dim)
        position = torch.arange(0, max_pages, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("sin_pe", pe, persistent=False)

        # Parity (trang chẵn = 0, trang lẻ = 1)
        self.parity_embed = nn.Embedding(2, embed_dim)

        # Relative progress (i / total_pages)
        self.rel_proj = nn.Sequential(
            nn.Linear(1, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        page_indices: torch.Tensor,
        total_pages: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            page_indices: (N_pages,) tensor chứa chỉ số trang 0, 1, ..., N-1.
            total_pages: (N_pages,) hoặc scalar tổng số trang N.

        Returns:
            e_page: (N_pages, embed_dim)
        """
        idx_clamped = page_indices.clamp(0, self.sin_pe.size(0) - 1).long()
        sin_part = self.sin_pe[idx_clamped]
        parity_part = self.parity_embed(idx_clamped % 2)

        if total_pages is None:
            denom = float(max(int(page_indices.max().item()) + 1, 1))
            rel_pos = (page_indices.float() / denom).unsqueeze(-1)
        else:
            rel_pos = (page_indices.float() / total_pages.float().clamp(min=1.0)).unsqueeze(-1)

        rel_part = self.rel_proj(rel_pos)
        return self.norm(sin_part + parity_part + rel_part)


class HierarchicalBookEncoder(nn.Module):
    """Mục 3.3: Kiến trúc phân cấp tổng hợp ngữ cảnh toàn sách (Page Encoder -> Book Encoder).

    Tránh ma trận self-attention O((N * M)^2) khổng lồ trên hàng chục nghìn vùng bằng 2 tầng:
    - Tầng 1 (Page Encoder): Tóm tắt M vùng của mỗi trang thành K token đại diện trang (Page Summary Tokens).
    - Tầng 2 (Book Encoder): Học quan hệ liên trang giữa các vector đại diện của N trang trong sách/cửa sổ trang.
    """

    def __init__(
        self,
        num_classes: int = 11,
        embed_dim: int = 256,
        num_summary_tokens: int = 4,
        page_encoder_layers: int = 2,
        book_encoder_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_summary_tokens = num_summary_tokens

        # 1. Biểu diễn từng vùng + vị trí trang
        self.region_encoder = RegionRepresentationMLP(
            num_classes=num_classes,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        self.page_pos_encoder = PagePositionalEncoding(embed_dim=embed_dim)

        # 2. Page Encoder: K learnable summary tokens cho mỗi trang + Transformer
        self.page_summary_queries = nn.Parameter(torch.randn(1, num_summary_tokens, embed_dim) * 0.02)
        page_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.page_transformer = nn.TransformerEncoder(page_layer, num_layers=page_encoder_layers)

        # 3. Book Encoder: Transformer liên trang trên chuỗi (N_pages * K, embed_dim)
        book_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.book_transformer = nn.TransformerEncoder(book_layer, num_layers=book_encoder_layers)
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        spatial_features: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        page_indices: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        total_pages: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            spatial_features: (N_pages, M_boxes, 8)
            labels: (N_pages, M_boxes)
            scores: (N_pages, M_boxes)
            page_indices: (N_pages,)
            padding_mask: (N_pages, M_boxes) - True tại các vị trí box padding.
            total_pages: Optional (N_pages,)

        Returns:
            c_book_tokens: (N_pages * K, embed_dim) - Chuỗi token ngữ cảnh toàn sách C_{book}.
            c_page_vectors: (N_pages, embed_dim) - Vector ngữ cảnh tổng hợp cho từng trang i.
        """
        N_pages, M_boxes, _ = spatial_features.shape

        # Bước 3.1: r_{ij}
        r_ij = self.region_encoder(spatial_features, labels, scores)  # (N_pages, M_boxes, D)

        # Bước 3.2: z_{ij} = r_{ij} + e_{page}(i)
        e_page = self.page_pos_encoder(page_indices, total_pages)  # (N_pages, D)
        z_ij = r_ij + e_page.unsqueeze(1)  # (N_pages, M_boxes, D)

        # Bước 3.3a: Page Encoder (Ghép K summary tokens vào đầu mỗi trang)
        summary_tokens = self.page_summary_queries.expand(N_pages, -1, -1) + e_page.unsqueeze(1)
        page_tokens = torch.cat([summary_tokens, z_ij], dim=1)  # (N_pages, K + M_boxes, D)

        if padding_mask is not None:
            summary_mask = torch.zeros(
                (N_pages, self.num_summary_tokens), dtype=torch.bool, device=padding_mask.device
            )
            full_page_mask = torch.cat([summary_mask, padding_mask], dim=1)
        else:
            full_page_mask = None

        encoded_page = self.page_transformer(page_tokens, src_key_padding_mask=full_page_mask)
        page_reps = encoded_page[:, : self.num_summary_tokens, :]  # (N_pages, K, D)

        # Bước 3.3b: Book Encoder (Học tương tác liên trang trên toàn bộ N_pages * K đại diện)
        flat_book_tokens = page_reps.reshape(1, N_pages * self.num_summary_tokens, self.embed_dim)
        c_book = self.book_transformer(flat_book_tokens)  # (1, N_pages * K, D)
        c_book = self.out_norm(c_book.squeeze(0))  # (N_pages * K, D)

        c_page_vectors = c_book.view(N_pages, self.num_summary_tokens, self.embed_dim).mean(dim=1)
        return c_book, c_page_vectors


class ContextFusionModule(nn.Module):
    """Mục 4: Module kết hợp ngữ cảnh toàn sách C_{book} vào đặc trưng F_i của Pipeline 2.

    Công thức Cross-Attention kèm kết nối Residual có trọng số học được \\alpha:
        Q_i = W_Q F_i
        K   = W_K C_{book},   V = W_V C_{book}
        F_i' = Attention(Q_i, K, V)
        \\widetilde{F}_i = F_i + \\alpha F_i'
    """

    def __init__(
        self,
        embed_dim: int = 256,
        nhead: int = 8,
        dropout: float = 0.1,
        init_alpha: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Hệ số \alpha khởi tạo bằng 0 để giữ nguyên hành vi pretrained của DINO+M2Doc ở bước đầu
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha), dtype=torch.float32))

    def forward(
        self,
        page_features: torch.Tensor,
        book_context: torch.Tensor,
        page_context_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            page_features: F_i có kích thước (B, N_queries, D) hoặc (N_queries, B, D)
            book_context: C_{book} có kích thước (L_ctx, D) hoặc (B, L_ctx, D)
            page_context_vec: Vector ngữ cảnh trang i (B, D) tùy chọn

        Returns:
            fused_features: \\widetilde{F}_i cùng kích thước với page_features
        """
        transposed = False
        if page_features.dim() == 3 and page_features.size(0) > 32 and page_features.size(1) <= 32:
            # DINO decoder queries có dạng (num_queries=900, bs, embed_dims=256)
            F_i = page_features.permute(1, 0, 2)
            transposed = True
        else:
            F_i = page_features

        B = F_i.size(0)
        if book_context.dim() == 2:
            C_book = book_context.unsqueeze(0).expand(B, -1, -1)
        else:
            C_book = book_context

        # Cross-Attention: Q_i = F_i, K = V = C_{book}
        attn_out, _ = self.cross_attn(query=self.norm1(F_i), key=C_book, value=C_book)
        if page_context_vec is not None:
            attn_out = attn_out + page_context_vec.unsqueeze(1)

        F_prime = attn_out + self.ffn(self.norm2(F_i + attn_out))

        # Residual connection: \widetilde{F}_i = F_i + \alpha * F_i'
        F_tilde = F_i + self.alpha * F_prime

        if transposed:
            return F_tilde.permute(1, 0, 2).contiguous()
        return F_tilde


class ContextAwareBoxRefiner(nn.Module):
    r"""Mục 4 (Cách B): Module tinh chỉnh trực tiếp ở cấp độ Bounding Box dựa trên Book Context.

    Nhận danh sách box dự đoán từ DINO + M2Doc và ngữ cảnh toàn sách C_{book} để dự đoán:
    - Logits nhãn mới cho mỗi box (11 classes DocLayNet)
    - Offset tinh chỉnh tọa độ (\Delta x, \Delta y, \Delta w, \Delta h)
    - Điểm độ tin cậy mới (Refined Confidence Gate)
    """

    def __init__(self, num_classes: int = 11, embed_dim: int = 256, nhead: int = 8):
        super().__init__()
        self.num_classes = num_classes
        self.box_encoder = RegionRepresentationMLP(num_classes=num_classes, embed_dim=embed_dim)
        self.context_fusion = ContextFusionModule(embed_dim=embed_dim, nhead=nhead, init_alpha=0.1)

        self.cls_head = nn.Linear(embed_dim, num_classes)
        self.box_offset_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 4),
            nn.Tanh(),  # Giới hạn biên độ dịch chuyển tinh chỉnh
        )
        self.conf_gate = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        spatial_features: torch.Tensor,
        init_labels: torch.Tensor,
        init_scores: torch.Tensor,
        book_context: torch.Tensor,
        page_context_vec: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        box_feats = self.box_encoder(spatial_features, init_labels, init_scores)
        fused = self.context_fusion(box_feats, book_context, page_context_vec)

        cls_logits = self.cls_head(fused)
        box_deltas = self.box_offset_head(fused) * 0.05  # Tinh chỉnh tối đa +-5% kích thước trang
        conf_mult = self.conf_gate(fused).squeeze(-1)

        return {
            "cls_logits": cls_logits,
            "box_deltas": box_deltas,
            "conf_multiplier": conf_mult,
            "fused_features": fused,
        }
