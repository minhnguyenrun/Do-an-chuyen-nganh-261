"""BBox-Level Multimodal Transformer for Document Layout Classification (Pipeline B).

Mô hình phân loại nhãn layout cho từng Bounding Box (từ output của Pipeline A)
bằng cách kết hợp đồng thời 4 nguồn đặc trưng:
1. Visual: Ảnh crop của từng BBox qua CNN (MobileNetV3).
2. Textual: Văn bản OCR trong từng BBox qua Transformer / MiniLM.
3. Spatial: Tọa độ 2D hình học và tỷ lệ tương đối trên toàn trang.
4. Relational Context: Tương tác ngữ cảnh giữa tất cả các BBox trong cùng một trang
   thông qua các tầng Multi-Head Self-Attention.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models


class SpatialBoxEncoder(nn.Module):
    """Mã hóa vị trí 2D và đặc trưng hình học của Bounding Box thành vector dense."""

    def __init__(self, in_dim: int = 8, out_dim: int = 64):
        super().__init__()
        # Input: [x0, y0, x1, y1, w, h, aspect_ratio, area_ratio]
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, spatial_coords: torch.Tensor) -> torch.Tensor:
        # spatial_coords: (batch_size, num_boxes, 8)
        return self.mlp(spatial_coords)


class VisualPatchEncoder(nn.Module):
    """Trích xuất vector đặc trưng thị giác từ ảnh crop của từng Bounding Box."""

    def __init__(self, out_dim: int = 128, pretrained: bool = True):
        super().__init__()
        # Sử dụng MobileNetV3-Small siêu nhẹ làm visual backbone
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        # Bỏ classification head cuối cùng, giữ lại feature extractor (576 channels)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(576, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        # crops: (B * N, 3, H, W)
        feat = self.features(crops)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        return self.proj(feat)


class TextContentEncoder(nn.Module):
    """Mã hóa văn bản trích xuất bên trong từng Bounding Box.
    
    Hỗ trợ 2 chế độ:
    1. Nhận vector text embedding đã trích xuất sẵn (từ MiniLM / RoBERTa).
    2. Hoặc mã hóa qua Text MLP projection.
    """

    def __init__(self, in_dim: int = 384, out_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, text_embeds: torch.Tensor) -> torch.Tensor:
        # text_embeds: (batch_size, num_boxes, in_dim)
        return self.proj(text_embeds)


class BBoxMultimodalTransformer(nn.Module):
    """Mô hình cốt lõi của Pipeline B: Multimodal Transformer phân loại Bounding Box.
    
    Cơ chế tương tác liên hộp (Inter-BBox Interaction):
    Mỗi trang tài liệu có N hộp. Ta tạo chuỗi N vector embedding và đưa qua
    Multi-Head Self-Attention. Tại đây, mỗi hộp sẽ đồng thời 'quan sát' vị trí,
    chữ và hình ảnh của tất cả các hộp khác trên cùng trang để đưa ra quyết định
    nhãn phân loại chính xác nhất.
    """

    def __init__(
        self,
        num_classes: int = 11,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        dim_visual: int = 128,
        dim_text: int = 128,
        dim_spatial: int = 64,
        text_in_dim: int = 384,
        pretrained_visual: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model

        # 1. Các module trích xuất đặc trưng đơn hộp
        self.spatial_encoder = SpatialBoxEncoder(in_dim=8, out_dim=dim_spatial)
        self.visual_encoder = VisualPatchEncoder(out_dim=dim_visual, pretrained=pretrained_visual)
        self.text_encoder = TextContentEncoder(in_dim=text_in_dim, out_dim=dim_text)

        # 2. Tầng dung hợp (Fusion Projection)
        fused_dim = dim_visual + dim_text + dim_spatial
        self.fusion_proj = nn.Sequential(
            nn.Linear(fused_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # 3. Tầng Transformer Encoder liên hộp (Inter-BBox Context Modeling)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Đầu phân loại nhãn (Classification Head)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        crops: torch.Tensor,
        text_embeddings: torch.Tensor,
        spatial_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Thực thi forward pass.
        
        Args:
            crops: (B, N, 3, H, W) - Tensor ảnh crop của N hộp trên B trang.
            text_embeddings: (B, N, text_in_dim) - Vector văn bản của từng hộp.
            spatial_features: (B, N, 8) - Tọa độ hình học 2D.
            mask: (B, N) - Boolean mask (True ở các vị trí padding nếu số hộp < N_max).
            
        Returns:
            logits: (B, N, num_classes) - Điểm xác suất phân loại cho từng BBox.
        """
        B, N, C, H, W = crops.shape

        # A. Trích xuất đặc trưng thị giác
        crops_flat = crops.view(B * N, C, H, W)
        vis_feat_flat = self.visual_encoder(crops_flat)
        vis_feat = vis_feat_flat.view(B, N, -1)  # (B, N, dim_visual)

        # B. Trích xuất đặc trưng văn bản
        text_feat = self.text_encoder(text_embeddings)  # (B, N, dim_text)

        # C. Trích xuất đặc trưng vị trí 2D
        spatial_feat = self.spatial_encoder(spatial_features)  # (B, N, dim_spatial)

        # D. Hòa trộn các đặc trưng (Multimodal Fusion)
        fused = torch.cat([vis_feat, text_feat, spatial_feat], dim=-1)  # (B, N, fused_dim)
        box_tokens = self.fusion_proj(fused)  # (B, N, d_model)

        # E. Tương tác ngữ cảnh giữa các BBox trên cùng trang qua Self-Attention
        # Mọi box sẽ tự chú ý (self-attend) đến các box khác trên trang
        context_tokens = self.transformer_encoder(box_tokens, src_key_padding_mask=mask)

        # F. Phân loại nhãn cho từng BBox
        logits = self.classifier(context_tokens)  # (B, N, num_classes)
        return logits
