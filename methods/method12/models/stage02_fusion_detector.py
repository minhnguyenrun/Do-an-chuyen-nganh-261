"""Stage 02 & 03.2: Lightweight Detector + Early Fusion (Heatmap + OCR) + Late Fusion (Anchor Cross-Attention)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.method12.common import NUM_DLN_CLASSES
from methods.method12.models.stage01_detector import Stage01LightDetector


class EarlySpatialSidecarCNN(nn.Module):
    """Multi-scale CNN encoding 19-ch [11 Book Heatmap + 8 OCR Grid] -> [P3, P4, P5]."""

    def __init__(self, in_channels: int = 19, out_dim: int = 256):
        super().__init__()
        self.stage0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, out_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, out_dim),
            nn.SiLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, out_dim),
            nn.SiLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, out_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        s0 = self.stage0(x)
        s1 = self.stage1(s0)
        s2 = self.stage2(s1)
        return [s0, s1, s2]


class LateAnchorCrossAttention(nn.Module):
    """Zero-gated Cross-Attention between Decoder Object Queries and Book Anchor Boxes."""

    def __init__(self, anchor_dim: int = 16, d_model: int = 256, num_heads: int = 8):
        super().__init__()
        self.anchor_proj = nn.Sequential(
            nn.Linear(anchor_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        queries: torch.Tensor,
        anchor_mem: torch.Tensor,
        anchor_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            queries: [B, Q, D]
            anchor_mem: [B, K, 16]
            anchor_mask: [B, K] bool (True = valid anchor)
        """
        has_any = anchor_mask.any(dim=1)  # [B]
        if not has_any.any():
            return queries

        q_in = queries
        kv = self.anchor_proj(anchor_mem.to(queries.dtype))

        # Avoid NaN in MHA when an entire row in batch has 0 valid anchors
        safe_mask = anchor_mask.clone()
        empty_rows = ~has_any
        if empty_rows.any():
            safe_mask[empty_rows, 0] = True

        key_padding_mask = ~safe_mask  # True = ignore
        attn_out, _ = self.attn(q_in, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)

        row_valid = has_any.view(-1, 1, 1).to(queries.dtype)
        return queries + (self.gate * row_valid) * self.norm(attn_out)


class Stage02FusionDetector(nn.Module):
    """Light Detector + Early (Book Heatmap + OCR Grid) + Late (Anchor Cross-Attention)."""

    def __init__(
        self,
        stage01_ckpt_path: Optional[str] = None,
        model_id: str = "docling-project/docling-layout-egret-medium",
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.base = Stage01LightDetector(model_id=model_id, freeze_backbone=freeze_backbone, num_classes=NUM_DLN_CLASSES)
        if stage01_ckpt_path is not None:
            ckpt = torch.load(stage01_ckpt_path, map_location="cpu", weights_only=False)
            state = ckpt["model"] if "model" in ckpt else ckpt
            self.base.load_state_dict(state, strict=True)
            print(f"[Stage02] Loaded Stage 01 checkpoint from: {stage01_ckpt_path}")

        d_model = self.base.config.d_model  # 256
        self.early_sidecar = EarlySpatialSidecarCNN(in_channels=19, out_dim=d_model)
        self.early_gates = nn.Parameter(torch.zeros(3))
        self.late_fusion = LateAnchorCrossAttention(anchor_dim=16, d_model=d_model, num_heads=8)

        self._current_early_feats: Optional[list[torch.Tensor]] = None
        self._current_anchor_mem: Optional[torch.Tensor] = None
        self._current_anchor_mask: Optional[torch.Tensor] = None

        # Hook 1: Early Fusion into Hybrid Encoder [P3, P4, P5]
        def encoder_pre_hook(module, inputs):
            if self._current_early_feats is not None:
                proj_feats = inputs[0]
                fused = [
                    p + self.early_gates[i] * s.to(p.dtype)
                    for i, (p, s) in enumerate(zip(proj_feats, self._current_early_feats))
                ]
                return (fused,)
            return inputs

        self.base.detector.model.encoder.register_forward_pre_hook(encoder_pre_hook)

        # Hook 2: Late Fusion right before final class_embed[-1] runs inside decoder
        def class_embed_pre_hook(module, inputs):
            if self._current_anchor_mem is not None and self._current_anchor_mask is not None:
                hidden_states = inputs[0]  # [B, Q, D]
                fused_h = self.late_fusion(
                    hidden_states,
                    self._current_anchor_mem,
                    self._current_anchor_mask,
                )
                return (fused_h,) + inputs[1:]
            return inputs

        self.base.detector.class_embed[-1].register_forward_pre_hook(class_embed_pre_hook)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.train(mode)
        return self

    def count_parameters(self) -> dict[str, int]:
        return self.base.count_parameters() if False else {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "early_sidecar": sum(p.numel() for p in self.early_sidecar.parameters()),
            "late_fusion": sum(p.numel() for p in self.late_fusion.parameters()),
        }

    def forward(
        self,
        pixel_values: torch.Tensor,
        early_map_19ch: torch.Tensor,
        anchor_mem: torch.Tensor,
        anchor_mask: torch.Tensor,
        labels: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor | dict | None]:
        self._current_early_feats = self.early_sidecar(early_map_19ch)
        self._current_anchor_mem = anchor_mem
        self._current_anchor_mask = anchor_mask

        try:
            out = self.base(pixel_values, labels=labels)
        finally:
            self._current_early_feats = None
            self._current_anchor_mem = None
            self._current_anchor_mask = None

        out["early_gates"] = self.early_gates.detach().cpu()
        out["late_gate"] = self.late_fusion.gate.detach().cpu()
        return out
