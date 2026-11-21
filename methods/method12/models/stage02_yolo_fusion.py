"""Stage 02 & 03.2: YOLOv26-DocLayNet + Early (Heatmap + OCR) + Late (Anchor Cross-Attention)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.method12.models.stage01_yolo import Stage01YOLODetector


class EarlyYOLOSidecarCNN(nn.Module):
    """Multi-scale CNN encoding 19-ch [11 Book Heatmap + 8 OCR Grid] -> [256@80, 512@40, 512@20]."""

    def __init__(self, in_channels: int = 19):
        super().__init__()
        self.stage0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 256, 3, padding=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.SiLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.GroupNorm(32, 512),
            nn.SiLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.GroupNorm(32, 512),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        s0 = self.stage0(x)   # [B, 256, 80, 80]
        s1 = self.stage1(s0)  # [B, 512, 40, 40]
        s2 = self.stage2(s1)  # [B, 512, 20, 20]
        return [s0, s1, s2]


class LateYOLOAnchorCrossAttention(nn.Module):
    """Zero-gated Cross-Attention between Neck P5 spatial tokens and 12.1 Book Anchor Boxes."""

    def __init__(self, anchor_dim: int = 16, feat_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.anchor_proj = nn.Sequential(
            nn.Linear(anchor_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.SiLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )
        self.attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(feat_dim)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        feat_p5: torch.Tensor,
        anchor_mem: torch.Tensor,
        anchor_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            feat_p5: [B, 512, 20, 20]
            anchor_mem: [B, K, 16]
            anchor_mask: [B, K] bool
        """
        has_any = anchor_mask.any(dim=1)
        if not has_any.any():
            return feat_p5

        bs, c, h, w = feat_p5.shape
        q = feat_p5.flatten(2).transpose(1, 2)  # [B, 400, 512]
        kv = self.anchor_proj(anchor_mem.to(feat_p5.dtype))

        safe_mask = anchor_mask.clone()
        empty_rows = ~has_any
        if empty_rows.any():
            safe_mask[empty_rows, 0] = True

        attn_out, _ = self.attn(q, kv, kv, key_padding_mask=~safe_mask, need_weights=False)
        row_valid = has_any.view(-1, 1, 1).to(feat_p5.dtype)
        fused_q = q + (self.gate * row_valid) * self.norm(attn_out)
        return fused_q.transpose(1, 2).view(bs, c, h, w)


class Stage02YOLOFusionDetector(nn.Module):
    """YOLOv26 + Early (19ch Heatmap+OCR) + Late (Anchor Cross-Attention on P5 before Detect)."""

    def __init__(
        self,
        stage01_ckpt_path: Optional[str] = None,
        pretrained_weights: Optional[str] = None,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.base = Stage01YOLODetector(pretrained_weights=pretrained_weights, freeze_backbone=freeze_backbone)
        if stage01_ckpt_path is not None:
            ckpt = torch.load(stage01_ckpt_path, map_location="cpu", weights_only=False)
            state = ckpt["model"] if "model" in ckpt else ckpt
            self.base.load_state_dict(state, strict=True)
            print(f"[Stage02] Loaded Stage 01 YOLO checkpoint: {stage01_ckpt_path}")

        self.early_sidecar = EarlyYOLOSidecarCNN(in_channels=19)
        self.early_gates = nn.Parameter(torch.zeros(3))
        self.late_fusion = LateYOLOAnchorCrossAttention(anchor_dim=16, feat_dim=512, num_heads=8)

        self._current_early_feats: Optional[list[torch.Tensor]] = None
        self._current_anchor_mem: Optional[torch.Tensor] = None
        self._current_anchor_mask: Optional[torch.Tensor] = None

        # Hooks at Neck outputs P3 (16), P4 (19), P5 (22) right before Detect (23)
        def hook_p3(m, inp, out):
            if self._current_early_feats is not None:
                return out + self.early_gates[0] * self._current_early_feats[0].to(out.dtype)
            return out

        def hook_p4(m, inp, out):
            if self._current_early_feats is not None:
                return out + self.early_gates[1] * self._current_early_feats[1].to(out.dtype)
            return out

        def hook_p5(m, inp, out):
            if self._current_early_feats is not None:
                out = out + self.early_gates[2] * self._current_early_feats[2].to(out.dtype)
            if self._current_anchor_mem is not None and self._current_anchor_mask is not None:
                out = self.late_fusion(out, self._current_anchor_mem, self._current_anchor_mask)
            return out

        self.base.det_model.model[16].register_forward_hook(hook_p3)
        self.base.det_model.model[19].register_forward_hook(hook_p4)
        self.base.det_model.model[22].register_forward_hook(hook_p5)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.train(mode)
        return self

    def count_parameters(self) -> dict[str, int]:
        return {
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
        targets: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        self._current_early_feats = self.early_sidecar(early_map_19ch)
        self._current_anchor_mem = anchor_mem
        self._current_anchor_mask = anchor_mask
        try:
            out = self.base(pixel_values, targets=targets)
        finally:
            self._current_early_feats = None
            self._current_anchor_mem = None
            self._current_anchor_mask = None

        out["early_gates"] = self.early_gates.detach().cpu()
        out["late_gate"] = self.late_fusion.gate.detach().cpu()
        return out
