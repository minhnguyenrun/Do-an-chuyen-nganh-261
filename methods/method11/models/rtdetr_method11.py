"""Method 11: Frozen-Backbone Residual-Concat Book-Context Fusion + Objective VFL Loss for RT-DETRv2.

Cải tiến so với Method 10:
1. Residual Concat Fusion: ghép [page_feat (256), loo_book_ctx (256)] -> 512 kênh, qua khối
   Conv1x1(512->256) -> GN -> SiLU -> Conv3x3 -> GN -> SiLU -> Conv1x1(zero-init) để cộng residual
   trực tiếp vào page_feat (bảo toàn 100% đường truyền pretrained RT-DETRv2).
2. Objective mAP-Aligned Loss: dùng RTDetrLoss chuẩn (VFL đồng đều 1.0 cho cả 11 lớp + L1 + GIoU),
   loại bỏ trọng số lệch 4.5x gây triệt tiêu mAP giữa Title/Caption và Section/Page-header/Text.
3. Float32 Leave-One-Out + Context Dropout (0.1) khi train để chống overfitting vào 2,587 sách Train.
"""

from __future__ import annotations

import copy
import sys
from typing import List, Optional

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
from transformers import RTDetrV2ForObjectDetection
from transformers.loss.loss_rt_detr import RTDetrLoss, _set_aux_loss


class ResidualConcatFusionBlock(nn.Module):
    """Residual Concat Fusion: page_feat + delta(cat([page_feat, book_ctx])).

    Lớp Conv cuối được khởi tạo = 0 để tại Step 0 mô hình giữ nguyên 100% đầu vào
    của Hybrid Encoder pretrained (docling-layout-heron).
    """

    def __init__(self, d_model: int = 256, num_scales: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_scales):
            block = nn.Sequential(
                nn.Conv2d(d_model * 2, d_model, kernel_size=1, bias=False),
                nn.GroupNorm(32, d_model),
                nn.SiLU(inplace=True),
                nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(32, d_model),
                nn.SiLU(inplace=True),
                nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
            )
            nn.init.zeros_(block[-1].weight)
            self.blocks.append(block)
        self.last_delta_ratios: List[float] = [0.0] * num_scales

    def forward(self, page_feats: List[torch.Tensor], ctx_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        fused = []
        ratios = []
        for i in range(len(page_feats)):
            cat_512 = torch.cat([page_feats[i], ctx_feats[i]], dim=1)
            delta = self.blocks[i](cat_512)
            fused.append(page_feats[i] + delta)
            with torch.no_grad():
                p_norm = page_feats[i].float().norm() + 1e-6
                d_norm = delta.float().norm()
                ratios.append(float((d_norm / p_norm).item()))
        self.last_delta_ratios = ratios
        return fused


def compute_objective_rtdetr_loss(
    criterion: RTDetrLoss,
    logits: torch.Tensor,
    labels: list[dict[str, torch.Tensor]],
    device: torch.device,
    pred_boxes: torch.Tensor,
    config,
    outputs_class=None,
    outputs_coord=None,
    enc_topk_logits=None,
    enc_topk_bboxes=None,
    denoising_meta_values=None,
):
    criterion.to(device)
    outputs_loss = {
        "logits": logits,
        "pred_boxes": pred_boxes,
    }

    if config.auxiliary_loss:
        if denoising_meta_values is not None:
            dn_out_coord, outputs_coord = torch.split(outputs_coord, denoising_meta_values["dn_num_split"], dim=2)
            dn_out_class, outputs_class = torch.split(outputs_class, denoising_meta_values["dn_num_split"], dim=2)
            outputs_loss["logits"] = outputs_class[:, -1]
            outputs_loss["pred_boxes"] = outputs_coord[:, -1]

        auxiliary_outputs = _set_aux_loss(outputs_class[:, :-1].transpose(0, 1), outputs_coord[:, :-1].transpose(0, 1))
        outputs_loss["auxiliary_outputs"] = auxiliary_outputs
        outputs_loss["auxiliary_outputs"].extend(_set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))
        if denoising_meta_values is not None:
            outputs_loss["dn_auxiliary_outputs"] = _set_aux_loss(
                dn_out_class.transpose(0, 1), dn_out_coord.transpose(0, 1)
            )
            outputs_loss["denoising_meta_values"] = denoising_meta_values

    loss_dict = criterion(outputs_loss, labels)
    loss = sum(loss_dict.values())
    return loss, loss_dict


class ResidualConcatRTDetrMethod11(nn.Module):
    def __init__(
        self,
        pretrained_model_path: str | None = None,
        num_classes: int = 17,
        ctx_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ctx_dropout = ctx_dropout
        model_name = pretrained_model_path or "docling-project/docling-layout-heron"
        print(f"[Method11] Nạp pretrained RT-DETRv2 từ: {model_name}")
        try:
            self.rtdetr = RTDetrV2ForObjectDetection.from_pretrained(model_name, local_files_only=True)
        except Exception:
            self.rtdetr = RTDetrV2ForObjectDetection.from_pretrained(model_name)

        self.config = self.rtdetr.config
        d_model = self.config.d_model  # 256

        # 1. Đóng băng hoàn toàn Backbone RT-DETR & bản sao projection để trừ Leave-One-Out
        for param in self.rtdetr.model.backbone.parameters():
            param.requires_grad = False
        self.rtdetr.model.backbone.eval()

        self.frozen_ctx_proj = copy.deepcopy(self.rtdetr.model.encoder_input_proj)
        for param in self.frozen_ctx_proj.parameters():
            param.requires_grad = False
        self.frozen_ctx_proj.eval()

        # 2. Residual Concat Fusion Block (512 -> 256, zero-init)
        self.fusion_block = ResidualConcatFusionBlock(d_model=d_model, num_scales=3)

        self._captured_bb_feats: Optional[List[torch.Tensor]] = None
        self._current_book_means: Optional[List[torch.Tensor]] = None
        self._current_book_counts: Optional[torch.Tensor] = None

        # 3. Hook bắt output của Backbone đóng băng trong 1 pass
        def backbone_hook(module, inputs, outputs):
            if self._current_book_means is not None:
                self._captured_bb_feats = [
                    (f[0] if isinstance(f, (tuple, list)) else f).detach()
                    for f in outputs
                ]

        self.rtdetr.model.backbone.register_forward_hook(backbone_hook)

        # 4. Pre-hook Residual Concat Fusion trước Hybrid Encoder
        def encoder_pre_hook(module, inputs):
            if self._current_book_means is not None and self._captured_bb_feats is not None:
                proj_feats = inputs[0]
                bs = proj_feats[0].shape[0]
                counts_f32 = self._current_book_counts.view(-1, 1, 1, 1).float()
                valid_f32 = (counts_f32 > 1.0).float()
                denom_f32 = torch.clamp(counts_f32 - 1.0, min=1.0)

                if self.training and self.ctx_dropout > 0.0:
                    keep_mask = (torch.rand((bs, 1, 1, 1), device=proj_feats[0].device) >= self.ctx_dropout).float()
                    valid_f32 = valid_f32 * keep_mask

                with torch.no_grad():
                    page_ctx_feats = [
                        self.frozen_ctx_proj[i](self._captured_bb_feats[i]).float()
                        for i in range(len(proj_feats))
                    ]

                loo_contexts = []
                for i in range(len(proj_feats)):
                    book_mean_i = self._current_book_means[i].float()
                    ctx_i = (((book_mean_i * counts_f32 - page_ctx_feats[i]) / denom_f32) * valid_f32).to(proj_feats[i].dtype)
                    loo_contexts.append(ctx_i)

                fused = self.fusion_block(proj_feats, loo_contexts)
                return (fused,)
            return inputs

        self.rtdetr.model.encoder.register_forward_pre_hook(encoder_pre_hook)

        # 5. Objective mAP-Aligned Loss (Standard RTDetrLoss - đồng đều 11 lớp)
        self.criterion = RTDetrLoss(self.config)
        print("[Method11] Backbone đóng băng | Fusion: Residual Concat (512->256) | Loss: Objective Standard VFL.")

    def train(self, mode: bool = True):
        """Giữ backbone và frozen_ctx_proj luôn ở eval() ngay cả khi model.train()."""
        super().train(mode)
        self.rtdetr.model.backbone.eval()
        self.frozen_ctx_proj.eval()
        return self

    def get_param_groups(self, lr: float = 5e-5, fusion_lr: float = 1e-4, weight_decay: float = 1e-4):
        """Tách learning rate cho fusion_block và Transformer."""
        fusion_params = list(self.fusion_block.parameters())
        fusion_ids = {id(p) for p in fusion_params}
        other_params = [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in fusion_ids
        ]
        return [
            {"params": other_params, "lr": lr, "weight_decay": weight_decay},
            {"params": fusion_params, "lr": fusion_lr, "weight_decay": weight_decay},
        ]

    @torch.no_grad()
    def extract_frozen_feats(self, pixel_values: torch.Tensor) -> List[torch.Tensor]:
        self.rtdetr.model.backbone.eval()
        self.frozen_ctx_proj.eval()
        bs, _, h, w = pixel_values.shape
        pixel_mask = torch.ones((bs, h, w), dtype=torch.bool, device=pixel_values.device)
        bb_out = self.rtdetr.model.backbone(pixel_values, pixel_mask)
        feats = [f[0] if isinstance(f, (tuple, list)) else f for f in bb_out]
        return [self.frozen_ctx_proj[i](feats[i]) for i in range(len(feats))]

    def forward(
        self,
        pixel_values: torch.Tensor,
        book_means: List[torch.Tensor],
        book_counts: torch.Tensor,
        labels: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor]:
        device = pixel_values.device
        self._current_book_means = book_means
        self._current_book_counts = book_counts.to(device)
        self._captured_bb_feats = None

        outputs = self.rtdetr.model(pixel_values, labels=labels)

        self._current_book_means = None
        self._current_book_counts = None
        self._captured_bb_feats = None

        denoising_meta_values = outputs.denoising_meta_values if self.training else None
        outputs_class = outputs.intermediate_logits
        outputs_coord = outputs.intermediate_reference_points

        pred_logits = outputs_class[:, -1]
        pred_boxes = outputs_coord[:, -1]

        result = {
            "logits": pred_logits,
            "pred_boxes": pred_boxes,
            "delta_ratios": list(self.fusion_block.last_delta_ratios),
            "loss": None,
            "loss_dict": None,
        }

        if labels is not None:
            loss, loss_dict = compute_objective_rtdetr_loss(
                criterion=self.criterion,
                logits=pred_logits,
                labels=labels,
                device=device,
                pred_boxes=pred_boxes,
                config=self.rtdetr.config,
                outputs_class=outputs_class,
                outputs_coord=outputs_coord,
                enc_topk_logits=outputs.enc_topk_logits,
                enc_topk_bboxes=outputs.enc_topk_bboxes,
                denoising_meta_values=denoising_meta_values,
            )
            result["loss"] = loss
            result["loss_dict"] = loss_dict

        return result
