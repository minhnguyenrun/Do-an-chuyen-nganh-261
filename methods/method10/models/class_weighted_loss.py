"""Class-Weighted Varifocal Loss for RT-DETRv2 on DocLayNet.
Khắc phục triệt để hiện tượng mất cân bằng lớp (Long-Tail Class Imbalance).
Cấp trọng số cao cho các lớp hiếm (Title, Caption, Footnote, Section-header, Header/Footer)
để ngăn chặn triệt để hiện tượng Catastrophic Forgetting.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.loss.loss_rt_detr import RTDetrLoss, _set_aux_loss, box_iou
from transformers.image_transforms import center_to_corners_format


# Heron Class IDs:
# 0: Caption (DocLayNet 1) -> 3.0
# 1: Footnote (DocLayNet 2) -> 4.0
# 2: Formula (DocLayNet 3) -> 4.5
# 3: List-item (DocLayNet 4) -> 1.8
# 4: Page-footer (DocLayNet 5) -> 1.8
# 5: Page-header (DocLayNet 6) -> 2.0
# 6: Picture (DocLayNet 9) -> 1.6
# 7: Section-header (DocLayNet 7) -> 2.2
# 8: Table (DocLayNet 8) -> 1.2
# 9: Text (DocLayNet 10) -> 1.0
# 10: Title (DocLayNet 11) -> 4.5

DEFAULT_HERON_CLASS_WEIGHTS = {
    0: 3.0,   # Caption
    1: 4.0,   # Footnote
    2: 4.5,   # Formula
    3: 1.8,   # List-item
    4: 1.8,   # Page-footer
    5: 2.0,   # Page-header
    6: 1.6,   # Picture
    7: 2.2,   # Section-header
    8: 1.2,   # Table
    9: 1.0,   # Text (dominant class)
    10: 4.5,  # Title (rare class)
}


class ClassWeightedRTDetrLoss(RTDetrLoss):
    def __init__(self, config, class_weights_dict: dict[int, float] | None = None):
        super().__init__(config)
        weights_dict = class_weights_dict or DEFAULT_HERON_CLASS_WEIGHTS

        # Tạo tensor class_weights shape [1, 1, num_classes]
        weights_tensor = torch.ones(1, 1, self.num_classes, dtype=torch.float32)
        for c_id, w in weights_dict.items():
            if c_id < self.num_classes:
                weights_tensor[0, 0, c_id] = w

        self.register_buffer("class_weights", weights_tensor)

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):
        if "pred_boxes" not in outputs:
            raise KeyError("No predicted boxes found in outputs")
        if "logits" not in outputs:
            raise KeyError("No predicted logits found in outputs")
        idx = self._get_source_permutation_idx(indices)

        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([_target["boxes"][i] for _target, (_, i) in zip(targets, indices)], dim=0)
        ious, _ = box_iou(center_to_corners_format(src_boxes.detach()), center_to_corners_format(target_boxes))
        ious = torch.diag(ious)

        src_logits = outputs["logits"]
        dtype = src_logits.dtype
        target_classes_original = torch.cat([_target["class_labels"][i] for _target, (_, i) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_original
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_original = torch.zeros_like(target_classes, dtype=dtype)
        target_score_original[idx] = ious.to(dtype)
        target_score = target_score_original.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits.detach())
        weight = (self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score).to(dtype)

        # Áp dụng Class Weights để phạt nặng các lớp hiếm
        cls_w = self.class_weights.to(device=src_logits.device, dtype=dtype)
        weight = weight * cls_w

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction="none")
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_vfl": loss}


def compute_class_weighted_rtdetr_loss(
    criterion: ClassWeightedRTDetrLoss,
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
