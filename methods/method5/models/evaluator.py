"""Evaluator for Method 5 using standard COCO mAP@[0.50:0.95].

Tính toán:
- mAP@[.50:.95] (COCO standard)
- mAP@50, mAP@75
- Per-category AP@[.50:.95] và AP@50
Sử dụng logic chuẩn của src.common.metrics.coco_map (pycocotools).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.common.metrics import coco_map

DOCLAYNET_CATEGORIES = [
    {"id": 1, "name": "Caption"},
    {"id": 2, "name": "Footnote"},
    {"id": 3, "name": "Formula"},
    {"id": 4, "name": "List-item"},
    {"id": 5, "name": "Page-footer"},
    {"id": 6, "name": "Page-header"},
    {"id": 7, "name": "Picture"},
    {"id": 8, "name": "Section-header"},
    {"id": 9, "name": "Table"},
    {"id": 10, "name": "Text"},
    {"id": 11, "name": "Title"},
]

HERON_TO_DOCLAYNET_ID = {
    0: 1,   # Caption
    1: 2,   # Footnote
    2: 3,   # Formula
    3: 4,   # List-item
    4: 5,   # Page-footer
    5: 6,   # Page-header
    6: 7,   # Picture
    7: 8,   # Section-header
    8: 9,   # Table
    9: 10,  # Text
    10: 11, # Title
}


def evaluate_coco_map(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    conf_thresh: float = 0.2,
    use_fusion: bool = True,
    max_batches: int | None = None,
) -> dict:
    """
    Đánh giá mô hình theo chuẩn COCO mAP@[0.50:0.95].
    Returns:
        dict chứa:
            - 'map': mAP@[0.50:0.95]
            - 'map50': AP50
            - 'map75': AP75
            - 'per_category': dict AP từng nhãn
    """
    model.eval()

    gt_images = []
    gt_annotations = []
    predictions = []

    ann_id_counter = 1

    use_amp = (device.type == "cuda")
    with torch.no_grad():
        for b_idx, batch in enumerate(dataloader):
            if max_batches is not None and b_idx >= max_batches:
                break

            pixel_values = batch["pixel_values"].to(device)
            ctx_elements = batch["context_elements"]
            image_ids = batch["image_ids"]
            file_names = batch["file_names"]
            orig_sizes = batch["orig_sizes"]
            batch_annotations = batch["annotations"]

            bs = len(pixel_values)

            # 1. Forward with AMP
            with torch.amp.autocast("cuda", enabled=use_amp):
                if use_fusion and hasattr(model, "fusion"):
                    out = model(pixel_values, context_elements=ctx_elements)
                    logits = out["logits"]
                    pred_boxes = out["pred_boxes"]
                elif hasattr(model, "rtdetr"):
                    out = model.rtdetr(pixel_values=pixel_values)
                    logits = out.logits
                    pred_boxes = out.pred_boxes
                else:
                    out = model(pixel_values=pixel_values)
                    logits = out.logits
                    pred_boxes = out.pred_boxes

            # Sigmoid probabilities cho RT-DETRv2
            probs = torch.sigmoid(logits)

            for i in range(bs):
                img_id = image_ids[i]
                orig_w, orig_h = orig_sizes[i]

                # Thu thập Ground Truth COCO
                gt_images.append({
                    "id": img_id,
                    "file_name": file_names[i],
                    "width": orig_w,
                    "height": orig_h,
                })

                for ann in batch_annotations[i]:
                    gt_annotations.append({
                        "id": ann.get("id", ann_id_counter),
                        "image_id": img_id,
                        "category_id": ann["category_id"],
                        "bbox": ann["bbox"],  # [x, y, w, h] pixel
                        "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                        "iscrowd": ann.get("iscrowd", 0),
                    })
                    ann_id_counter += 1

                # Thu thập Predictions
                sample_probs = probs[i]       # [300, 17]
                sample_boxes = pred_boxes[i]   # [300, 4] normalized [cx, cy, w, h]

                max_scores, pred_classes = sample_probs.max(dim=-1)
                valid_mask = (max_scores >= conf_thresh)

                v_scores = max_scores[valid_mask]
                v_classes = pred_classes[valid_mask]
                v_boxes = sample_boxes[valid_mask]

                for s_val, c_val, b_val in zip(v_scores, v_classes, v_boxes):
                    heron_cls = int(c_val.item())
                    if heron_cls not in HERON_TO_DOCLAYNET_ID:
                        continue
                    doclaynet_cat_id = HERON_TO_DOCLAYNET_ID[heron_cls]
                    score = float(s_val.item())

                    cx, cy, bw, bh = b_val.tolist()
                    xmin = max(0.0, (cx - bw / 2.0) * orig_w)
                    ymin = max(0.0, (cy - bh / 2.0) * orig_h)
                    box_w = min(orig_w - xmin, bw * orig_w)
                    box_h = min(orig_h - ymin, bh * orig_h)

                    predictions.append({
                        "image_id": img_id,
                        "category_id": doclaynet_cat_id,
                        "bbox": [round(xmin, 2), round(ymin, 2), round(box_w, 2), round(box_h, 2)],
                        "score": round(score, 4),
                    })

    # Xây dựng GT COCO dict
    gt_coco = {
        "images": gt_images,
        "annotations": gt_annotations,
        "categories": DOCLAYNET_CATEGORIES,
    }

    if not gt_annotations:
        return {"map": 0.0, "map50": 0.0, "map75": 0.0, "per_category": {}}

    res = coco_map(gt_coco, predictions)
    overall = res.get("overall", {})
    return {
        "map": overall.get("map", 0.0),
        "map50": overall.get("map50", 0.0),
        "map75": overall.get("map75", 0.0),
        "per_category": res.get("per_category", {}),
    }
