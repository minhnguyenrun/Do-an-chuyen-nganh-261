"""Evaluator for Method 6 using standard COCO mAP@[0.50:0.95]."""

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


@torch.no_grad()
def evaluate_method6_coco_map(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    conf_thresh: float = 0.2,
    max_batches: int | None = None,
) -> dict:
    model.eval()

    gt_images = []
    gt_annotations = []
    coco_predictions = []
    ann_id_counter = 1

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        pixel_values = batch["pixel_values"].to(device)
        layout_heatmaps = batch["layout_heatmaps"].to(device)
        targets = batch["targets"]

        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            outputs = model(pixel_values, layout_heatmaps)

        logits = outputs["logits"]
        pred_boxes = outputs["pred_boxes"]

        probs = torch.sigmoid(logits)

        bs = pixel_values.size(0)
        for i in range(bs):
            tgt = targets[i]
            img_id = tgt["image_id"]
            h_orig, w_orig = tgt["orig_size"].tolist()

            gt_images.append({
                "id": img_id,
                "width": int(w_orig),
                "height": int(h_orig),
            })

            gt_b = tgt["boxes"]
            gt_c = tgt["class_labels"]
            for b_idx in range(len(gt_b)):
                cx, cy, w, h = gt_b[b_idx].tolist()
                x1 = (cx - w / 2.0) * w_orig
                y1 = (cy - h / 2.0) * h_orig
                bw = w * w_orig
                bh = h * h_orig
                heron_cid = int(gt_c[b_idx].item())
                doclaynet_cid = HERON_TO_DOCLAYNET_ID.get(heron_cid, 10)

                gt_annotations.append({
                    "id": ann_id_counter,
                    "image_id": img_id,
                    "category_id": doclaynet_cid,
                    "bbox": [x1, y1, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                })
                ann_id_counter += 1

            p_probs = probs[i]
            p_boxes = pred_boxes[i]

            max_scores, max_cids = p_probs[:, :11].max(dim=-1)
            keep_mask = max_scores >= conf_thresh

            p_scores_filtered = max_scores[keep_mask]
            p_cids_filtered = max_cids[keep_mask]
            p_boxes_filtered = p_boxes[keep_mask]

            for d_idx in range(len(p_scores_filtered)):
                score = float(p_scores_filtered[d_idx].item())
                heron_cid = int(p_cids_filtered[d_idx].item())
                doclaynet_cid = HERON_TO_DOCLAYNET_ID.get(heron_cid, 10)

                cx, cy, w, h = p_boxes_filtered[d_idx].tolist()
                x1 = (cx - w / 2.0) * w_orig
                y1 = (cy - h / 2.0) * h_orig
                bw = w * w_orig
                bh = h * h_orig

                coco_predictions.append({
                    "image_id": img_id,
                    "category_id": doclaynet_cid,
                    "bbox": [x1, y1, bw, bh],
                    "score": score,
                })

    if not coco_predictions or not gt_annotations:
        return {"map": 0.0, "map50": 0.0, "map75": 0.0, "per_category": {}}

    gt_coco = {
        "images": gt_images,
        "annotations": gt_annotations,
        "categories": DOCLAYNET_CATEGORIES,
    }

    try:
        res = coco_map(gt_coco, coco_predictions)
        return {
            "map": res["overall"]["map"],
            "map50": res["overall"]["map50"],
            "map75": res["overall"]["map75"],
            "per_category": res["per_category"],
        }
    except Exception as e:
        print(f"[evaluate_coco_map] Warning: {e}")
        return {"map": 0.0, "map50": 0.0, "map75": 0.0, "per_category": {}}
