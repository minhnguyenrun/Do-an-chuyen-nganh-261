"""Evaluator for Method 7 using standard COCO mAP@[0.50:0.95].

Chuẩn COCO chính thức:
- Top-100 detections per image, không lọc theo confidence threshold.
- pycocotools đánh giá trên GT gốc.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from methods.method7.common import (
    NUM_DLN_CLASSES,
    coco_eval,
    to_coco_dets,
    topk_detections,
)


@torch.no_grad()
def evaluate_method7_coco_map(
    model: torch.nn.Module,
    dataloader: DataLoader,
    coco_gt: dict,
    device: torch.device,
    max_batches: int | None = None,
) -> dict:
    model.eval()
    dets = []
    image_ids = []

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        pixel_values, layout_heatmaps, targets = batch
        pixel_values = pixel_values.to(device, non_blocking=True)
        layout_heatmaps = layout_heatmaps.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16):
            outputs = model(pixel_values, layout_heatmaps)

        logits = outputs["logits"]
        pred_boxes = outputs["pred_boxes"]
        probs = torch.sigmoid(logits)

        bs = pixel_values.size(0)
        for i in range(bs):
            tgt = targets[i]
            img_id = tgt["image_id"]
            h_orig, w_orig = tgt["orig_size"].tolist()
            image_ids.append(img_id)

            sc, cl, bx = topk_detections(probs[i], pred_boxes[i], k=100)
            dets += to_coco_dets(img_id, sc.cpu(), cl.cpu(), bx.cpu(), w_orig, h_orig)

    if not dets:
        return {"map": 0.0, "map50": 0.0, "map75": 0.0, "per_category": {}}

    return coco_eval(coco_gt, dets, image_ids=image_ids)
