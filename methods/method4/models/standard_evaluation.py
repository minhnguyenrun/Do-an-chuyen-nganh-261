"""Module đánh giá chung (Standard Evaluation) dùng chung logic với src/common/metrics.py.

Đầu ra định dạng chuẩn của toàn bộ dự án:
- Strict Evaluation: Yêu cầu đúng IoU >= threshold VÀ đúng Class.
- Loose Evaluation: Chỉ đánh giá vị trí Bounding Box (Class-agnostic).
- COCO mAP@[.50:.95], mAP50, mAP75.
- Class-level Breakdown: Precision, Recall, F1 theo từng nhãn.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np

from src.common.metrics import (
    evaluate,
    evaluate_boxes,
    coco_map,
    coco_box_map,
)


def run_standard_evaluation(
    gt_coco: dict,
    base_predictions: list[dict],
    fusion_predictions: list[dict],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """
    So sánh toàn diện giữa Base Model và Fusion Model bằng code chuẩn của src/common/metrics.py.
    
    gt_coco: {"images": [...], "annotations": [...], "categories": [...]}
    base_predictions / fusion_predictions: list các dict {"image_id", "category_id", "bbox": [x, y, w, h], "score"}
    """
    cat_names = {c["id"]: c["name"] for c in gt_coco.get("categories", [])}

    # Gom annotations theo image_id
    gt_by_image = {}
    for ann in gt_coco.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in gt_by_image:
            gt_by_image[img_id] = []
        gt_by_image[img_id].append({
            "category_id": ann["category_id"],
            "bbox": ann["bbox"],
        })

    # Gom predictions theo image_id
    def _group_preds(preds_list):
        grouped = {}
        for p in preds_list:
            img_id = p["image_id"]
            if img_id not in grouped:
                grouped[img_id] = []
            grouped[img_id].append({
                "category_id": p["category_id"],
                "bbox": p["bbox"],
                "score": p.get("score", 1.0),
            })
        return grouped

    base_by_img = _group_preds(base_predictions)
    fusion_by_img = _group_preds(fusion_predictions)

    # 1. Strict Evaluation (IoU >= 0.5 + Same Category)
    strict_base = evaluate(gt_by_image, base_by_img, cat_names, iou_threshold=iou_threshold)
    strict_fusion = evaluate(gt_by_image, fusion_by_img, cat_names, iou_threshold=iou_threshold)

    # 2. Loose Evaluation (Localization only, class-agnostic)
    loose_base = evaluate_boxes(gt_by_image, base_by_img, iou_threshold=iou_threshold)
    loose_fusion = evaluate_boxes(gt_by_image, fusion_by_img, iou_threshold=iou_threshold)

    # 3. COCO mAP (nếu có thư viện pycocotools)
    coco_metrics_base = {}
    coco_metrics_fusion = {}
    try:
        coco_metrics_base = coco_map(gt_coco, base_predictions)
        coco_metrics_fusion = coco_map(gt_coco, fusion_predictions)
    except Exception as e:
        print(f"[Warning] Không thể tính COCO mAP: {e}")

    return {
        "strict": {
            "base": strict_base,
            "fusion": strict_fusion,
        },
        "loose": {
            "base": loose_base,
            "fusion": loose_fusion,
        },
        "coco_map": {
            "base": coco_metrics_base.get("overall", {}),
            "fusion": coco_metrics_fusion.get("overall", {}),
        },
    }
