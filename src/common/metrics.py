"""Tinh Precision/Recall/F1 cho bai toan Document Layout Analysis bang cach
so khop bounding box du doan voi ground truth theo IoU.

Dung chung cho moi method (docling, paddleocr_vl, ...): moi method chi can
xuat du doan dung dinh dang {"category_id", "bbox": [x, y, w, h]} la co the
danh gia bang module nay.
"""

from collections import defaultdict
import contextlib
import io

import numpy as np


def iou(box_a: list, box_b: list) -> float:
    """IoU giua 2 box dang [x, y, w, h]."""
    ax1, ay1, ax2, ay2 = box_a[0], box_a[1], box_a[0] + box_a[2], box_a[1] + box_a[3]
    bx1, by1, bx2, by2 = box_b[0], box_b[1], box_b[0] + box_b[2], box_b[1] + box_b[3]

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def match_image(gt_boxes: list, pred_boxes: list, iou_threshold: float = 0.5,
                 require_same_category: bool = True):
    """Ghep du doan voi ground truth cho MOT anh bang greedy-IoU.

    Tra ve (matches, unmatched_gt_idx, unmatched_pred_idx):
      - matches: list (gt_idx, pred_idx, iou)
      - unmatched_gt_idx / unmatched_pred_idx: index con lai khong duoc ghep
    """
    candidates = []
    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            if require_same_category and gt["category_id"] != pred["category_id"]:
                continue
            score = iou(gt["bbox"], pred["bbox"])
            if score >= iou_threshold:
                candidates.append((score, gi, pi))

    candidates.sort(key=lambda c: c[0], reverse=True)

    matched_gt, matched_pred = set(), set()
    matches = []
    for score, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        matches.append((gi, pi, score))

    unmatched_gt = [i for i in range(len(gt_boxes)) if i not in matched_gt]
    unmatched_pred = [i for i in range(len(pred_boxes)) if i not in matched_pred]
    return matches, unmatched_gt, unmatched_pred


def evaluate(gt_by_image: dict, pred_by_image: dict, category_names: dict,
             iou_threshold: float = 0.5, require_same_category: bool = True) -> dict:
    """gt_by_image / pred_by_image: {image_id: [{"category_id", "bbox"}, ...]}.

    Tra ve dict {"overall": {...}, "per_category": {category_id: {...}}}
    voi precision/recall/f1/support/tp/fp/fn.
    """
    per_cat_tp = defaultdict(int)
    per_cat_fp = defaultdict(int)
    per_cat_fn = defaultdict(int)
    iou_sum, iou_count = 0.0, 0

    # Duyet hop 2 tap ID: du doan tren anh khong co trong GT van bi tinh FP,
    # thay vi bi bo qua am tham. Noi goi can tu loc pred_by_image ve dung tap
    # anh can danh gia (vd. cung split) truoc khi goi ham nay.
    for image_id in set(gt_by_image) | set(pred_by_image):
        gt_boxes = gt_by_image.get(image_id, [])
        pred_boxes = pred_by_image.get(image_id, [])
        matches, unmatched_gt, unmatched_pred = match_image(
            gt_boxes, pred_boxes, iou_threshold, require_same_category)

        for gi, pi, score in matches:
            cat_id = gt_boxes[gi]["category_id"]
            per_cat_tp[cat_id] += 1
            iou_sum += score
            iou_count += 1
        for gi in unmatched_gt:
            per_cat_fn[gt_boxes[gi]["category_id"]] += 1
        for pi in unmatched_pred:
            per_cat_fp[pred_boxes[pi]["category_id"]] += 1

    all_cats = set(per_cat_tp) | set(per_cat_fp) | set(per_cat_fn)
    per_category = {}
    for cat_id in sorted(all_cats, key=lambda c: (c is None, c)):
        tp, fp, fn = per_cat_tp[cat_id], per_cat_fp[cat_id], per_cat_fn[cat_id]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_category[cat_id] = {
            "name": category_names.get(cat_id, str(cat_id)),
            "tp": tp, "fp": fp, "fn": fn, "support": tp + fn,
            "precision": precision, "recall": recall, "f1": f1,
        }

    total_tp = sum(per_cat_tp.values())
    total_fp = sum(per_cat_fp.values())
    total_fn = sum(per_cat_fn.values())
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "overall": {
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "precision": precision, "recall": recall, "f1": f1,
            "mean_iou_matched": iou_sum / iou_count if iou_count > 0 else 0.0,
        },
        "per_category": per_category,
    }


def coco_map(gt_coco: dict, predictions: list, iou_type: str = "bbox") -> dict:
    """Tinh mAP/AR kieu COCO bang pycocotools.

    gt_coco: dict COCO ({"images", "annotations", "categories"}), chi nen chua
        cac anh da co du doan.
    predictions: list {"image_id", "category_id", "bbox": [x, y, w, h], "score"}.
        Box co category_id khong thuoc gt_coco["categories"] phai loai truoc.

    Tra ve dict {"overall": {...}, "per_category": {category_id: {"name", "ap", "ap50"}}}.
    """

    gt_coco = dict(gt_coco)
    gt_coco["annotations"] = [
        {**ann, "iscrowd": ann.get("iscrowd", 0),
         "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3])}
        for ann in gt_coco["annotations"]
    ]

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        raise ImportError(
            "Cần cài đặt thư viện 'pycocotools' để tính mAP. Vui lòng cài đặt: pip install pycocotools"
        )

    # pycocotools in rat nhieu log, gom lai de khong lan voi bao cao.
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        coco_gt = COCO()
        coco_gt.dataset = gt_coco
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(predictions) if predictions else COCO()
        coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    keys = ["map", "map50", "map75", "map_small", "map_medium", "map_large",
            "ar1", "ar10", "ar100", "ar_small", "ar_medium", "ar_large"]
    overall = {k: float(v) for k, v in zip(keys, coco_eval.stats)}

    # precision: [T iou thresholds, R recall points, K categories, A areas, M maxDets]
    precision = coco_eval.eval["precision"]
    names = {c["id"]: c["name"] for c in gt_coco["categories"]}
    per_category = {}
    for k, cat_id in enumerate(coco_eval.params.catIds):
        p_all = precision[:, :, k, 0, -1]
        p_50 = precision[0, :, k, 0, -1]
        per_category[cat_id] = {
            "name": names.get(cat_id, str(cat_id)),
            "ap": float(np.mean(p_all[p_all > -1])) if (p_all > -1).any() else -1.0,
            "ap50": float(np.mean(p_50[p_50 > -1])) if (p_50 > -1).any() else -1.0,
        }
    return {"overall": overall, "per_category": per_category}


def evaluate_boxes(
    gt_by_image: dict,
    pred_by_image: dict,
    iou_threshold: float = 0.5,
) -> dict:
    """Danh gia chat luong Bounding Box thuan tuy (Class-agnostic / Localization-only),
    hoan toan bo qua nhan class.

    gt_by_image / pred_by_image: {image_id: [{"bbox": [x, y, w, h], ...}, ...] hoac [[x, y, w, h], ...]}.

    Tra ve dict:
        total_gt, total_pred, tp, fp, fn, precision, recall, f1, mean_iou_matched, iou_threshold
    """
    total_tp = 0
    total_gt = 0
    total_pred = 0
    iou_sum = 0.0
    iou_count = 0

    all_image_ids = set(gt_by_image) | set(pred_by_image)
    for image_id in all_image_ids:
        raw_gts = gt_by_image.get(image_id, [])
        raw_preds = pred_by_image.get(image_id, [])

        gt_boxes = [{"bbox": b["bbox"] if isinstance(b, dict) else b} for b in raw_gts]
        pred_boxes = [{"bbox": b["bbox"] if isinstance(b, dict) else b} for b in raw_preds]

        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)

        matches, _, _ = match_image(
            gt_boxes, pred_boxes, iou_threshold=iou_threshold, require_same_category=False
        )
        total_tp += len(matches)
        for _, _, score in matches:
            iou_sum += score
            iou_count += 1

    total_fp = total_pred - total_tp
    total_fn = total_gt - total_tp
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = iou_sum / iou_count if iou_count > 0 else 0.0

    return {
        "iou_threshold": iou_threshold,
        "total_gt": total_gt,
        "total_pred": total_pred,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou_matched": mean_iou,
    }


def evaluate_boxes_multi_iou(
    gt_by_image: dict,
    pred_by_image: dict,
    iou_thresholds: list[float] | None = None,
) -> list[dict]:
    """Danh gia Bounding Box tren danh sach nhieu nguong IoU (mac dinh: 0.50 den 0.95 buoc 0.05)."""
    if iou_thresholds is None:
        iou_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    return [
        evaluate_boxes(gt_by_image, pred_by_image, iou_threshold=thresh)
        for thresh in iou_thresholds
    ]


def coco_box_map(gt_coco: dict, predictions: list, iou_type: str = "bbox") -> dict:
    """Tinh COCO mAP thuan tuy cho Bounding Box (Class-agnostic Box mAP), bo qua class.

    Quy doi toan bo ground truth va predictions ve mot class duy nhat: ID=1 ('BBox').
    """
    dummy_gt_coco = dict(gt_coco)
    dummy_gt_coco["categories"] = [{"id": 1, "name": "BBox"}]
    dummy_gt_coco["annotations"] = [
        {**ann, "category_id": 1} for ann in dummy_gt_coco.get("annotations", [])
    ]
    dummy_predictions = [
        {**pred, "category_id": 1} for pred in predictions
    ]
    return coco_map(dummy_gt_coco, dummy_predictions, iou_type=iou_type)

