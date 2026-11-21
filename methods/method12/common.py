"""Method 12 - Shared utilities, dataset paths, anchor sampling, COCO & F1 evaluation."""

from __future__ import annotations

import contextlib
import io
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT_DIR = Path(__file__).resolve().parents[2]

# DocLayNet annotations & raw images
TRAIN_ANNO = ROOT_DIR / "datasets" / "doclaynet" / "processed" / "annotations" / "train.json"
TRAIN_IMG_SRC = ROOT_DIR / "datasets" / "doclaynet" / "processed" / "images"
VAL_ANNO = ROOT_DIR / "datasets" / "doclaynet_val" / "processed" / "annotations" / "validation.json"
VAL_IMG_SRC = ROOT_DIR / "datasets" / "doclaynet_val" / "processed" / "images"
TEST_ANNO = ROOT_DIR / "datasets" / "doclaynet_test" / "processed" / "annotations" / "test.json"
TEST_IMG_SRC = ROOT_DIR / "datasets" / "doclaynet_test" / "processed" / "images"

# Pre-resized 640x640 images on SSD
DATA_640_DIR = ROOT_DIR / "datasets" / "doclaynet_640"
TRAIN_IMG_640 = DATA_640_DIR / "train"
VAL_IMG_640 = DATA_640_DIR / "val"
TEST_IMG_640 = DATA_640_DIR / "test"

# Method 12 directories
M12_DIR = ROOT_DIR / "methods" / "method12"
CACHE_DIR = Path("C:/doclaynet_cache/method12")
WEIGHTS_DIR = M12_DIR / "weights"
REPORT_DIR = M12_DIR / "report"
NOTEBOOKS_DIR = M12_DIR / "notebooks"

for _d in [CACHE_DIR, WEIGHTS_DIR, REPORT_DIR, NOTEBOOKS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 640
HEATMAP_SIZE = 80
NUM_DLN_CLASSES = 11

CATEGORIES = [
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
CAT_NAMES = {c["id"]: c["name"] for c in CATEGORIES}
NAME_TO_IDX = {c["name"]: c["id"] - 1 for c in CATEGORIES}


def load_coco(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for a in data.get("annotations", []):
        if "iscrowd" not in a:
            a["iscrowd"] = 0
    return data


def book_of(img: dict) -> str:
    return img.get("original_filename") or "_".join(img["file_name"].split("_")[:-1])


def select_anchor_pages(
    images: List[dict],
    min_pages: int = 3,
    max_pages: int = 5,
    ratio: float = 0.10,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """Stratified random sampling of 3-5 anchor page IDs per book.

    For small books (< 10 pages), caps anchors at max(1, min(min_pages, n // 2))
    so remaining pages always exist for training/evaluation.
    """
    rng = random.Random(seed)
    book_to_imgs: Dict[str, List[dict]] = defaultdict(list)
    for im in images:
        book_to_imgs[book_of(im)].append(im)

    anchors_by_book: Dict[str, List[int]] = {}
    for bname in sorted(book_to_imgs.keys()):
        pages = sorted(book_to_imgs[bname], key=lambda x: x.get("page_no", x["id"]))
        n = len(pages)
        if n <= 1:
            anchors_by_book[bname] = [pages[0]["id"]] if n == 1 else []
            continue

        if n < 10:
            k = max(1, min(min_pages, n // 2))
        else:
            k = max(min_pages, min(max_pages, round(ratio * n)))
            k = min(k, n - 1)

        # Stratified segments across the book
        chosen: List[int] = []
        for seg_idx in range(k):
            lo = (seg_idx * n) // k
            hi = ((seg_idx + 1) * n) // k
            hi = max(hi, lo + 1)
            pick = rng.randrange(lo, min(hi, n))
            chosen.append(pages[pick]["id"])
        anchors_by_book[bname] = chosen
    return anchors_by_book


def load_image_640_tensor(
    img_640_path: Path,
    raw_fallback_path: Optional[Path] = None,
    augment: bool = False,
) -> torch.Tensor:
    """Load 640x640 RGB float32 tensor in [0, 1]."""
    pil_img = None
    if img_640_path.exists():
        with Image.open(img_640_path) as img:
            pil_img = img.convert("RGB")
    elif raw_fallback_path is not None and raw_fallback_path.exists():
        with Image.open(raw_fallback_path) as img:
            pil_img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR)

    if pil_img is None:
        return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)

    if augment:
        if np.random.rand() < 0.5:
            pil_img = ImageEnhance.Brightness(pil_img).enhance(float(np.random.uniform(0.88, 1.12)))
        if np.random.rand() < 0.5:
            pil_img = ImageEnhance.Contrast(pil_img).enhance(float(np.random.uniform(0.88, 1.12)))

    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def topk_detections(probs: torch.Tensor, boxes: torch.Tensor, k: int = 100):
    """Top-k across (queries x 11 DocLayNet classes)."""
    p = probs[:, :NUM_DLN_CLASSES]
    q, c = p.shape
    topk_v, topk_idx = torch.topk(p.reshape(-1), min(k, q * c))
    q_idx = topk_idx // c
    c_idx = topk_idx % c
    return c_idx, topk_v, boxes[q_idx]


def to_coco_dets(image_id: int, classes, scores, boxes, orig_h: int, orig_w: int) -> list[dict]:
    """Convert normalized [cx, cy, w, h] detections to COCO [x, y, w, h] in original pixels."""
    dets = []
    for c, s, b in zip(classes.tolist(), scores.tolist(), boxes.tolist()):
        cx, cy, bw, bh = b
        x1 = max(0.0, (cx - bw / 2) * orig_w)
        y1 = max(0.0, (cy - bh / 2) * orig_h)
        w = max(0.0, bw * orig_w)
        h = max(0.0, bh * orig_h)
        dets.append({
            "image_id": int(image_id),
            "category_id": int(c) + 1,
            "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
            "score": round(float(s), 5),
        })
    return dets


def coco_eval(dets: list[dict], coco_gt_dict: dict) -> dict:
    """Run silent COCOeval and return standard summary + per-category AP."""
    if not dets:
        return {"map": 0.0, "map50": 0.0, "map75": 0.0, "per_category": {}}

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = coco_gt_dict
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dets)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    s = ev.stats
    per_cat = {}
    precisions = ev.eval["precision"]  # [T, R, K, A, M]
    for idx, cat in enumerate(CATEGORIES):
        p_all = precisions[:, :, idx, 0, 2]
        p_50 = precisions[0, :, idx, 0, 2]
        ap = float(np.mean(p_all[p_all > -1])) if np.any(p_all > -1) else 0.0
        ap50 = float(np.mean(p_50[p_50 > -1])) if np.any(p_50 > -1) else 0.0
        per_cat[cat["name"]] = {"ap": ap, "ap50": ap50}

    return {
        "map": float(s[0]),
        "map50": float(s[1]),
        "map75": float(s[2]),
        "map_small": float(s[3]),
        "map_medium": float(s[4]),
        "map_large": float(s[5]),
        "ar1": float(s[6]),
        "ar10": float(s[7]),
        "ar100": float(s[8]),
        "per_category": per_cat,
    }


def _bbox_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute IoU matrix [N, M] for [x, y, w, h] boxes."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    b1_x1, b1_y1 = boxes1[:, 0], boxes1[:, 1]
    b1_x2, b1_y2 = b1_x1 + boxes1[:, 2], b1_y1 + boxes1[:, 3]
    b2_x1, b2_y1 = boxes2[:, 0], boxes2[:, 1]
    b2_x2, b2_y2 = b2_x1 + boxes2[:, 2], b2_y1 + boxes2[:, 3]

    ix1 = np.maximum(b1_x1[:, None], b2_x1[None, :])
    iy1 = np.maximum(b1_y1[:, None], b2_y1[None, :])
    ix2 = np.minimum(b1_x2[:, None], b2_x2[None, :])
    iy2 = np.minimum(b1_y2[:, None], b2_y2[None, :])

    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area1 = np.maximum(0.0, boxes1[:, 2]) * np.maximum(0.0, boxes1[:, 3])
    area2 = np.maximum(0.0, boxes2[:, 2]) * np.maximum(0.0, boxes2[:, 3])
    union = area1[:, None] + area2[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-6), 0.0)


def compute_strict_loose_f1(
    dets: list[dict],
    coco_gt_dict: dict,
    score_thresh: float = 0.35,
    iou_thresh: float = 0.50,
) -> dict:
    """Compute Strict (class + IoU) and Loose (IoU only) Precision/Recall/F1."""
    gt_by_img = defaultdict(list)
    for a in coco_gt_dict.get("annotations", []):
        gt_by_img[a["image_id"]].append(a)

    dt_by_img = defaultdict(list)
    for d in dets:
        if d["score"] >= score_thresh:
            dt_by_img[d["image_id"]].append(d)

    strict_tp = defaultdict(int)
    strict_fp = defaultdict(int)
    strict_fn = defaultdict(int)
    loose_tp, loose_fp, loose_fn = 0, 0, 0

    for im in coco_gt_dict.get("images", []):
        iid = im["id"]
        gts = gt_by_img.get(iid, [])
        dts = sorted(dt_by_img.get(iid, []), key=lambda x: -x["score"])

        gt_boxes = np.array([g["bbox"] for g in gts], dtype=np.float32) if gts else np.zeros((0, 4), dtype=np.float32)
        dt_boxes = np.array([d["bbox"] for d in dts], dtype=np.float32) if dts else np.zeros((0, 4), dtype=np.float32)
        ious = _bbox_iou_matrix(dt_boxes, gt_boxes)

        # Strict matching (same class)
        matched_gt = set()
        for di, d in enumerate(dts):
            cid = d["category_id"]
            best_gi, best_iou = -1, iou_thresh
            for gi, g in enumerate(gts):
                if gi in matched_gt or g["category_id"] != cid:
                    continue
                if ious[di, gi] >= best_iou:
                    best_iou = float(ious[di, gi])
                    best_gi = gi
            if best_gi >= 0:
                matched_gt.add(best_gi)
                strict_tp[cid] += 1
            else:
                strict_fp[cid] += 1
        for gi, g in enumerate(gts):
            if gi not in matched_gt:
                strict_fn[g["category_id"]] += 1

        # Loose matching (ignore class)
        matched_gt_l = set()
        for di in range(len(dts)):
            best_gi, best_iou = -1, iou_thresh
            for gi in range(len(gts)):
                if gi in matched_gt_l:
                    continue
                if ious[di, gi] >= best_iou:
                    best_iou = float(ious[di, gi])
                    best_gi = gi
            if best_gi >= 0:
                matched_gt_l.add(best_gi)
                loose_tp += 1
            else:
                loose_fp += 1
        loose_fn += len(gts) - len(matched_gt_l)

    tot_tp = sum(strict_tp.values())
    tot_fp = sum(strict_fp.values())
    tot_fn = sum(strict_fn.values())
    micro_p = tot_tp / max(tot_tp + tot_fp, 1)
    micro_r = tot_tp / max(tot_tp + tot_fn, 1)
    micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-8)

    per_class = {}
    f1s = []
    for cat in CATEGORIES:
        cid = cat["id"]
        tp, fp, fn = strict_tp[cid], strict_fp[cid], strict_fn[cid]
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-8)
        per_class[cat["name"]] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}
        if tp + fn > 0:
            f1s.append(f1)

    loose_p = loose_tp / max(loose_tp + loose_fp, 1)
    loose_r = loose_tp / max(loose_tp + loose_fn, 1)
    loose_f1 = 2 * loose_p * loose_r / max(loose_p + loose_r, 1e-8)

    return {
        "score_thresh": score_thresh,
        "iou_thresh": iou_thresh,
        "strict_micro_p": round(micro_p, 4),
        "strict_micro_r": round(micro_r, 4),
        "strict_micro_f1": round(micro_f1, 4),
        "strict_macro_f1": round(float(np.mean(f1s)) if f1s else 0.0, 4),
        "loose_micro_p": round(loose_p, 4),
        "loose_micro_r": round(loose_r, 4),
        "loose_micro_f1": round(loose_f1, 4),
        "per_class": per_class,
    }
