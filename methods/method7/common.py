"""Method 7 - Tiện ích dùng chung.

Khác biệt cốt lõi so với Method 6 (sửa lỗi rò rỉ nhãn):
- Heatmap (Page Prior + Book Prior) được vẽ từ DỰ ĐOÁN của Base RT-DETRv2 (docling-layout-heron),
  KHÔNG dùng nhãn GT. GT chỉ dùng cho loss và để chấm điểm.
- Tiền xử lý ảnh đúng theo preprocessor_config của Heron: resize 640x640 (bilinear),
  rescale /255, KHÔNG normalize ImageNet (do_normalize=false).
- Val tách từ các cuốn sách của tập train (không lấy từ test).
- COCO mAP chấm bằng pycocotools chuẩn: top-100 dự đoán / ảnh, không lọc ngưỡng.
"""

from __future__ import annotations

import io
import json
import contextlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[2]

TRAIN_ANNO = ROOT_DIR / "datasets" / "doclaynet" / "processed" / "annotations" / "train.json"
TRAIN_IMG = ROOT_DIR / "datasets" / "doclaynet" / "processed" / "images"
TEST_ANNO = ROOT_DIR / "datasets" / "doclaynet_test" / "processed" / "annotations" / "test.json"
TEST_IMG = ROOT_DIR / "datasets" / "doclaynet_test" / "processed" / "images"

M7_DIR = ROOT_DIR / "methods" / "method7"
CACHE_DIR = M7_DIR / "cache"
WEIGHTS_DIR = M7_DIR / "weights"
REPORT_DIR = M7_DIR / "report"

BASE_MODEL = "docling-project/docling-layout-heron"
IMAGE_SIZE = 640
HEATMAP_SIZE = 80
NUM_DLN_CLASSES = 11          # DocLayNet 11 lớp = Heron id 0..10
PRIOR_SCORE_THRESH = 0.3      # ngưỡng box của Base dùng để vẽ heatmap
WINDOW_SIZE = 64
VAL_NUM_BOOKS = 20
SPLIT_SEED = 2026

CATEGORIES = [
    {"id": 1, "name": "Caption"}, {"id": 2, "name": "Footnote"}, {"id": 3, "name": "Formula"},
    {"id": 4, "name": "List-item"}, {"id": 5, "name": "Page-footer"}, {"id": 6, "name": "Page-header"},
    {"id": 7, "name": "Picture"}, {"id": 8, "name": "Section-header"}, {"id": 9, "name": "Table"},
    {"id": 10, "name": "Text"}, {"id": 11, "name": "Title"},
]
CAT_NAMES = {c["id"]: c["name"] for c in CATEGORIES}
# Heron id (0..10) <-> DocLayNet category_id (1..11)


def load_coco(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def book_of(img: dict) -> str:
    return img.get("original_filename") or "_".join(img["file_name"].split("_")[:-1])


def split_train_val_books(train_coco: dict) -> tuple[set[str], set[str]]:
    """Tách VAL_NUM_BOOKS cuốn từ tập train làm val (cố định theo seed)."""
    books = sorted({book_of(im) for im in train_coco["images"]})
    rng = np.random.RandomState(SPLIT_SEED)
    val = set(rng.choice(books, size=VAL_NUM_BOOKS, replace=False).tolist())
    return set(books) - val, val


def load_image_tensor(path: Path) -> tuple[torch.Tensor, tuple[int, int]]:
    """Tiền xử lý đúng Heron: resize 640 bilinear, /255, không normalize."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous(), (h, w)


def render_pred_heatmap(preds: np.ndarray, grid: int = HEATMAP_SIZE) -> np.ndarray:
    """preds: [N, 6] = (heron_cls, score, cx, cy, w, h) chuẩn hóa [0,1].
    Trả về [11, grid, grid], mỗi ô = max score của box phủ ô đó."""
    hm = np.zeros((NUM_DLN_CLASSES, grid, grid), dtype=np.float32)
    for c, s, cx, cy, w, h in preds:
        c = int(c)
        if c >= NUM_DLN_CLASSES:
            continue
        x1 = int(np.clip((cx - w / 2) * grid, 0, grid - 1))
        y1 = int(np.clip((cy - h / 2) * grid, 0, grid - 1))
        x2 = int(np.clip(np.ceil((cx + w / 2) * grid), x1 + 1, grid))
        y2 = int(np.clip(np.ceil((cy + h / 2) * grid), y1 + 1, grid))
        region = hm[c, y1:y2, x1:x2]
        np.maximum(region, s, out=region)
    return hm


def topk_detections(probs: torch.Tensor, boxes: torch.Tensor, k: int = 100):
    """Hậu xử lý chuẩn RT-DETR: top-k trên (query x class) phẳng, chỉ 11 lớp DocLayNet."""
    p = probs[:, :NUM_DLN_CLASSES]
    q, c = p.shape
    scores, idx = p.reshape(-1).topk(min(k, q * c))
    return scores, idx % c, boxes[idx // c]


def to_coco_dets(image_id, scores, classes, boxes, w, h) -> list[dict]:
    out = []
    for s, c, b in zip(scores.tolist(), classes.tolist(), boxes.tolist()):
        cx, cy, bw, bh = b
        out.append({
            "image_id": image_id,
            "category_id": int(c) + 1,
            "bbox": [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h],
            "score": float(s),
        })
    return out


def coco_eval(gt_coco: dict, dets: list[dict], image_ids: list[int] | None = None) -> dict:
    """Chấm COCO mAP chuẩn bằng pycocotools trên GT gốc trong file json."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ids = set(image_ids) if image_ids is not None else {im["id"] for im in gt_coco["images"]}
    gt = {
        "info": {}, "licenses": [],
        "images": [im for im in gt_coco["images"] if im["id"] in ids],
        "categories": gt_coco["categories"],
        "annotations": [],
    }
    for i, a in enumerate(x for x in gt_coco["annotations"] if x["image_id"] in ids):
        a = dict(a)
        a.setdefault("iscrowd", 0)
        a.setdefault("area", a["bbox"][2] * a["bbox"][3])
        a["id"] = i + 1
        gt["annotations"].append(a)
    dets = [d for d in dets if d["image_id"] in ids]
    if not dets:
        return {"map": 0.0, "map50": 0.0, "map75": 0.0, "per_category": {}}

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dets)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    s = ev.stats
    per_cat = {}
    prec = ev.eval["precision"]  # [T, R, K, A, M]
    for k, cid in enumerate(ev.params.catIds):
        p = prec[:, :, k, 0, -1]
        p50 = prec[0, :, k, 0, -1]
        per_cat[CAT_NAMES[cid]] = {
            "ap": float(p[p > -1].mean()) if (p > -1).any() else 0.0,
            "ap50": float(p50[p50 > -1].mean()) if (p50 > -1).any() else 0.0,
        }
    return {
        "map": float(s[0]), "map50": float(s[1]), "map75": float(s[2]),
        "map_small": float(s[3]), "map_medium": float(s[4]), "map_large": float(s[5]),
        "ar1": float(s[6]), "ar10": float(s[7]), "ar100": float(s[8]),
        "per_category": per_cat,
    }
