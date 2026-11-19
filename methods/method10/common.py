"""Method 10 - Tiện ích dùng chung (DocLayNet 640 SSD + Frozen-Backbone Book Context Cache).

Cache lưu trên ổ C (C:/doclaynet_cache/method10) để tránh đầy ổ D.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT_DIR = Path(__file__).resolve().parents[2]

# Đường dẫn dữ liệu gốc DocLayNet
TRAIN_ANNO = ROOT_DIR / "datasets" / "doclaynet" / "processed" / "annotations" / "train.json"
TRAIN_IMG_SRC = ROOT_DIR / "datasets" / "doclaynet" / "processed" / "images"
VAL_ANNO = ROOT_DIR / "datasets" / "doclaynet_val" / "processed" / "annotations" / "validation.json"
VAL_IMG_SRC = ROOT_DIR / "datasets" / "doclaynet_val" / "processed" / "images"
TEST_ANNO = ROOT_DIR / "datasets" / "doclaynet_test" / "processed" / "annotations" / "test.json"
TEST_IMG_SRC = ROOT_DIR / "datasets" / "doclaynet_test" / "processed" / "images"

# Ảnh 640x640 đã pre-resize trên SSD
DATA_640_DIR = ROOT_DIR / "datasets" / "doclaynet_640"
TRAIN_IMG_640 = DATA_640_DIR / "train"
VAL_IMG_640 = DATA_640_DIR / "val"
TEST_IMG_640 = DATA_640_DIR / "test"

# Đường dẫn Method 10
M10_DIR = ROOT_DIR / "methods" / "method10"
CACHE_DIR = Path("C:/doclaynet_cache/method10")
WEIGHTS_DIR = M10_DIR / "weights"
REPORT_DIR = M10_DIR / "report"

for _d in [CACHE_DIR, WEIGHTS_DIR, REPORT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "docling-project/docling-layout-heron"
IMAGE_SIZE = 640
NUM_DLN_CLASSES = 11

# Kích thước 3 tầng feature map sau encoder_input_proj của RT-DETRv2
FEAT_SHAPES: Tuple[Tuple[int, int, int], ...] = (
    (256, 80, 80),
    (256, 40, 40),
    (256, 20, 20),
)
TOTAL_FEAT_ELEMS = sum(c * h * w for c, h, w in FEAT_SHAPES)  # 2,150,400 floats (~4.3 MB ở float16)

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


def load_coco(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for a in data.get("annotations", []):
        if "iscrowd" not in a:
            a["iscrowd"] = 0
    return data


def book_of(img: dict) -> str:
    return img.get("original_filename") or "_".join(img["file_name"].split("_")[:-1])


def load_image_640_tensor(img_640_path: Path, raw_fallback_path: Optional[Path] = None) -> torch.Tensor:
    """Nạp ảnh 640x640 dạng float32 [3, 640, 640] trong [0, 1]."""
    if img_640_path.exists():
        with Image.open(img_640_path) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    if raw_fallback_path is not None and raw_fallback_path.exists():
        with Image.open(raw_fallback_path) as img:
            img_res = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR)
            arr = np.asarray(img_res, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)


def flatten_feats_to_fp16(feats: List[torch.Tensor]) -> np.ndarray:
    """Nối [B, 256, 80, 80], [B, 256, 40, 40], [B, 256, 20, 20] -> [B, 2150400] float16."""
    parts = [f.reshape(f.size(0), -1) for f in feats]
    return torch.cat(parts, dim=1).to(torch.float16).cpu().numpy()


def unflatten_feats(flat: torch.Tensor) -> List[torch.Tensor]:
    """Tách tensor [B, 2150400] thành list 3 feature maps [P3, P4, P5]."""
    bs = flat.size(0)
    out = []
    offset = 0
    for c, h, w in FEAT_SHAPES:
        n = c * h * w
        out.append(flat[:, offset : offset + n].view(bs, c, h, w))
        offset += n
    return out


def topk_detections(probs: torch.Tensor, boxes: torch.Tensor, k: int = 100):
    """Hậu xử lý chuẩn RT-DETR: top-k trên (query x class) phẳng, chỉ 11 lớp DocLayNet."""
    p = probs[:, :NUM_DLN_CLASSES]
    q, c = p.shape
    topk_v, topk_idx = torch.topk(p.reshape(-1), min(k, q * c))
    q_idx = topk_idx // c
    c_idx = topk_idx % c
    top_boxes = boxes[q_idx]
    return c_idx, topk_v, top_boxes


def to_coco_dets(image_id: int, classes, scores, boxes, orig_h: int, orig_w: int) -> list[dict]:
    """Chuyển top-k sang format COCO detections: bbox=[x,y,w,h] theo pixel gốc, category_id 1..11."""
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


def coco_eval(results: list[dict], coco_gt: dict, categories: list[dict] | None = None) -> dict:
    """Đánh giá COCO mAP chuẩn bằng pycocotools."""
    for a in coco_gt.get("annotations", []):
        if "iscrowd" not in a:
            a["iscrowd"] = 0
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt_obj = COCO()
        coco_gt_obj.dataset = coco_gt
        coco_gt_obj.createIndex()
        if not results:
            return {
                "map": 0.0, "map50": 0.0, "map75": 0.0,
                "map_small": 0.0, "map_medium": 0.0, "map_large": 0.0,
                "ar1": 0.0, "ar10": 0.0, "ar100": 0.0, "per_category": {},
            }
        coco_dt_obj = coco_gt_obj.loadRes(results)
        ev = COCOeval(coco_gt_obj, coco_dt_obj, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    out = {
        "map": float(ev.stats[0]),
        "map50": float(ev.stats[1]),
        "map75": float(ev.stats[2]),
        "map_small": float(ev.stats[3]),
        "map_medium": float(ev.stats[4]),
        "map_large": float(ev.stats[5]),
        "ar1": float(ev.stats[6]),
        "ar10": float(ev.stats[7]),
        "ar100": float(ev.stats[8]),
        "per_category": {},
    }
    cats = categories or coco_gt.get("categories", CATEGORIES)
    for cat in cats:
        cid = cat["id"]
        cname = cat["name"]
        with contextlib.redirect_stdout(io.StringIO()):
            ev_c = COCOeval(coco_gt_obj, coco_dt_obj, iouType="bbox")
            ev_c.params.catIds = [cid]
            ev_c.evaluate()
            ev_c.accumulate()
            ev_c.summarize()
        out["per_category"][cname] = {
            "ap": float(ev_c.stats[0]) if ev_c.stats[0] >= 0 else 0.0,
            "ap50": float(ev_c.stats[1]) if ev_c.stats[1] >= 0 else 0.0,
        }
    return out
