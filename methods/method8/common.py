"""Method 8 - Tiện ích dùng chung và Tiền xử lý Tối ưu (Fast 640 SSD + Base RT-DETRv2 Prior).

Cải tiến của Method 8:
1. Pre-resize Offline toàn bộ ảnh về kích thước 640x640 (JPEG quality 95) lưu sẵn trên SSD.
   -> Giải phóng hoàn toàn nghẽn CPU I/O & Decoding khi train, tăng tốc train gấp 2-3 lần.
2. Hỗ trợ cơ chế Incremental Processing (xử lý cuốn chiếu):
   -> Đang tải được bao nhiêu trang thì resize và chạy Base RT-DETRv2 suy luận bấy nhiêu.
   -> Khi tải thêm data mới, hệ thống tự động nhận diện và xử lý bù mà không chạy lại từ đầu.
3. Heatmap 22 kênh (11 Book Prior + 11 Page Prior) sinh từ dự đoán của Base Heron RT-DETRv2
   (loại trừ trang mục tiêu, score >= 0.3, chuẩn hóa [0,1], float16 memmap).
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import contextlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT_DIR = Path(__file__).resolve().parents[2]

# Đường dẫn dữ liệu gốc
TRAIN_ANNO = ROOT_DIR / "datasets" / "doclaynet" / "processed" / "annotations" / "train.json"
TRAIN_IMG_SRC = ROOT_DIR / "datasets" / "doclaynet" / "processed" / "images"
VAL_ANNO = ROOT_DIR / "datasets" / "doclaynet_val" / "processed" / "annotations" / "validation.json"
VAL_IMG_SRC = ROOT_DIR / "datasets" / "doclaynet_val" / "processed" / "images"
TEST_ANNO = ROOT_DIR / "datasets" / "doclaynet_test" / "processed" / "annotations" / "test.json"
TEST_IMG_SRC = ROOT_DIR / "datasets" / "doclaynet_test" / "processed" / "images"

# Đường dẫn Method 8
M8_DIR = ROOT_DIR / "methods" / "method8"
CACHE_DIR = Path("C:/doclaynet_cache/method8")
WEIGHTS_DIR = M8_DIR / "weights"
REPORT_DIR = M8_DIR / "report"
DATA_640_DIR = ROOT_DIR / "datasets" / "doclaynet_640"
TRAIN_IMG_640 = DATA_640_DIR / "train"
VAL_IMG_640 = DATA_640_DIR / "val"
TEST_IMG_640 = DATA_640_DIR / "test"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_IMG_640.mkdir(parents=True, exist_ok=True)
VAL_IMG_640.mkdir(parents=True, exist_ok=True)
TEST_IMG_640.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "docling-project/docling-layout-heron"
IMAGE_SIZE = 640
HEATMAP_SIZE = 80
NUM_DLN_CLASSES = 11          # DocLayNet 11 lớp = Heron id 0..10
PRIOR_SCORE_THRESH = 0.3      # ngưỡng box của Base dùng để vẽ heatmap
WINDOW_SIZE = 64
VAL_NUM_BOOKS = 20
SPLIT_SEED = 2026

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

# Bảng màu trực quan hóa 11 lớp
CAT_COLORS = {
    1: "#1f77b4",   # Caption (Blue)
    2: "#aec7e8",   # Footnote (Light blue)
    3: "#ff7f0e",   # Formula (Orange)
    4: "#2ca02c",   # List-item (Green)
    5: "#d62728",   # Page-footer (Red)
    6: "#9467bd",   # Page-header (Purple)
    7: "#8c564b",   # Picture (Brown)
    8: "#e377c2",   # Section-header (Pink)
    9: "#7f7f7f",   # Table (Gray)
    10: "#bcbd22",  # Text (Olive yellow)
    11: "#17becf",  # Title (Cyan)
}


def load_coco(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for a in data.get("annotations", []):
        if "iscrowd" not in a:
            a["iscrowd"] = 0
    return data


def book_of(img: dict) -> str:
    return img.get("original_filename") or "_".join(img["file_name"].split("_")[:-1])


def split_train_val_books(train_coco: dict) -> Tuple[Set[str], Set[str]]:
    """Tách VAL_NUM_BOOKS cuốn từ tập train làm val độc lập theo seed cố định."""
    books = sorted({book_of(im) for im in train_coco["images"]})
    rng = np.random.RandomState(SPLIT_SEED)
    val = set(rng.choice(books, size=VAL_NUM_BOOKS, replace=False).tolist())
    return set(books) - val, val


def resize_single_image(src_path: Path, dst_path: Path, quality: int = 95) -> bool:
    """Resize 1 ảnh về kích thước IMAGE_SIZE x IMAGE_SIZE và lưu dạng JPEG."""
    if dst_path.exists():
        return False
    try:
        with Image.open(src_path) as im:
            im_rgb = im.convert("RGB")
            im_resized = im_rgb.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR)
            im_resized.save(dst_path, format="JPEG", quality=quality)
        return True
    except Exception as e:
        print(f"Lỗi resize ảnh {src_path.name}: {e}")
        return False


def batch_resize_images(
    file_list: List[str],
    src_dir: Path,
    dst_dir: Path,
    max_workers: int = 8,
    quality: int = 95,
) -> Tuple[int, int]:
    """Resize đa luồng một danh sách tên file. Trả về (số ảnh mới resize, số ảnh đã có sẵn)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    already_done = 0
    for fname in file_list:
        src = src_dir / fname
        stem = Path(fname).stem
        dst = dst_dir / f"{stem}.jpg"
        if dst.exists():
            already_done += 1
        elif src.exists():
            todo.append((src, dst))

    if not todo:
        return 0, already_done

    def _worker(item):
        s, d = item
        return resize_single_image(s, d, quality=quality)

    new_done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for res in executor.map(_worker, todo):
            if res:
                new_done += 1

    return new_done, already_done


def load_image_640_tensor(img_640_path: Path) -> torch.Tensor:
    """Nạp nhanh ảnh đã pre-resize 640x640 trực tiếp từ SSD thành Tensor float32 [3, 640, 640]."""
    with Image.open(img_640_path) as img:
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def load_raw_image_tensor(path: Path) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Hàm dự phòng nếu ảnh chưa được pre-resize: đọc ảnh gốc và resize trên CPU."""
    with Image.open(path) as img:
        img_rgb = img.convert("RGB")
        w, h = img_rgb.size
        img_res = img_rgb.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR)
        arr = np.asarray(img_res, dtype=np.float32) / 255.0
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
            return {"map": 0.0, "map50": 0.0, "map75": 0.0, "per_category": {}}
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
