"""Build Stage 04 Notebook: Full End-to-End Live Pipeline (OCR -> DINO 3-5p -> YOLOv26 90%) + Comparison."""

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent / "04_stage04_end2end_eval_compare.ipynb"


def md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in source.splitlines()]}


def code_cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in source.splitlines()]}


cells = [
    md_cell(
        """# Method 12 — Stage 04: Pipeline Toàn Cục Live trên `Val` & So Sánh Tổng Hợp (`01` vs `02` vs `04`)

**Luồng chạy live end-to-end trên `Val` (6,489 trang, 160 sách):**
1. **Bước 1 (`T_ocr`):** Chạy OCR cả sách (xuất boxes + text + `OCR Grid 8×80×80`).
2. **Bước 2 (`T_dino_anchor`):** Chạy `DINO + M2Doc` trên **3–5 trang/sách** (tái sử dụng OCR Bước 1).
3. **Bước 3 (`T_yolo_infer`):** Chạy **YOLOv26 (`best_stage03_2.pt`)** trên ~90% trang còn lại kết hợp Early + Late + OCR Fusion.
4. **Bước 4:** Tổng hợp `mAP`, `AP50`, `AP75`, `Strict/Loose F1`, `FPS`, `ms/page`, `Peak VRAM` so sánh trực tiếp **Stage 01**, **Stage 02 (GT Oracle)**, và **Stage 04 (90% & Overall Book)**."""
    ),
    code_cell(
        """import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

REPO_ROOT = Path(r"d:\\Semester_261\\DoAnChuyenNganh\\DO-AN-261")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from methods.method12.common import (
    VAL_ANNO, VAL_IMG_640, VAL_IMG_SRC,
    HEATMAP_SIZE, ROOT_DIR, WEIGHTS_DIR, REPORT_DIR,
    coco_eval, compute_strict_loose_f1, load_coco, select_anchor_pages,
)
from methods.method12.ocr_utils import render_ocr_grid
from methods.method12.dino_anchor_runner import run_dino_m2doc_on_anchors
from methods.method12.models.stage02_yolo_fusion import Stage02YOLOFusionDetector
from methods.method12.yolo_dataset import Method12YOLOFusionDataset, collate_fn_yolo_fusion, evaluate_yolo_model

SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STAGE03_CKPT = WEIGHTS_DIR / "best_stage03_2.pt"

val_coco = load_coco(VAL_ANNO)
val_anchors_ids = select_anchor_pages(val_coco["images"], min_pages=3, max_pages=5, ratio=0.10, seed=SEED)
val_anchor_set = {pid for pids in val_anchors_ids.values() for pid in pids}
val_rem_coco = {
    "images": [im for im in val_coco["images"] if im["id"] not in val_anchor_set],
    "annotations": [a for a in val_coco["annotations"] if a["image_id"] not in val_anchor_set],
    "categories": val_coco["categories"],
}"""
    ),
    code_cell(
        """# === BƯỚC 1: Chạy OCR toàn bộ tập Val (đo T_ocr) ===
if str(ROOT_DIR / "methods" / "dino_m2doc" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "methods" / "dino_m2doc" / "src"))
from ocr_provider import OCRProvider

t_ocr_start = time.time()
ocr = OCRProvider(dataset="doclaynet_val")
val_ocr_raw = {}
val_ocr_grids = {}

for im in tqdm(val_coco["images"], desc="Step 1: Live OCR on Val"):
    iid = im["id"]
    w, h = float(im.get("width", 1025)), float(im.get("height", 1025))
    praw = VAL_IMG_SRC / im["file_name"]
    p640 = VAL_IMG_640 / f"{Path(im['file_name']).stem}.jpg"
    img_path = str(praw if praw.exists() else p640)

    boxes, texts = ocr.get_ocr_for_image(image_path=img_path, image_id=iid, target_size=(w, h), orig_size=(w, h))
    val_ocr_raw[iid] = (boxes, texts)
    val_ocr_grids[iid] = render_ocr_grid(boxes, texts, src_size=(w, h), grid=HEATMAP_SIZE).astype(np.float16)

t_ocr_sec = time.time() - t_ocr_start
print(f"[Step 1] OCR completed in {t_ocr_sec:.2f}s ({len(val_coco['images'])/t_ocr_sec:.1f} FPS)")"""
    ),
    code_cell(
        """# === BƯỚC 2: Chạy DINO+M2Doc trên 3-5 trang/sách của Val ===
t_dino_start = time.time()
val_dino_anchors, val_anchor_coco_dets = run_dino_m2doc_on_anchors(
    coco=val_coco,
    anchors_by_book=val_anchors_ids,
    dataset_name="doclaynet_val",
    img_raw_dir=VAL_IMG_SRC,
    img_640_dir=VAL_IMG_640,
    device=str(device),
    score_thresh=0.35,
    ocr_cache=val_ocr_raw,
)
t_dino_sec = time.time() - t_dino_start
print(f"[Step 2] DINO+M2Doc ran on {len(val_anchor_set):,} anchor pages in {t_dino_sec:.2f}s")"""
    ),
    code_cell(
        """# === BƯỚC 3: Chạy YOLOv26 (Stage 03.2) trên ~90% trang còn lại ===
model = Stage02YOLOFusionDetector(stage01_ckpt_path=None, freeze_backbone=True).to(device)
ckpt = torch.load(STAGE03_CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"], strict=True)

val_rem_ds = Method12YOLOFusionDataset(
    val_rem_coco, VAL_IMG_640, VAL_IMG_SRC,
    book_anchors=val_dino_anchors,
    ocr_grids=val_ocr_grids,
)
val_rem_loader = DataLoader(val_rem_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn_yolo_fusion)

if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()

val_rem_res = evaluate_yolo_model(model, val_rem_loader, val_rem_coco, device, is_fusion=True, desc="Step 3: Stage04 YOLOv26")
rem_dets = val_rem_res.pop("detections")
t_yolo_sec = val_rem_res["infer_time_sec"]
peak_vram_gb = (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else 0.0

all_book_dets = val_anchor_coco_dets + rem_dets
overall_coco = coco_eval(all_book_dets, val_coco)
overall_f1 = compute_strict_loose_f1(all_book_dets, val_coco, score_thresh=0.35, iou_thresh=0.50)

total_e2e_sec = t_ocr_sec + t_dino_sec + t_yolo_sec
overall_fps = len(val_coco["images"]) / max(total_e2e_sec, 1e-6)

stage04_summary = {
    "stage": "04_end2end_yolo_val",
    "timing": {
        "t_ocr_sec": round(t_ocr_sec, 2),
        "t_dino_anchor_sec": round(t_dino_sec, 2),
        "t_yolo_infer_sec": round(t_yolo_sec, 2),
        "total_e2e_sec": round(total_e2e_sec, 2),
        "overall_fps": round(overall_fps, 2),
        "ms_per_page": round(1000.0 * total_e2e_sec / len(val_coco["images"]), 2),
        "peak_vram_gb": round(peak_vram_gb, 2),
    },
    "remaining_90pct": val_rem_res,
    "overall_book": {
        "mAP": overall_coco.get("map", 0.0),
        "AP50": overall_coco.get("map50", 0.0),
        "AP75": overall_coco.get("map75", 0.0),
        "AP_small": overall_coco.get("map_small", 0.0),
        "AP_medium": overall_coco.get("map_medium", 0.0),
        "AP_large": overall_coco.get("map_large", 0.0),
        "per_category": overall_coco.get("per_category", {}),
        "f1": overall_f1,
    },
}
with open(REPORT_DIR / "stage04_eval_results.json", "w", encoding="utf-8") as f:
    json.dump(stage04_summary, f, indent=2)
print("Saved Stage 04 summary ->", REPORT_DIR / "stage04_eval_results.json")"""
    ),
    code_cell(
        """# === BƯỚC 4: Bảng so sánh Stage 01 vs Stage 02 vs Stage 04 ===
def load_json_safe(p):
    return json.load(open(p, "r", encoding="utf-8")) if p.exists() else None

s1 = load_json_safe(REPORT_DIR / "stage01_eval_results.json")
s2 = load_json_safe(REPORT_DIR / "stage02_eval_results.json")
s4 = stage04_summary

rows = []
if s1:
    v = s1["val"]
    rows.append(("Stage 01 (YOLO Single-Page)", v["mAP"], v["AP50"], v["AP75"], v["f1"]["strict_micro_f1"], v["f1"]["loose_micro_f1"], v["fps"]))
if s2:
    v = s2["val"]
    rows.append(("Stage 02 (GT Oracle + OCR)", v["mAP"], v["AP50"], v["AP75"], v["f1"]["strict_micro_f1"], v["f1"]["loose_micro_f1"], v["fps"]))
if s4:
    r90 = s4["remaining_90pct"]
    ov = s4["overall_book"]
    rows.append(("Stage 04 (90% YOLOv26)", r90["mAP"], r90["AP50"], r90["AP75"], r90["f1"]["strict_micro_f1"], r90["f1"]["loose_micro_f1"], r90["fps"]))
    rows.append(("Stage 04 (Overall Book)", ov["mAP"], ov["AP50"], ov["AP75"], ov["f1"]["strict_micro_f1"], ov["f1"]["loose_micro_f1"], s4["timing"]["overall_fps"]))

print(f"{'Pipeline':<28} | {'mAP':>6} | {'AP50':>6} | {'AP75':>6} | {'Strict-F1':>9} | {'Loose-F1':>8} | {'FPS':>6}")
print("-" * 86)
for name, m, a50, a75, sf1, lf1, fps in rows:
    print(f"{name:<28} | {m:>6.4f} | {a50:>6.4f} | {a75:>6.4f} | {sf1:>9.4f} | {lf1:>8.4f} | {fps:>6.1f}")

print("\\nTiming Breakdown (Val 6,489 pages):", json.dumps(s4["timing"], indent=2))"""
    ),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12.0"}}, "nbformat": 4, "nbformat_minor": 5}
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
print(f"Generated notebook: {NB_PATH}")
