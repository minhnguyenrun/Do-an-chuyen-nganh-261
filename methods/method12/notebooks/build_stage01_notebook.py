"""Build Stage 01 Notebook (Eval-Only Baseline): YOLOv26-DocLayNet Pretrained on Val & Test."""

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent / "01_stage01_yolo_baseline.ipynb"


def md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in source.splitlines()]}


def code_cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in source.splitlines()]}


cells = [
    md_cell(
        """# Method 12 — Stage 01 (Baseline Eval-Only): YOLOv26-DocLayNet

Trọng số `yolo26_doclaynet_pretrained.pt` đã được pretrain sẵn trên đúng **11 lớp DocLayNet**, nên **không cần train lại**.
Notebook này chỉ chạy **Eval 1 lần (~1–2 phút)** trên `Val` và `Test` để lưu mốc chuẩn `report/stage01_eval_results.json` phục vụ bảng so sánh ở Stage 04."""
    ),
    code_cell(
        """import sys
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(r"d:\\Semester_261\\DoAnChuyenNganh\\DO-AN-261")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from methods.method12.common import (
    VAL_ANNO, TEST_ANNO,
    VAL_IMG_640, TEST_IMG_640,
    VAL_IMG_SRC, TEST_IMG_SRC,
    REPORT_DIR, load_coco,
)
from methods.method12.models.stage01_yolo import Stage01YOLODetector
from methods.method12.yolo_dataset import Method12YOLODataset, collate_fn_yolo, evaluate_yolo_model

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)"""
    ),
    code_cell(
        """val_coco = load_coco(VAL_ANNO)
test_coco = load_coco(TEST_ANNO)

val_ds = Method12YOLODataset(val_coco, VAL_IMG_640, VAL_IMG_SRC)
test_ds = Method12YOLODataset(test_coco, TEST_IMG_640, TEST_IMG_SRC)

val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn_yolo)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn_yolo)

model = Stage01YOLODetector(freeze_backbone=True).to(device)
print("Params:", model.count_parameters())

val_final = evaluate_yolo_model(model, val_loader, val_coco, device, is_fusion=False, desc="Baseline Val Eval")
val_final.pop("detections", None)
test_final = evaluate_yolo_model(model, test_loader, test_coco, device, is_fusion=False, desc="Baseline Test Eval")
test_final.pop("detections", None)

summary = {
    "stage": "01_yolo26_pretrained_baseline",
    "params": model.count_parameters(),
    "val": val_final,
    "test": test_final,
}
out_path = REPORT_DIR / "stage01_eval_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("=== STAGE 01 (PRETRAINED YOLO BASELINE) ===")
print(f"Val  -> mAP: {val_final['mAP']:.4f} | AP50: {val_final['AP50']:.4f} | Strict-F1: {val_final['f1']['strict_micro_f1']:.4f} | FPS: {val_final['fps']:.1f}")
print(f"Test -> mAP: {test_final['mAP']:.4f} | AP50: {test_final['AP50']:.4f} | Strict-F1: {test_final['f1']['strict_micro_f1']:.4f} | FPS: {test_final['fps']:.1f}")
print("Saved ->", out_path)"""
    ),
]

nb = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12.0"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
print(f"Generated Eval-Only notebook: {NB_PATH}")
