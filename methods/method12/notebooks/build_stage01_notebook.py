"""Build Stage 01 Notebook: Lightweight Frozen-Backbone Detector on DocLayNet 11 classes."""

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent / "01_stage01_light_detector.ipynb"


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.splitlines()],
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.splitlines()],
    }


cells = [
    md_cell(
        """# Method 12 — Stage 01: Lightweight Detector (Frozen Backbone) trên DocLayNet 11 lớp

**Mục tiêu Stage 01:**
- Sử dụng model nhẹ hơn RT-DETR-R50vd (`docling-project/docling-layout-egret-medium` ~19.5M params, hoặc tùy chọn `PekingU/rtdetr_v2_r18vd` ~20M params).
- Đóng băng hoàn toàn Backbone, chỉ fine-tune Encoder + Decoder + Classification Head để khớp với **11 lớp DocLayNet**.
- Đánh giá chuẩn xác trên **Val (6,489 trang)** và **Test (4,999 trang)** (COCO mAP, Strict/Loose F1, FPS, Peak VRAM) và lưu checkpoint **`weights/best_stage01.pt`** làm nền tảng cho Stage 02."""
    ),
    code_cell(
        """import os
import sys
import json
import time
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

REPO_ROOT = Path(r"d:\\Semester_261\\DoAnChuyenNganh\\DO-AN-261")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from methods.method12.common import (
    TRAIN_ANNO, VAL_ANNO, TEST_ANNO,
    TRAIN_IMG_640, VAL_IMG_640, TEST_IMG_640,
    TRAIN_IMG_SRC, VAL_IMG_SRC, TEST_IMG_SRC,
    WEIGHTS_DIR, REPORT_DIR, CATEGORIES, load_coco,
)
from methods.method12.models.stage01_detector import Stage01LightDetector
from methods.method12.dataset import DocLayNet640Dataset, collate_fn_det, evaluate_stage01_model

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Cấu hình huấn luyện Stage 01
MODEL_ID = "docling-project/docling-layout-egret-medium"  # hoặc "PekingU/rtdetr_v2_r18vd"
FREEZE_BACKBONE = True
BATCH_SIZE = 12
GRAD_ACCUM = 2          # Effective batch size = 24
EPOCHS = 12
LR = 1e-4
MIN_LR = 5e-6
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
RESUME = True"""
    ),
    code_cell(
        """print("Loading DocLayNet COCO annotations...")
train_coco = load_coco(TRAIN_ANNO)
val_coco = load_coco(VAL_ANNO)
test_coco = load_coco(TEST_ANNO)

print(f"Train: {len(train_coco['images']):,} pages")
print(f"Val:   {len(val_coco['images']):,} pages")
print(f"Test:  {len(test_coco['images']):,} pages")

train_ds = DocLayNet640Dataset(train_coco, TRAIN_IMG_640, TRAIN_IMG_SRC, augment=True)
val_ds = DocLayNet640Dataset(val_coco, VAL_IMG_640, VAL_IMG_SRC, augment=False)
test_ds = DocLayNet640Dataset(test_coco, TEST_IMG_640, TEST_IMG_SRC, augment=False)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0), collate_fn=collate_fn_det,
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0), collate_fn=collate_fn_det,
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0), collate_fn=collate_fn_det,
)"""
    ),
    code_cell(
        """model = Stage01LightDetector(model_id=MODEL_ID, freeze_backbone=FREEZE_BACKBONE).to(device)
p_info = model.count_parameters()
print(f"Model: {MODEL_ID}")
print(f"Params -> Total: {p_info['total']:,} | Trainable: {p_info['trainable']:,} | Frozen: {p_info['frozen']:,}")

trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)

steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM)
total_steps = steps_per_epoch * EPOCHS
warmup_steps = max(1, int(total_steps * WARMUP_RATIO))

def lr_lambda(step: int) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_ratio = MIN_LR / LR
    return min_ratio + (1.0 - min_ratio) * cosine

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

ckpt_last = WEIGHTS_DIR / "last_stage01.pt"
ckpt_best = WEIGHTS_DIR / "best_stage01.pt"
history_path = REPORT_DIR / "stage01_history.json"

start_epoch = 1
best_val_map = -1.0
history = []

if RESUME and ckpt_last.exists():
    ckpt = torch.load(ckpt_last, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = ckpt["epoch"] + 1
    best_val_map = ckpt.get("best_val_map", -1.0)
    history = ckpt.get("history", [])
    print(f"Resumed from epoch {ckpt['epoch']} | best_val_mAP={best_val_map:.4f}")"""
    ),
    code_cell(
        """# Baseline Zero-shot Eval trước khi fine-tune (Epoch 0)
if start_epoch == 1 and not history:
    print("Evaluating Zero-shot (Epoch 0) on Val...")
    ep0 = evaluate_stage01_model(model, val_loader, val_coco, device, desc="Epoch 0 Val")
    print(
        f"[Epoch 0] Val mAP={ep0['mAP']:.4f} | AP50={ep0['AP50']:.4f} | "
        f"Strict-F1={ep0['f1']['strict_micro_f1']:.4f} | FPS={ep0['fps']:.1f}"
    )
    history.append({"epoch": 0, "val": ep0})

for epoch in range(start_epoch, EPOCHS + 1):
    t_ep = time.time()
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    n_batches = 0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Stage01 Epoch {epoch}/{EPOCHS}")
    for step_idx, (pixs, labels) in pbar:
        pixs = pixs.to(device, non_blocking=True)
        labels_dev = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in labels]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            out = model(pixs, labels=labels_dev)
            loss = out["loss"] / GRAD_ACCUM

        loss.backward()
        running_loss += float(out["loss"].item())
        n_batches += 1

        if (step_idx + 1) % GRAD_ACCUM == 0 or (step_idx + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.1)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if step_idx % 100 == 0:
            pbar.set_postfix({"loss": f"{running_loss / max(n_batches, 1):.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

    avg_loss = running_loss / max(n_batches, 1)
    peak_vram_gb = (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else 0.0

    val_m = evaluate_stage01_model(model, val_loader, val_coco, device, desc=f"Val Epoch {epoch}")
    ep_sec = time.time() - t_ep

    print(
        f"[Epoch {epoch:02d}] loss={avg_loss:.4f} | val_mAP={val_m['mAP']:.4f} | "
        f"AP50={val_m['AP50']:.4f} | Strict-F1={val_m['f1']['strict_micro_f1']:.4f} | "
        f"FPS={val_m['fps']:.1f} | VRAM={peak_vram_gb:.2f}GB | time={ep_sec/60:.1f}m"
    )

    history.append({
        "epoch": epoch,
        "train_loss": round(avg_loss, 5),
        "lr": scheduler.get_last_lr()[0],
        "peak_vram_gb": round(peak_vram_gb, 2),
        "epoch_time_sec": round(ep_sec, 1),
        "val": val_m,
    })

    if val_m["mAP"] > best_val_map:
        best_val_map = val_m["mAP"]
        torch.save({"model": model.state_dict(), "model_id": MODEL_ID, "epoch": epoch, "val_mAP": best_val_map}, ckpt_best)
        print(f"  -> Saved NEW BEST Stage 01 checkpoint: {ckpt_best} (mAP={best_val_map:.4f})")

    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_map": best_val_map,
        "history": history,
    }, ckpt_last)

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)"""
    ),
    code_cell(
        """# Đánh giá cuối cùng trên Val & Test bằng Best Checkpoint
best_ckpt = torch.load(ckpt_best, map_location="cpu", weights_only=False)
model.load_state_dict(best_ckpt["model"])
model.to(device)

val_final = evaluate_stage01_model(model, val_loader, val_coco, device, desc="Final Val Eval")
test_final = evaluate_stage01_model(model, test_loader, test_coco, device, desc="Final Test Eval")

summary = {
    "stage": "01_light_detector_frozen_backbone",
    "model_id": MODEL_ID,
    "best_epoch": best_ckpt["epoch"],
    "params": model.count_parameters(),
    "val": val_final,
    "test": test_final,
}
with open(REPORT_DIR / "stage01_eval_results.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("=== STAGE 01 FINAL RESULTS ===")
print(f"Val  -> mAP: {val_final['mAP']:.4f} | AP50: {val_final['AP50']:.4f} | Strict F1: {val_final['f1']['strict_micro_f1']:.4f} | FPS: {val_final['fps']:.1f}")
print(f"Test -> mAP: {test_final['mAP']:.4f} | AP50: {test_final['AP50']:.4f} | Strict F1: {test_final['f1']['strict_micro_f1']:.4f} | FPS: {test_final['fps']:.1f}")"""
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
print(f"Generated notebook: {NB_PATH}")
