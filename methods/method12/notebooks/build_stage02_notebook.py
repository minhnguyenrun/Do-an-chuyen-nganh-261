"""Build Stage 02 Notebook: YOLOv26 + Early (Heatmap + OCR) + Late (Anchor Cross-Attention) with GT."""

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent / "02_stage02_yolo_early_late_ocr_gt.ipynb"


def md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in source.splitlines()]}


def code_cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in source.splitlines()]}


cells = [
    md_cell(
        """# Method 12 — Stage 02: YOLOv26 + Early Fusion (Heatmap + OCR M2Doc-style) + Late Fusion (Anchor Cross-Attention)

**Mục tiêu Stage 02:**
- Nạp `weights/best_stage01.pt` (YOLOv26-DocLayNet).
- Cắm **Early Fusion** (`11-ch Book Heatmap` từ 3–5 trang GT Anchor/sách + `8-ch OCR Grid` M2Doc-style $\\rightarrow$ `19ch Sidecar CNN` vào P3/P4/P5).
- Cắm **Late Fusion** (`Anchor Cross-Attention` trên P5 trước Detect Head sử dụng trực tiếp GT Anchor BBoxes của sách).
- Loại bỏ hoàn toàn trang Anchor khỏi tập train/eval để **Zero Leakage**, lưu **`weights/best_stage02.pt`**."""
    ),
    code_cell(
        """import sys
import json
import time
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

REPO_ROOT = Path(r"d:\\Semester_261\\DoAnChuyenNganh\\DO-AN-261")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from methods.method12.common import (
    TRAIN_ANNO, VAL_ANNO, TEST_ANNO,
    TRAIN_IMG_640, VAL_IMG_640, TEST_IMG_640,
    TRAIN_IMG_SRC, VAL_IMG_SRC, TEST_IMG_SRC,
    CACHE_DIR, WEIGHTS_DIR, REPORT_DIR,
    load_coco, select_anchor_pages,
)
from methods.method12.models.stage01_yolo import gpu_photometric_augment
from methods.method12.models.stage02_yolo_fusion import Stage02YOLOFusionDetector
from methods.method12.yolo_dataset import (
    Method12YOLOFusionDataset,
    collate_fn_yolo_fusion,
    evaluate_yolo_model,
    extract_gt_anchors_by_book,
    precompute_ocr_grids,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

STAGE01_CKPT = WEIGHTS_DIR / "best_stage01.pt"
BATCH_SIZE = 16
GRAD_ACCUM = 1
EPOCHS = 6
LR_DETECTOR = 5e-5
LR_FUSION = 2e-4
MIN_LR = 1e-5
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
RESUME = True"""
    ),
    code_cell(
        """train_coco = load_coco(TRAIN_ANNO)
val_coco = load_coco(VAL_ANNO)
test_coco = load_coco(TEST_ANNO)

train_anchors_ids = select_anchor_pages(train_coco["images"], min_pages=3, max_pages=5, ratio=0.10, seed=SEED)
val_anchors_ids = select_anchor_pages(val_coco["images"], min_pages=3, max_pages=5, ratio=0.10, seed=SEED)
test_anchors_ids = select_anchor_pages(test_coco["images"], min_pages=3, max_pages=5, ratio=0.10, seed=SEED)

train_anchor_set = {pid for pids in train_anchors_ids.values() for pid in pids}
val_anchor_set = {pid for pids in val_anchors_ids.values() for pid in pids}
test_anchor_set = {pid for pids in test_anchors_ids.values() for pid in pids}

train_gt_anchors = extract_gt_anchors_by_book(train_coco, train_anchors_ids)
val_gt_anchors = extract_gt_anchors_by_book(val_coco, val_anchors_ids)
test_gt_anchors = extract_gt_anchors_by_book(test_coco, test_anchors_ids)

train_ocr_grids = precompute_ocr_grids(train_coco, "doclaynet", TRAIN_IMG_SRC, TRAIN_IMG_640, CACHE_DIR / "ocr_grids_train.pt")
val_ocr_grids = precompute_ocr_grids(val_coco, "doclaynet_val", VAL_IMG_SRC, VAL_IMG_640, CACHE_DIR / "ocr_grids_val.pt")
test_ocr_grids = precompute_ocr_grids(test_coco, "doclaynet_test", TEST_IMG_SRC, TEST_IMG_640, CACHE_DIR / "ocr_grids_test.pt")

def filter_coco_excluding(coco: dict, exclude_ids: set) -> dict:
    return {
        "images": [im for im in coco["images"] if im["id"] not in exclude_ids],
        "annotations": [a for a in coco["annotations"] if a["image_id"] not in exclude_ids],
        "categories": coco["categories"],
    }

val_rem_coco = filter_coco_excluding(val_coco, val_anchor_set)
test_rem_coco = filter_coco_excluding(test_coco, test_anchor_set)

train_ds = Method12YOLOFusionDataset(train_coco, TRAIN_IMG_640, TRAIN_IMG_SRC, train_gt_anchors, exclude_image_ids=train_anchor_set, ocr_grids=train_ocr_grids)
val_ds = Method12YOLOFusionDataset(val_rem_coco, VAL_IMG_640, VAL_IMG_SRC, val_gt_anchors, ocr_grids=val_ocr_grids)
test_ds = Method12YOLOFusionDataset(test_rem_coco, TEST_IMG_640, TEST_IMG_SRC, test_gt_anchors, ocr_grids=test_ocr_grids)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=(NUM_WORKERS > 0), collate_fn=collate_fn_yolo_fusion)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=(NUM_WORKERS > 0), collate_fn=collate_fn_yolo_fusion)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=(NUM_WORKERS > 0), collate_fn=collate_fn_yolo_fusion)
print(f"Remaining -> Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")"""
    ),
    code_cell(
        """model = Stage02YOLOFusionDetector(
    stage01_ckpt_path=str(STAGE01_CKPT) if STAGE01_CKPT.exists() else None,
    freeze_backbone=True,
).to(device)
print("Params:", model.count_parameters())

fusion_params = list(model.early_sidecar.parameters()) + [model.early_gates] + list(model.late_fusion.parameters())
fusion_ids = {id(p) for p in fusion_params}
det_params = [p for p in model.parameters() if p.requires_grad and id(p) not in fusion_ids]

optimizer = torch.optim.AdamW([
    {"params": det_params, "lr": LR_DETECTOR, "weight_decay": WEIGHT_DECAY},
    {"params": fusion_params, "lr": LR_FUSION, "weight_decay": WEIGHT_DECAY},
])

steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM)
total_steps = steps_per_epoch * EPOCHS
warmup_steps = max(1, int(total_steps * WARMUP_RATIO))

def lr_lambda(step: int) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (MIN_LR / LR_FUSION) + (1.0 - (MIN_LR / LR_FUSION)) * cosine

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

ckpt_last = WEIGHTS_DIR / "last_stage02.pt"
ckpt_best = WEIGHTS_DIR / "best_stage02.pt"
history_path = REPORT_DIR / "stage02_history.json"

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
    print(f"Resumed Stage 02 from epoch {ckpt['epoch']} | best_val_mAP={best_val_map:.4f}")"""
    ),
    code_cell(
        """all_trainable = det_params + fusion_params
for epoch in range(start_epoch, EPOCHS + 1):
    t_ep = time.time()
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    n_batches = 0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Stage02 YOLO Epoch {epoch}/{EPOCHS}")
    for step_idx, (pixs, early_19, anchor_mem, anchor_mask, targets, _) in pbar:
        pixs = gpu_photometric_augment(pixs.to(device, non_blocking=True))
        early_19 = early_19.to(device, non_blocking=True)
        anchor_mem = anchor_mem.to(device, non_blocking=True)
        anchor_mask = anchor_mask.to(device, non_blocking=True)
        targets_dev = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            out = model(pixs, early_19, anchor_mem, anchor_mask, targets=targets_dev)
            loss = out["loss"] / GRAD_ACCUM

        loss.backward()
        running_loss += float(out["loss"].item())
        n_batches += 1

        if (step_idx + 1) % GRAD_ACCUM == 0 or (step_idx + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=10.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if step_idx % 200 == 0:
            pbar.set_postfix({"loss": f"{running_loss / max(n_batches, 1):.4f}"})

    avg_loss = running_loss / max(n_batches, 1)
    peak_vram_gb = (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else 0.0
    val_m = evaluate_yolo_model(model, val_loader, val_rem_coco, device, is_fusion=True, desc=f"Val Epoch {epoch}")
    val_m.pop("detections", None)
    ep_sec = time.time() - t_ep

    print(
        f"[Epoch {epoch:02d}] loss={avg_loss:.4f} | val_mAP={val_m['mAP']:.4f} | "
        f"AP50={val_m['AP50']:.4f} | Strict-F1={val_m['f1']['strict_micro_f1']:.4f} | "
        f"early={model.early_gates.detach().cpu().numpy().round(3).tolist()} | "
        f"late={float(model.late_fusion.gate.item()):.3f} | time={ep_sec/60:.1f}m"
    )

    history.append({"epoch": epoch, "train_loss": round(avg_loss, 5), "peak_vram_gb": round(peak_vram_gb, 2), "epoch_time_sec": round(ep_sec, 1), "val": val_m})
    if val_m["mAP"] > best_val_map:
        best_val_map = val_m["mAP"]
        torch.save({"model": model.state_dict(), "epoch": epoch, "val_mAP": best_val_map}, ckpt_best)
        print(f"  -> Saved NEW BEST Stage 02 YOLO checkpoint: {ckpt_best}")

    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch, "best_val_map": best_val_map, "history": history}, ckpt_last)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)"""
    ),
    code_cell(
        """best_ckpt = torch.load(ckpt_best, map_location="cpu", weights_only=False)
model.load_state_dict(best_ckpt["model"])
model.to(device)

val_final = evaluate_yolo_model(model, val_loader, val_rem_coco, device, is_fusion=True, desc="Final Val Eval")
val_final.pop("detections", None)
test_final = evaluate_yolo_model(model, test_loader, test_rem_coco, device, is_fusion=True, desc="Final Test Eval")
test_final.pop("detections", None)

summary = {"stage": "02_yolo26_early_late_ocr_gt_oracle", "best_epoch": best_ckpt["epoch"], "val": val_final, "test": test_final}
with open(REPORT_DIR / "stage02_eval_results.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("=== STAGE 02 (GT ORACLE) FINAL RESULTS ===")
print(f"Val  -> mAP: {val_final['mAP']:.4f} | AP50: {val_final['AP50']:.4f} | Strict-F1: {val_final['f1']['strict_micro_f1']:.4f} | FPS: {val_final['fps']:.1f}")
print(f"Test -> mAP: {test_final['mAP']:.4f} | AP50: {test_final['AP50']:.4f} | Strict-F1: {test_final['f1']['strict_micro_f1']:.4f} | FPS: {test_final['fps']:.1f}")"""
    ),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12.0"}}, "nbformat": 4, "nbformat_minor": 5}
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
print(f"Generated notebook: {NB_PATH}")
