import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent / "02_train_and_eval_method11_full.ipynb"


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
        """# Method 11 (Full Complete Pipeline): Residual Concat (512->256) + Objective Standard VFL

**Nâng cấp toàn diện từ Method 10:**
1. **Residual Concat Fusion (`page_feat + delta(cat([page_feat, book_ctx]))`)**: Ghép kênh thành 512 chiều rồi trộn có chọn lọc và cộng residual (khởi tạo `delta=0` để bảo toàn 100% đường truyền pretrained `docling-layout-heron`).
2. **Objective mAP-Aligned Loss**: Dùng `RTDetrLoss` chuẩn (VFL đồng đều `1.0` cho cả 11 lớp + L1 + GIoU), khắc phục hiện tượng `Title`/`Caption` triệt tiêu `Section-header`/`Page-header`/`Text`.
3. **Tái sử dụng Book Context Cache từ Method 10** (`C:/doclaynet_cache/method10/full_book_ctx_*.npy`, không cần chạy lại 1.5 giờ precompute).
4. **Chống Overfit & Tăng tốc**: `BookGroupedBatchSampler` + LRU RAM cache, Photometric Augmentation (Brightness/Contrast), `ctx_dropout = 0.1`, `float32` Leave-One-Out, `BATCH_SIZE = 8`, `GRAD_ACCUM = 2`, `NUM_WORKERS = 2`, bỏ `GradScaler` thừa khi dùng `bfloat16`.
5. **Đánh giá 100% chính xác**: Train (`69,375`), Val (`6,489` mỗi epoch bằng `COCOeval`), Test (`4,999` độc lập) + bảng so sánh với Base RT-DETRv2, Method 8, Method 10."""
    ),
    code_cell(
        """import os
import sys
import json
import time
import math
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

REPO_ROOT = Path(r"d:\\Semester_261\\DoAnChuyenNganh\\DO-AN-261")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from methods.method11.common import (
    TRAIN_ANNO, VAL_ANNO, TEST_ANNO,
    TRAIN_IMG_640, VAL_IMG_640, TEST_IMG_640,
    TRAIN_IMG_SRC, VAL_IMG_SRC, TEST_IMG_SRC,
    CACHE_DIR, WEIGHTS_DIR, REPORT_DIR,
    CATEGORIES, load_coco, book_of,
)
from methods.method11.models.rtdetr_method11 import ResidualConcatRTDetrMethod11
from methods.method11.dataset import (
    precompute_backbone_book_context,
    Method11FastDataset,
    BookGroupedBatchSampler,
    collate_fn_m11,
    evaluate_method11_coco_map,
)

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

c_free_gb = shutil.disk_usage("C:/").free / 1e9
print(f"C:/ free space: {c_free_gb:.2f} GB | Shared CACHE_DIR: {CACHE_DIR}")

# Hyperparameters (RTX 4070 8GB optimized + Book-Grouped Batching)
BATCH_SIZE = 8
GRAD_ACCUM = 2       # Effective batch size = 16
EPOCHS = 15
LR = 5e-5
FUSION_LR = 1e-4
MIN_LR = 5e-6
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 1e-4
CTX_DROPOUT = 0.1
NUM_WORKERS = 2      # Prefetch images + reuse book_ctx in worker RAM cache
RESUME = True        # Auto-resume if last_full_method11.pt exists"""
    ),
    code_cell(
        """print("Loading 100% DocLayNet COCO annotations (Train, Val, Test)...")
train_coco = load_coco(TRAIN_ANNO)
val_coco = load_coco(VAL_ANNO)
test_coco = load_coco(TEST_ANNO)

train_books = {book_of(img) for img in train_coco["images"]}
val_books = {book_of(img) for img in val_coco["images"]}
test_books = {book_of(img) for img in test_coco["images"]}

print(f"Train: {len(train_coco['images']):,} pages | {len(train_books):,} books | {len(train_coco['annotations']):,} boxes")
print(f"Val:   {len(val_coco['images']):,} pages | {len(val_books):,} books | {len(val_coco['annotations']):,} boxes")
print(f"Test:  {len(test_coco['images']):,} pages | {len(test_books):,} books | {len(test_coco['annotations']):,} boxes")

assert len(train_coco["images"]) == 69375, f"Unexpected train count: {len(train_coco['images'])}"
assert len(val_coco["images"]) == 6489, f"Unexpected val count: {len(val_coco['images'])}"
assert len(test_coco["images"]) == 4999, f"Unexpected test count: {len(test_coco['images'])}"
assert len(train_books & val_books) == 0, "Train/Val book leakage!"
assert len(train_books & test_books) == 0, "Train/Test book leakage!"
assert len(val_books & test_books) == 0, "Val/Test book leakage!"
print("Verified 100% dataset counts and zero book overlap across Train/Val/Test.")"""
    ),
    code_cell(
        """model = ResidualConcatRTDetrMethod11(ctx_dropout=CTX_DROPOUT).to(device)

bb_trainable = sum(p.numel() for p in model.rtdetr.model.backbone.parameters() if p.requires_grad)
ctx_proj_trainable = sum(p.numel() for p in model.frozen_ctx_proj.parameters() if p.requires_grad)
enc_trainable = sum(p.numel() for p in model.rtdetr.model.encoder.parameters() if p.requires_grad)
dec_trainable = sum(p.numel() for p in model.rtdetr.model.decoder.parameters() if p.requires_grad)
fusion_trainable = sum(p.numel() for p in model.fusion_block.parameters() if p.requires_grad)
total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print(f"Frozen backbone trainable params:        {bb_trainable:,} (expected 0)")
print(f"Frozen ctx proj trainable params:        {ctx_proj_trainable:,} (expected 0)")
print(f"Trainable encoder params:                {enc_trainable:,}")
print(f"Trainable decoder params:                {dec_trainable:,}")
print(f"Trainable ResidualConcatFusion params:   {fusion_trainable:,}")
print(f"Total trainable / Total params:          {total_trainable:,} / {total_params:,} ({100*total_trainable/total_params:.2f}%)")
assert bb_trainable == 0 and ctx_proj_trainable == 0"""
    ),
    code_cell(
        """# Tái sử dụng trực tiếp cache book_ctx từ Method 10 tại C:/doclaynet_cache/method10
train_mmap_path, train_book_idx = precompute_backbone_book_context(
    images_list=train_coco["images"],
    img_640_dir=TRAIN_IMG_640,
    img_raw_dir=TRAIN_IMG_SRC,
    model=model,
    split="train",
    batch_size=16,
    prefix="full_",
    device=device,
)

val_mmap_path, val_book_idx = precompute_backbone_book_context(
    images_list=val_coco["images"],
    img_640_dir=VAL_IMG_640,
    img_raw_dir=VAL_IMG_SRC,
    model=model,
    split="val",
    batch_size=16,
    prefix="full_",
    device=device,
)

test_mmap_path, test_book_idx = precompute_backbone_book_context(
    images_list=test_coco["images"],
    img_640_dir=TEST_IMG_640,
    img_raw_dir=TEST_IMG_SRC,
    model=model,
    split="test",
    batch_size=16,
    prefix="full_",
    device=device,
)

train_ds = Method11FastDataset(train_coco, TRAIN_IMG_640, TRAIN_IMG_SRC, train_mmap_path, train_book_idx, augment=True)
val_ds = Method11FastDataset(val_coco, VAL_IMG_640, VAL_IMG_SRC, val_mmap_path, val_book_idx, augment=False)
test_ds = Method11FastDataset(test_coco, TEST_IMG_640, TEST_IMG_SRC, test_mmap_path, test_book_idx, augment=False)

train_batch_sampler = BookGroupedBatchSampler(
    train_ds, batch_size=BATCH_SIZE, seed=SEED, shuffle_books=True, shuffle_pages_in_book=True
)
dl_extra = {"persistent_workers": True, "prefetch_factor": 2} if NUM_WORKERS > 0 else {}

train_loader = DataLoader(
    train_ds,
    batch_sampler=train_batch_sampler,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
    collate_fn=collate_fn_m11,
    **dl_extra,
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"), collate_fn=collate_fn_m11, **dl_extra,
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"), collate_fn=collate_fn_m11, **dl_extra,
)
print(f"Loaders ready (Book-Grouped) | Train: {len(train_loader)} batches | Val: {len(val_loader)} batches | Test: {len(test_loader)} batches")"""
    ),
    code_cell(
        """WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

best_ckpt_path = WEIGHTS_DIR / "best_full_method11.pt"
last_ckpt_path = WEIGHTS_DIR / "last_full_method11.pt"
history_path = REPORT_DIR / "full_history.json"

trainable_params = [p for p in model.parameters() if p.requires_grad]
param_groups = model.get_param_groups(lr=LR, fusion_lr=FUSION_LR, weight_decay=WEIGHT_DECAY)
optimizer = torch.optim.AdamW(param_groups)

steps_per_epoch = max(1, math.ceil(len(train_loader) / GRAD_ACCUM))
total_opt_steps = steps_per_epoch * EPOCHS
warmup_steps = max(1, int(total_opt_steps * WARMUP_RATIO))

def lr_lambda(step: int) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = float(step - warmup_steps) / float(max(1, total_opt_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_ratio = MIN_LR / LR
    return min_ratio + (1.0 - min_ratio) * cosine

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

start_epoch = 1
best_map = -1.0
history = []
ep0_metrics = None

if RESUME and last_ckpt_path.exists():
    print(f"Resuming from checkpoint: {last_ckpt_path}")
    ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    try:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    except Exception as exc:
        print(f"Note: reset optimizer/scheduler state on resume ({exc})")
    start_epoch = ckpt["epoch"] + 1
    best_map = ckpt.get("best_map", -1.0)
    history = ckpt.get("history", [])
    ep0_metrics = ckpt.get("epoch0", None)
    print(f"Resumed at epoch {start_epoch} (best_map={best_map:.4f})")

if ep0_metrics is None:
    print("Evaluating Epoch 0 (Before Fine-Tuning - verifying 100% pretrained preservation) on 100% Val (6,489 pages)...")
    ep0_metrics = evaluate_method11_coco_map(model, val_loader, val_coco, device, desc="Full Val Ep0 (6489)")
    print(
        f"Epoch 0 | Val mAP@[.50:.95]: {ep0_metrics['mAP']:.4f} | "
        f"AP50: {ep0_metrics['AP50']:.4f} | AP75: {ep0_metrics['AP75']:.4f} | "
        f"Time: {ep0_metrics['eval_time_sec']:.1f}s"
    )"""
    ),
    code_cell(
        """for epoch in range(start_epoch, EPOCHS + 1):
    t_ep = time.time()
    train_batch_sampler.set_epoch(epoch)
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    optimizer.zero_grad(set_to_none=True)
    loss_acc = {"loss": 0.0, "loss_vfl": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0}
    n_batches = 0
    t_train_start = time.time()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train 69,375]", leave=True)
    for step, (pixs, book_means, book_counts, labels) in enumerate(pbar, start=1):
        pixs = pixs.to(device, non_blocking=True)
        book_means = [m.to(device, non_blocking=True) for m in book_means]
        book_counts = book_counts.to(device, non_blocking=True)
        labels_dev = [
            {
                "boxes": lb["boxes"].to(device, non_blocking=True),
                "class_labels": lb["class_labels"].to(device, non_blocking=True),
            }
            for lb in labels
        ]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            out = model(pixs, book_means, book_counts, labels=labels_dev)
            loss = out["loss"] / GRAD_ACCUM

        loss.backward()

        if step % GRAD_ACCUM == 0 or step == len(train_loader):
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        ld = out["loss_dict"]
        loss_acc["loss"] += float(out["loss"].detach().item())
        loss_acc["loss_vfl"] += float(ld.get("loss_vfl", 0.0))
        loss_acc["loss_bbox"] += float(ld.get("loss_bbox", 0.0))
        loss_acc["loss_giou"] += float(ld.get("loss_giou", 0.0))
        n_batches += 1

        if step % 100 == 0 or step == len(train_loader):
            pbar.set_postfix(
                loss=f"{loss_acc['loss']/n_batches:.4f}",
                vfl=f"{loss_acc['loss_vfl']/n_batches:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

    train_time = time.time() - t_train_start
    train_fps = len(train_coco["images"]) / max(train_time, 1e-6)

    val_metrics = evaluate_method11_coco_map(model, val_loader, val_coco, device, desc=f"Epoch {epoch} [Val 6489]")
    delta_ratios = [round(float(r), 5) for r in model.fusion_block.last_delta_ratios]
    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
    ep_time = time.time() - t_ep

    row = {
        "epoch": epoch,
        "train_loss": loss_acc["loss"] / max(n_batches, 1),
        "loss_vfl": loss_acc["loss_vfl"] / max(n_batches, 1),
        "loss_bbox": loss_acc["loss_bbox"] / max(n_batches, 1),
        "loss_giou": loss_acc["loss_giou"] / max(n_batches, 1),
        "val_mAP": val_metrics["mAP"],
        "val_AP50": val_metrics["AP50"],
        "val_AP75": val_metrics["AP75"],
        "val_APs": val_metrics["AP_small"],
        "val_APm": val_metrics["AP_medium"],
        "val_APl": val_metrics["AP_large"],
        "per_class": val_metrics["per_class"],
        "delta_ratios": delta_ratios,
        "lr": optimizer.param_groups[0]["lr"],
        "peak_vram_gb": round(peak_vram, 2),
        "train_fps": round(train_fps, 2),
        "val_eval_sec": val_metrics["eval_time_sec"],
        "epoch_time_sec": round(ep_time, 1),
    }
    history.append(row)

    print(
        f"Epoch {epoch}/{EPOCHS} | Loss: {row['train_loss']:.4f} "
        f"(VFL={row['loss_vfl']:.4f}, Bbox={row['loss_bbox']:.4f}, GIoU={row['loss_giou']:.4f}) | "
        f"Val mAP: {row['val_mAP']:.4f} | AP50: {row['val_AP50']:.4f} | AP75: {row['val_AP75']:.4f} | "
        f"DeltaRatio: {delta_ratios} | VRAM: {peak_vram:.2f} GB | {train_fps:.1f} img/s | Time: {ep_time:.1f}s"
    )

    if val_metrics["mAP"] > best_map:
        best_map = val_metrics["mAP"]
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "delta_ratios": delta_ratios,
            },
            best_ckpt_path,
        )
        print(f"  -> Saved new BEST checkpoint (Val mAP={best_map:.4f}) -> {best_ckpt_path}")

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_map": best_map,
            "history": history,
            "epoch0": ep0_metrics,
        },
        last_ckpt_path,
    )

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({"epoch0": ep0_metrics, "history": history}, f, indent=2)"""
    ),
    code_cell(
        """print(f"Loading best checkpoint from {best_ckpt_path} for 100% Test Evaluation (4,999 pages)...")
best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(best_ckpt["model_state_dict"])

test_metrics = evaluate_method11_coco_map(model, test_loader, test_coco, device, desc="Test 100% (4,999)")
test_fps = len(test_coco["images"]) / max(test_metrics["eval_time_sec"], 1e-6)
test_metrics["test_fps"] = round(test_fps, 2)
test_metrics["best_val_epoch"] = best_ckpt["epoch"]
test_metrics["best_val_mAP"] = best_ckpt["val_metrics"]["mAP"]

with open(REPORT_DIR / "test_results_method11.json", "w", encoding="utf-8") as f:
    json.dump(test_metrics, f, indent=2)

print("=" * 70)
print(f"METHOD 11 FINAL TEST RESULTS (100% Test = {test_metrics['num_images']} pages)")
print("=" * 70)
print(f"Best Val Epoch:     {test_metrics['best_val_epoch']} (Val mAP = {test_metrics['best_val_mAP']:.4f})")
print(f"Test mAP@[.50:.95]: {test_metrics['mAP']:.4f}")
print(f"Test AP50:          {test_metrics['AP50']:.4f}")
print(f"Test AP75:          {test_metrics['AP75']:.4f}")
print(f"Test AP_small:      {test_metrics['AP_small']:.4f}")
print(f"Test AP_medium:     {test_metrics['AP_medium']:.4f}")
print(f"Test AP_large:      {test_metrics['AP_large']:.4f}")
print(f"Eval Speed:         {test_fps:.2f} img/s ({test_metrics['eval_time_sec']:.1f}s)")
print("-" * 70)
print("Per-Class Test AP@[.50:.95]:")
for cls_name, ap in test_metrics["per_class"].items():
    print(f"  {cls_name:<18}: {ap:.4f}")"""
    ),
    code_cell(
        """epochs_x = [0] + [h["epoch"] for h in history]
maps_y = [ep0_metrics["mAP"]] + [h["val_mAP"] for h in history]
ap50_y = [ep0_metrics["AP50"]] + [h["val_AP50"] for h in history]
ap75_y = [ep0_metrics["AP75"]] + [h["val_AP75"] for h in history]

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0, 0].plot([h["epoch"] for h in history], [h["train_loss"] for h in history], marker="o", label="Total Loss")
axes[0, 0].plot([h["epoch"] for h in history], [h["loss_vfl"] for h in history], marker="s", label="VFL")
axes[0, 0].plot([h["epoch"] for h in history], [h["loss_bbox"] for h in history], marker="^", label="Bbox")
axes[0, 0].plot([h["epoch"] for h in history], [h["loss_giou"] for h in history], marker="d", label="GIoU")
axes[0, 0].set_title("Method 11 Training Losses per Epoch")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(epochs_x, maps_y, marker="o", linewidth=2, label="mAP@[.50:.95]")
axes[0, 1].plot(epochs_x, ap50_y, marker="s", label="AP50")
axes[0, 1].plot(epochs_x, ap75_y, marker="^", label="AP75")
axes[0, 1].set_title("Exact Validation COCO Metrics (6,489 pages)")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("AP")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

for s_idx, s_name in enumerate(["P3 (80x80)", "P4 (40x40)", "P5 (20x20)"]):
    axes[1, 0].plot([h["epoch"] for h in history], [h["delta_ratios"][s_idx] for h in history], marker="o", label=s_name)
axes[1, 0].set_title("Residual Delta Ratio ||delta|| / ||page||")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Ratio")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

cls_names = list(test_metrics["per_class"].keys())
cls_aps = [test_metrics["per_class"][c] for c in cls_names]
bars = axes[1, 1].barh(cls_names, cls_aps, color="#2b6cb0")
axes[1, 1].set_xlim(0, 1.0)
axes[1, 1].set_title(f"Test Per-Class AP@[.50:.95] (mAP={test_metrics['mAP']:.4f})")
axes[1, 1].set_xlabel("AP")
axes[1, 1].grid(True, axis="x", alpha=0.3)
for bar, v in zip(bars, cls_aps):
    axes[1, 1].text(v + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=8)

plt.tight_layout()
fig_path = REPORT_DIR / "method11_full_charts.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved charts to {fig_path}")

# Tổng hợp so sánh nhanh với Method 10 & Method 8 (nếu có report trên đĩa)
print("\\n=== Summary Comparison ===")
print(f"Base RT-DETRv2 (Epoch 0 Val 6,489): mAP = {ep0_metrics['mAP']:.4f} | AP50 = {ep0_metrics['AP50']:.4f}")
m10_hist_p = REPO_ROOT / "methods" / "method10" / "report" / "full_history.json"
if m10_hist_p.exists():
    m10_hist = json.loads(m10_hist_p.read_text(encoding="utf-8")).get("history", [])
    if m10_hist:
        m10_best = max(m10_hist, key=lambda r: r["val_mAP"])
        print(f"Method 10 Best Val (Ep {m10_best['epoch']}):      mAP = {m10_best['val_mAP']:.4f} | AP50 = {m10_best['val_AP50']:.4f}")
print(f"Method 11 Best Val (Ep {test_metrics['best_val_epoch']}):      mAP = {test_metrics['best_val_mAP']:.4f}")
print(f"Method 11 Test (4,999 pages):       mAP = {test_metrics['mAP']:.4f} | AP50 = {test_metrics['AP50']:.4f} | AP75 = {test_metrics['AP75']:.4f}")"""
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

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print(f"Generated {NB_PATH}")
