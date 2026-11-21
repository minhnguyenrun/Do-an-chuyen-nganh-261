"""Build Stage 03 Notebook: 03.1 (DINO+M2Doc 3-5 pages/book) + 03.2 (YOLOv26 Noise Adaptation)."""

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent / "03_stage03_dino_anchors_and_adapt.ipynb"


def md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in source.splitlines()]}


def code_cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in source.splitlines()]}


cells = [
    md_cell(
        """# Method 12 — Stage 03 (03.1 & 03.2): DINO+M2Doc Anchors & Noise Adaptation (YOLOv26)

- **Stage 03.1:** Chạy `DINO + M2Doc` trên **3–5 trang/sách** của tập `train` rồi lưu cache (`dino_anchors_train.pt`).
- **Stage 03.2:** Nạp checkpoint `weights/best_stage02.pt`, fine-tuning tiếp trên **Book-Context nhiễu thực tế từ DINO+M2Doc** trong 3 epochs, lưu **`weights/best_stage03_2.pt`**."""
    ),
    code_cell(
        """import sys
import time
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
    TRAIN_ANNO, TRAIN_IMG_640, TRAIN_IMG_SRC,
    CACHE_DIR, WEIGHTS_DIR, load_coco, select_anchor_pages,
)
from methods.method12.dino_anchor_runner import run_dino_m2doc_on_anchors
from methods.method12.models.stage01_yolo import gpu_photometric_augment
from methods.method12.models.stage02_yolo_fusion import Stage02YOLOFusionDetector
from methods.method12.yolo_dataset import Method12YOLOFusionDataset, collate_fn_yolo_fusion, precompute_ocr_grids

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STAGE02_CKPT = WEIGHTS_DIR / "best_stage02.pt"
DINO_TRAIN_CACHE = CACHE_DIR / "dino_anchors_train.pt"

BATCH_SIZE = 16
EPOCHS_ADAPT = 3
LR_DETECTOR = 2e-5
LR_FUSION = 8e-5
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2"""
    ),
    code_cell(
        """# === STAGE 03.1: DINO + M2Doc trên 3-5 trang/sách của tập Train ===
train_coco = load_coco(TRAIN_ANNO)
train_anchors_ids = select_anchor_pages(train_coco["images"], min_pages=3, max_pages=5, ratio=0.10, seed=SEED)
train_anchor_set = {pid for pids in train_anchors_ids.values() for pid in pids}
print(f"[Stage 03.1] Anchor pages: {len(train_anchor_set):,} across {len(train_anchors_ids):,} books.")

if DINO_TRAIN_CACHE.exists():
    train_dino_anchors = torch.load(DINO_TRAIN_CACHE, weights_only=False)["book_anchors"]
    print(f"[Stage 03.1] Loaded cached DINO+M2Doc anchors: {len(train_dino_anchors):,} books.")
else:
    train_dino_anchors, _ = run_dino_m2doc_on_anchors(
        coco=train_coco,
        anchors_by_book=train_anchors_ids,
        dataset_name="doclaynet",
        img_raw_dir=TRAIN_IMG_SRC,
        img_640_dir=TRAIN_IMG_640,
        device=str(device),
        score_thresh=0.35,
        save_cache_path=DINO_TRAIN_CACHE,
    )"""
    ),
    code_cell(
        """# === STAGE 03.2: Fine-tuning thích nghi nhiễu DINO+M2Doc ===
train_ocr_grids = precompute_ocr_grids(train_coco, "doclaynet", TRAIN_IMG_SRC, TRAIN_IMG_640, CACHE_DIR / "ocr_grids_train.pt")
train_ds = Method12YOLOFusionDataset(
    train_coco, TRAIN_IMG_640, TRAIN_IMG_SRC,
    book_anchors=train_dino_anchors,
    exclude_image_ids=train_anchor_set,
    ocr_grids=train_ocr_grids,
)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=(NUM_WORKERS > 0), collate_fn=collate_fn_yolo_fusion)

model = Stage02YOLOFusionDetector(stage01_ckpt_path=None, freeze_backbone=True).to(device)
ckpt02 = torch.load(STAGE02_CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt02["model"], strict=True)
print(f"[Stage 03.2] Loaded Stage 02 checkpoint: {STAGE02_CKPT}")

fusion_params = list(model.early_sidecar.parameters()) + [model.early_gates] + list(model.late_fusion.parameters())
fusion_ids = {id(p) for p in fusion_params}
det_params = [p for p in model.parameters() if p.requires_grad and id(p) not in fusion_ids]

optimizer = torch.optim.AdamW([
    {"params": det_params, "lr": LR_DETECTOR, "weight_decay": WEIGHT_DECAY},
    {"params": fusion_params, "lr": LR_FUSION, "weight_decay": WEIGHT_DECAY},
])
all_trainable = det_params + fusion_params

for epoch in range(1, EPOCHS_ADAPT + 1):
    t_ep = time.time()
    model.train()
    running_loss = 0.0
    n_batches = 0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Stage03.2 YOLO Adapt Epoch {epoch}/{EPOCHS_ADAPT}")
    for step_idx, (pixs, early_19, anchor_mem, anchor_mask, targets, _) in pbar:
        pixs = gpu_photometric_augment(pixs.to(device, non_blocking=True))
        early_19 = early_19.to(device, non_blocking=True)
        anchor_mem = anchor_mem.to(device, non_blocking=True)
        anchor_mask = anchor_mask.to(device, non_blocking=True)
        targets_dev = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            out = model(pixs, early_19, anchor_mem, anchor_mask, targets=targets_dev)
            loss = out["loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=10.0)
        optimizer.step()

        running_loss += float(loss.item())
        n_batches += 1
        if step_idx % 200 == 0:
            pbar.set_postfix({"loss": f"{running_loss / max(n_batches, 1):.4f}"})

    print(f"[Stage 03.2 Epoch {epoch}] loss={running_loss / max(n_batches, 1):.4f} | time={(time.time()-t_ep)/60:.1f}m")

out_ckpt = WEIGHTS_DIR / "best_stage03_2.pt"
torch.save({"model": model.state_dict(), "epochs_adapt": EPOCHS_ADAPT}, out_ckpt)
print(f"[Stage 03.2] Saved adapted checkpoint -> {out_ckpt}")"""
    ),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12.0"}}, "nbformat": 4, "nbformat_minor": 5}
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
print(f"Generated notebook: {NB_PATH}")
