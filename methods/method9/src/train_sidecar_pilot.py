"""Script huấn luyện thử nghiệm (Pilot Test) Method 9:
Kiến trúc YOLOv26 + Zero-Gated Multi-Scale Spatial Sidecar Heatmap trên tập Pilot DocLayNet.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO
from ultralytics.cfg import get_cfg

ROOT_DIR = Path(__file__).resolve().parents[3]
M9_DIR = ROOT_DIR / "methods" / "method9"
CACHE_DIR = M9_DIR / "cache"
WEIGHTS_DIR = M9_DIR / "weights"
REPORT_DIR = M9_DIR / "report"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 640
HEATMAP_SIZE = 80
NUM_DLN_CLASSES = 11
CATEGORIES = [
    "Caption", "Footnote", "Formula", "List-item", "Page-footer",
    "Page-header", "Picture", "Section-header", "Table", "Text", "Title"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_WEIGHTS = ROOT_DIR / "weights" / "yolo26_doclaynet_pretrained.pt"


# =========================================================================
# 1. DATASET VÀ PREDICTIONS
# =========================================================================
class PilotDataset(Dataset):
    def __init__(self, img_dir: Path, lbl_dir: Path, memmap_path: Path, index_path: Path):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        with open(index_path, "r", encoding="utf-8") as f:
            self.index_map = {int(k): v for k, v in json.load(f).items()}
        self.img_ids = sorted(list(self.index_map.keys()))
        self.hm_path = memmap_path
        self._hm = None

    @property
    def hm(self):
        if self._hm is None:
            self._hm = np.load(self.hm_path, mmap_mode="r")
        return self._hm

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        iid = self.img_ids[idx]
        img_p = self.img_dir / f"{iid}.jpg"
        lbl_p = self.lbl_dir / f"{iid}.txt"

        with Image.open(img_p) as im:
            im = im.convert("RGB")
            arr = np.array(im, dtype=np.float32) / 255.0
            pix = torch.from_numpy(arr.transpose(2, 0, 1))

        row = self.index_map[iid]
        heatmap = torch.from_numpy(np.array(self.hm[row], dtype=np.float32))

        # Đọc boxes [cls, cx, cy, nw, nh]
        boxes = []
        if lbl_p.exists():
            with open(lbl_p, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        boxes.append([float(parts[0])] + [float(x) for x in parts[1:]])
        boxes_t = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 5), dtype=torch.float32)
        return pix, heatmap, boxes_t, iid


def collate_pilot(batch):
    pixs = torch.stack([b[0] for b in batch])
    heatmaps = torch.stack([b[1] for b in batch])
    iids = [b[3] for b in batch]

    batch_idx = []
    cls_list = []
    bbox_list = []
    for i, b in enumerate(batch):
        boxes_i = b[2]  # [N_i, 5] -> cls, cx, cy, w, h
        if len(boxes_i) > 0:
            batch_idx.append(torch.full((len(boxes_i),), i, dtype=torch.long))
            cls_list.append(boxes_i[:, 0:1])
            bbox_list.append(boxes_i[:, 1:5])

    if cls_list:
        targets_dict = {
            "batch_idx": torch.cat(batch_idx),
            "cls": torch.cat(cls_list),
            "bboxes": torch.cat(bbox_list),
        }
    else:
        targets_dict = {
            "batch_idx": torch.zeros((0,), dtype=torch.long),
            "cls": torch.zeros((0, 1), dtype=torch.float32),
            "bboxes": torch.zeros((0, 4), dtype=torch.float32),
        }

    return pixs, heatmaps, targets_dict, iids


# =========================================================================
# 2. KIẾN TRÚC MẠNG SIDECAR + YOLOV26
# =========================================================================
class SpatialLayoutSidecarCNN(nn.Module):
    def __init__(self, in_channels: int = 22):
        super().__init__()
        self.stage0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        s0 = self.stage0(x)     # [B, 256, 80, 80]
        s1 = self.stage1(s0)    # [B, 512, 40, 40]
        s2 = self.stage2(s1)    # [B, 512, 20, 20]
        return [s0, s1, s2]


class Method9SidecarYOLO26(nn.Module):
    def __init__(self, pretrained_weights: str):
        super().__init__()
        # Nạp mô hình DetectionModel thuần PyTorch (tránh xung đột với ultralytics.engine.model.YOLO)
        yolo_obj = YOLO(pretrained_weights)
        self.det_model = yolo_obj.model.to(DEVICE)
        self.det_model.requires_grad_(True)

        # Cấu hình criterion
        self.criterion = self.det_model.init_criterion()
        cfg = get_cfg()
        self.criterion.one2many.hyp = cfg
        self.criterion.one2one.hyp = cfg

        self.sidecar = SpatialLayoutSidecarCNN(in_channels=22).to(DEVICE)
        self.fusion_gates = nn.Parameter(torch.zeros(3, device=DEVICE))
        self._current_sidecar_feats: list[torch.Tensor] | None = None

        # Hooks hòa trộn tại Layer 16, 19, 22
        def hook_p3(m, inp, out):
            if self._current_sidecar_feats is not None:
                return out + self.fusion_gates[0] * self._current_sidecar_feats[0]
            return out

        def hook_p4(m, inp, out):
            if self._current_sidecar_feats is not None:
                return out + self.fusion_gates[1] * self._current_sidecar_feats[1]
            return out

        def hook_p5(m, inp, out):
            if self._current_sidecar_feats is not None:
                return out + self.fusion_gates[2] * self._current_sidecar_feats[2]
            return out

        self.det_model.model[16].register_forward_hook(hook_p3)
        self.det_model.model[19].register_forward_hook(hook_p4)
        self.det_model.model[22].register_forward_hook(hook_p5)

    def forward(self, pixs: torch.Tensor, heatmaps: torch.Tensor):
        self._current_sidecar_feats = self.sidecar(heatmaps)
        out = self.det_model(pixs)
        self._current_sidecar_feats = None
        return out

    def compute_loss(self, preds, batch_targets):
        return self.criterion(preds, batch_targets)


# =========================================================================
# 3. HUẤN LUYỆN PILOT
# =========================================================================
def run_sidecar_pilot(epochs: int = 3, batch_size: int = 8):
    print("=" * 80)
    print("🚀 BẮT ĐẦU CHẠY THỬ NGHIỆM METHOD 9 (YOLOv26 + SPATIAL SIDECAR HEATMAP)")
    print(f" - Thiết bị: {DEVICE} ({torch.cuda.get_device_name(0)})")
    print(f" - Số Epochs: {epochs} | Batch Size: {batch_size}")
    print("=" * 80)

    train_img_dir = M9_DIR / "data" / "images" / "train"
    train_lbl_dir = M9_DIR / "data" / "labels" / "train"
    val_img_dir = M9_DIR / "data" / "images" / "val"
    val_lbl_dir = M9_DIR / "data" / "labels" / "val"

    train_hm_p = CACHE_DIR / "heatmaps_pilot_train.npy"
    train_idx_p = CACHE_DIR / "heatmap_index_pilot_train.json"
    val_hm_p = CACHE_DIR / "heatmaps_pilot_val.npy"
    val_idx_p = CACHE_DIR / "heatmap_index_pilot_val.json"

    train_ds = PilotDataset(train_img_dir, train_lbl_dir, train_hm_p, train_idx_p)
    val_ds = PilotDataset(val_img_dir, val_lbl_dir, val_hm_p, val_idx_p)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_pilot, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_pilot, pin_memory=True)

    model = Method9SidecarYOLO26(str(BASE_WEIGHTS))

    optimizer = torch.optim.AdamW([
        {"params": model.sidecar.parameters(), "lr": 1e-4, "weight_decay": 1e-4},
        {"params": [model.fusion_gates], "lr": 5e-3, "weight_decay": 0.0},
        {"params": model.det_model.parameters(), "lr": 5e-5, "weight_decay": 1e-4},
    ])
    scaler = torch.amp.GradScaler("cuda")

    print("\n🔥 BẮT ĐẦU HUẤN LUYỆN 3 EPOCHS:")
    t_start = time.time()
    history = []

    for ep in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        total_loss = 0.0
        step_count = 0

        for step, (pixs, heatmaps, targets, _) in enumerate(train_loader):
            pixs = pixs.to(DEVICE, non_blocking=True)
            heatmaps = heatmaps.to(DEVICE, non_blocking=True)
            targets = {k: v.to(DEVICE, non_blocking=True) for k, v in targets.items()}

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds = model(pixs, heatmaps)
                loss_tensor, loss_items = model.compute_loss(preds, targets)
                loss = loss_tensor.sum()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            step_count += 1

            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                vram = torch.cuda.memory_allocated() / (1024**2)
                fps = ((step + 1) * batch_size) / max(time.time() - ep_t0, 1e-4)
                print(f"   * Ep {ep} | Step [{step+1:3d}/{len(train_loader):3d}] | Loss: {loss.item():.4f} | VRAM: {vram:.0f} MB | {fps:.1f} FPS")

        ep_duration = time.time() - ep_t0
        avg_loss = total_loss / max(step_count, 1)
        gates_now = [round(float(g), 5) for g in model.fusion_gates.detach().cpu().numpy()]
        vram_mb = torch.cuda.memory_allocated() / (1024**2)

        print(f"🏁 [Epoch {ep}/{epochs}] Loss Trung Bình: {avg_loss:.4f} | Fusion Gates [P3, P4, P5]: {gates_now} | Thời gian: {ep_duration:.1f}s")
        history.append({
            "epoch": ep,
            "loss": avg_loss,
            "gates": gates_now,
            "duration_sec": ep_duration,
            "vram_mb": vram_mb,
        })

    # Đánh giá độ trễ suy luận trên tập Val
    print("\n🔍 ĐANG ĐÁNH GIÁ ĐỘ TRỄ SUY LUẬN TRÊN 500 ẢNH VAL...")
    model.eval()
    t_eval_start = time.time()
    with torch.no_grad():
        for pixs, heatmaps, _, _ in val_loader:
            pixs = pixs.to(DEVICE, non_blocking=True)
            heatmaps = heatmaps.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _ = model(pixs, heatmaps)
    val_time = time.time() - t_eval_start
    fps = len(val_ds) / max(val_time, 1e-4)

    # Lưu best checkpoint
    best_ckpt = WEIGHTS_DIR / "yolo26_sidecar_pilot_test.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "fusion_gates": model.fusion_gates.detach().cpu().numpy(),
        "history": history,
    }, best_ckpt)
    print(f"✅ Đã lưu checkpoint thử nghiệm vào: {best_ckpt.name}")

    results_data = {
        "model": "YOLOv26-Spatial-Sidecar",
        "epochs": epochs,
        "batch_size": batch_size,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "total_time_sec": time.time() - t_start,
        "val_latency_ms": (val_time / len(val_ds)) * 1000,
        "fps_eval": fps,
        "final_fusion_gates": [float(g) for g in model.fusion_gates.detach().cpu().numpy()],
        "history": history,
    }

    report_file = REPORT_DIR / "sidecar_pilot_results.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    print("\n" + "=" * 80)
    print("🏆 KẾT QUẢ THỬ NGHIỆM THÀNH CÔNG METHOD 9 (YOLOv26 + SPATIAL SIDECAR):")
    print(f" - Tổng thời gian chạy   : {results_data['total_time_sec']:.1f} giây")
    print(f" - Độ trễ suy luận       : {results_data['val_latency_ms']:.2f} ms / ảnh (~{fps:.1f} FPS)")
    print(f" - Mức tiêu thụ VRAM     : {vram_mb:.0f} MB")
    print(f" - Giá trị Fusion Gates  : {results_data['final_fusion_gates']}")
    print("=" * 80)
    return results_data


if __name__ == "__main__":
    run_sidecar_pilot(epochs=3, batch_size=8)
