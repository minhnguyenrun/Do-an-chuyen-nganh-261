"""Huấn luyện Thử Nghiệm (Pilot Run) Method 6: Spatial Sidecar Heatmap RT-DETR.

Mục tiêu:
1. Đánh giá tính khả thi và độ ổn định của kiến trúc Sidecar Heatmap Fusion.
2. Huấn luyện trên tập nhỏ (5 sách train ~150 trang, 2 sách val ~45 trang).
3. Đánh giá sự tiến hóa của trọng số Fusion Gates và sự suy giảm của Hungarian Loss.
4. Đo đạc COCO mAP@[0.50:0.95] sau mỗi epoch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Force UTF-8 stdout
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from methods.method6.models.rtdetr_method6 import SpatialSidecarRTDetrMethod6
from methods.method6.data.doclaynet_spatial_dataset import DocLayNetSpatialDataset, collate_spatial_fn
from methods.method6.models.evaluator import evaluate_method6_coco_map


def run_pilot_training():
    print("=" * 80)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN PILOT RUN: METHOD 6 (SPATIAL SIDECAR HEATMAP RT-DETR)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Thiết bị: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Đường dẫn dataset
    train_anno = ROOT_DIR / "datasets" / "doclaynet" / "processed" / "annotations" / "train.json"
    train_img = ROOT_DIR / "datasets" / "doclaynet" / "processed" / "images"
    val_anno = ROOT_DIR / "datasets" / "doclaynet_test" / "processed" / "annotations" / "test.json"
    val_img = ROOT_DIR / "datasets" / "doclaynet_test" / "processed" / "images"

    selected_books_file = ROOT_DIR / "methods" / "method5" / "data" / "selected_books.json"
    with open(selected_books_file, "r", encoding="utf-8") as f:
        sb_data = json.load(f)

    # Chọn 5 sách train (~150 trang) và 2 sách val (~45 trang)
    pilot_train_books = sb_data["train_books"][:5]
    pilot_val_books = sb_data["val_books"][:2]

    print(f"[Dataset] Sách train tuyển chọn (5 cuốn): {pilot_train_books}")
    print(f"[Dataset] Sách val tuyển chọn (2 cuốn):   {pilot_val_books}")

    train_ds = DocLayNetSpatialDataset(
        annotation_file=train_anno,
        images_dir=train_img,
        image_size=640,
        heatmap_size=80,
        window_size=64,
        is_train=True,
        allowed_books=pilot_train_books,
    )

    val_ds = DocLayNetSpatialDataset(
        annotation_file=val_anno,
        images_dir=val_img,
        image_size=640,
        heatmap_size=80,
        window_size=64,
        is_train=False,
        allowed_books=pilot_val_books,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_spatial_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_spatial_fn,
        pin_memory=True,
    )

    # 2. Khởi tạo Model
    model = SpatialSidecarRTDetrMethod6(
        pretrained_model_path="docling-project/docling-layout-heron",
        num_classes=17,
        heatmap_channels=22,
        freeze_backbone=True,  # Đóng băng CNN backbone để train nhẹ và nhanh
    ).to(device)

    # 3. Differential Learning Rates
    param_groups = [
        # Nhánh 1: Sidecar CNN (học trích xuất bản đồ nhiệt)
        {"params": model.sidecar_cnn.parameters(), "lr": 2e-4, "weight_decay": 1e-4},
        # Nhánh 2: Fusion Gates
        {"params": [model.fusion_gates], "lr": 1e-3, "weight_decay": 0.0},
        # Nhánh 3: Encoder & Decoder của RT-DETR 2 (fine-tuning nhẹ)
        {"params": [p for n, p in model.rtdetr.named_parameters() if p.requires_grad], "lr": 2e-5, "weight_decay": 1e-4},
    ]

    optimizer = torch.optim.AdamW(param_groups)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Đánh giá Baseline trước khi train (Epoch 0)
    print("\n--- ĐÁNH GIÁ BASELINE TRƯỚC HUẤN LUYỆN (EPOCH 0) ---")
    val_res_0 = evaluate_method6_coco_map(model, val_loader, device=device)
    print(f"✓ Epoch 0 Baseline: mAP@[.50:.95]={val_res_0['map']*100:.2f}% | AP50={val_res_0['map50']*100:.2f}% | AP75={val_res_0['map75']*100:.2f}%")
    print(f"  Fusion Gates ban đầu: {model.fusion_gates.detach().cpu().tolist()}")

    # 4. Huấn luyện 2 Epochs
    num_epochs = 2
    grad_accum_steps = 2
    history = []

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        print(f"\n>>> BẮT ĐẦU EPOCH {epoch}/{num_epochs}")
        for step, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            layout_heatmaps = batch["layout_heatmaps"].to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in batch["targets"]]

            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                out = model(pixel_values, layout_heatmaps, labels=targets)
                loss = out["loss"] / grad_accum_steps

            scaler.scale(loss).backward()
            total_loss += loss.item() * grad_accum_steps

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if (step + 1) % 5 == 0 or (step + 1) == len(train_loader):
                gates = model.fusion_gates.detach().cpu().tolist()
                print(f"  [Ep {epoch}] Step {step+1:02d}/{len(train_loader):02d} | Loss: {loss.item()*grad_accum_steps:.4f} | Gates [P3, P4, P5]: [{gates[0]:+.4f}, {gates[1]:+.4f}, {gates[2]:+.4f}]")

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(train_loader)

        # Đánh giá sau epoch
        print(f"\n--- ĐÁNH GIÁ SAU EPOCH {epoch} ({epoch_time:.1f}s) ---")
        val_res = evaluate_method6_coco_map(model, val_loader, device=device)
        gates = model.fusion_gates.detach().cpu().tolist()
        print(f"✓ Kết quả Epoch {epoch}:")
        print(f"  - Hungarian Loss:  {avg_loss:.4f}")
        print(f"  - COCO mAP@[.50:.95]: {val_res['map']*100:.2f}% (Thay đổi: {(val_res['map'] - val_res_0['map'])*100:+.2f}%)")
        print(f"  - COCO AP50:          {val_res['map50']*100:.2f}%")
        print(f"  - COCO AP75:          {val_res['map75']*100:.2f}%")
        print(f"  - Fusion Gates:       P3={gates[0]:+.4f}, P4={gates[1]:+.4f}, P5={gates[2]:+.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_map": val_res["map"],
            "val_map50": val_res["map50"],
            "val_map75": val_res["map75"],
            "gates": gates,
            "time_sec": epoch_time,
        })

    # Lưu trọng số checkpoint pilot
    save_dir = ROOT_DIR / "methods" / "method6" / "weights"
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "method6_pilot_checkpoint.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "history": history,
        "pilot_train_books": pilot_train_books,
        "pilot_val_books": pilot_val_books,
    }, ckpt_path)
    print(f"\n[OK] Đã lưu checkpoint pilot tại: {ckpt_path}")

    # Lưu báo cáo JSON
    def serialize_obj(obj):
        if isinstance(obj, (torch.Tensor, np.integer)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, dict):
            return {str(k): serialize_obj(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [serialize_obj(v) for v in obj]
        return obj

    report_dir = ROOT_DIR / "methods" / "method6" / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "pilot_run_result.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(serialize_obj({
            "epoch_0_baseline": val_res_0,
            "history": history,
        }), f, indent=2)
    print(f"[OK] Đã lưu kết quả pilot tại: {report_path}")

    print("\n" + "=" * 80)
    print("🎉 HOÀN THÀNH HUẤN LUYỆN PILOT RUN CHO METHOD 6!")
    print("=" * 80)


if __name__ == "__main__":
    run_pilot_training()
