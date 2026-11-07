"""Huấn Luyện Chính Thức Method 6: Spatial Sidecar Heatmap RT-DETR.

Quy mô thực nghiệm:
- 50 sách Train tuyển chọn (1,669 trang).
- 10 sách Validation tuyển chọn (221 trang).
- Cửa sổ ngữ cảnh sách: 64 trang (window_size=64).
- 2D Layout Heatmap 22 kênh (11 Book Prior + 11 Page Prior).
- Đóng băng Backbone RT-DETR, fine-tune Sidecar CNN + Hybrid Encoder + Decoder với Hungarian Matching Loss.
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


def train_method6(args):
    print("=" * 80)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN METHOD 6: 50 SÁCH TRAIN (1,669 TRANG) & 10 SÁCH VAL (221 TRANG)")
    print("=" * 80)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] Thiết bị: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Dataset & DataLoader
    train_anno = ROOT_DIR / "datasets" / "doclaynet" / "processed" / "annotations" / "train.json"
    train_img = ROOT_DIR / "datasets" / "doclaynet" / "processed" / "images"
    val_anno = ROOT_DIR / "datasets" / "doclaynet_test" / "processed" / "annotations" / "test.json"
    val_img = ROOT_DIR / "datasets" / "doclaynet_test" / "processed" / "images"

    selected_books_file = ROOT_DIR / "methods" / "method5" / "data" / "selected_books.json"
    with open(selected_books_file, "r", encoding="utf-8") as f:
        sb_data = json.load(f)

    if args.all_train_books:
        train_allowed_books = None
        print(f"[Dataset] SỬ DỤNG TOÀN BỘ SÁCH TRONG TẬP TRAIN (ALL 328 CUỐN SÁCH, ~10,604 TRANG)!")
    else:
        train_allowed_books = sb_data["train_books"][:args.num_train_books]
        print(f"[Dataset] Tuyển chọn: {len(train_allowed_books)} sách Train.")

    val_allowed_books = sb_data["val_books"][:10]
    print(f"[Dataset] Đánh giá định kỳ: {len(val_allowed_books)} sách Val (221 trang).")

    train_ds = DocLayNetSpatialDataset(
        annotation_file=train_anno,
        images_dir=train_img,
        image_size=640,
        heatmap_size=80,
        window_size=args.window_size,
        is_train=True,
        allowed_books=train_allowed_books,
    )

    val_ds = DocLayNetSpatialDataset(
        annotation_file=val_anno,
        images_dir=val_img,
        image_size=640,
        heatmap_size=80,
        window_size=args.window_size,
        is_train=False,
        allowed_books=val_allowed_books,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_spatial_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_spatial_fn,
        pin_memory=True,
    )

    print(f"[DataLoader] Train: {len(train_ds):,} trang ({len(train_loader)} batches) | Val: {len(val_ds):,} trang ({len(val_loader)} batches).")

    # 2. Khởi tạo Model
    model = SpatialSidecarRTDetrMethod6(
        pretrained_model_path="docling-project/docling-layout-heron",
        num_classes=17,
        heatmap_channels=22,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    start_epoch = 1
    if args.resume and Path(args.resume).exists():
        print(f"[Resume] Nạp trọng số từ checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
            print(f"[Resume] Bắt đầu tiếp tục từ Epoch {start_epoch}")
        print(f"[Resume] Fusion Gates hiện tại: {model.fusion_gates.detach().cpu().tolist()}")

    # 3. Differential Learning Rates
    param_groups = [
        # Nhánh 1: Sidecar CNN (học trích xuất bản đồ nhiệt layout)
        {"params": model.sidecar_cnn.parameters(), "lr": args.lr_sidecar, "weight_decay": 1e-4},
        # Nhánh 2: Fusion Gates
        {"params": [model.fusion_gates], "lr": args.lr_gates, "weight_decay": 0.0},
        # Nhánh 3: Encoder & Decoder của RT-DETR 2 (fine-tuning nhẹ)
        {"params": [p for n, p in model.rtdetr.named_parameters() if p.requires_grad], "lr": args.lr_rtdetr, "weight_decay": 1e-4},
    ]

    total_trainable = sum(p.numel() for pg in param_groups for p in pg["params"])
    print(f"[Model] Tổng tham số huấn luyện: {total_trainable:,}")

    optimizer = torch.optim.AdamW(param_groups)
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    print(f"[Device] Chế độ AMP: {amp_dtype}")

    # Đánh giá Baseline trước khi train (Epoch 0)
    print("\n--- ĐÁNH GIÁ BASELINE TRƯỚC HUẤN LUYỆN (EPOCH 0 TRÊN 221 TRANG VAL) ---")
    val_res_0 = evaluate_method6_coco_map(model, val_loader, device=device)
    print(f"✓ Epoch 0 Baseline: mAP@[.50:.95]={val_res_0['map']*100:.2f}% | AP50={val_res_0['map50']*100:.2f}% | AP75={val_res_0['map75']*100:.2f}%")
    print(f"  Fusion Gates ban đầu: {model.fusion_gates.detach().cpu().tolist()}")

    # 4. Huấn luyện các Epochs
    best_map = val_res_0["map"]
    best_epoch = 0
    history = []

    weights_dir = ROOT_DIR / "methods" / "method6" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    report_dir = ROOT_DIR / "methods" / "method6" / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        print(f"\n>>> BẮT ĐẦU EPOCH {epoch}/{start_epoch + args.epochs - 1}")
        for step, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            layout_heatmaps = batch["layout_heatmaps"].to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in batch["targets"]]

            try:
                with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=amp_dtype):
                    out = model(pixel_values, layout_heatmaps, labels=targets)
                    loss = out["loss"] / args.grad_accum_steps

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  [Warning] Bỏ qua batch {step+1} do Loss là NaN/Inf.")
                    optimizer.zero_grad()
                    continue

                loss.backward()
                total_loss += loss.item() * args.grad_accum_steps

                if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    optimizer.zero_grad()

            except Exception as e:
                print(f"  [Warning] Bỏ qua batch {step+1} do lỗi matching: {type(e).__name__}: {e}")
                optimizer.zero_grad()
                continue

            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                gates = model.fusion_gates.detach().cpu().tolist()
                print(f"  [Ep {epoch}] Step {step+1:04d}/{len(train_loader):04d} | Loss: {loss.item()*args.grad_accum_steps:.4f} | Gates [P3, P4, P5]: [{gates[0]:+.4f}, {gates[1]:+.4f}, {gates[2]:+.4f}]")

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(train_loader)

        # Đánh giá sau epoch
        print(f"\n--- ĐÁNH GIÁ SAU EPOCH {epoch} ({epoch_time:.1f}s) TRÊN 221 TRANG VAL ---")
        val_res = evaluate_method6_coco_map(model, val_loader, device=device)
        gates = model.fusion_gates.detach().cpu().tolist()
        delta_map = (val_res['map'] - val_res_0['map']) * 100
        print(f"✓ Kết quả Epoch {epoch}:")
        print(f"  - Hungarian Loss:     {avg_loss:.4f}")
        print(f"  - COCO mAP@[.50:.95]: {val_res['map']*100:.2f}% (Thay đổi: {delta_map:+.2f}%)")
        print(f"  - COCO AP50:          {val_res['map50']*100:.2f}%")
        print(f"  - COCO AP75:          {val_res['map75']*100:.2f}%")
        print(f"  - Fusion Gates:       P3={gates[0]:+.4f}, P4={gates[1]:+.4f}, P5={gates[2]:+.4f}")

        # Lưu checkpoint epoch
        epoch_ckpt = weights_dir / f"method6_epoch_{epoch}.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "fusion_gates": gates,
            "val_res": val_res,
        }, epoch_ckpt)

        if val_res["map"] > best_map:
            best_map = val_res["map"]
            best_epoch = epoch
            best_ckpt = weights_dir / "method6_best.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "fusion_gates": gates,
                "val_res": val_res,
            }, best_ckpt)
            print(f"  🔥 KỶ LỤC MỚI! Đã lưu checkpoint tốt nhất tại: {best_ckpt}")

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_map": val_res["map"],
            "val_map50": val_res["map50"],
            "val_map75": val_res["map75"],
            "gates": gates,
            "time_sec": epoch_time,
        })

    # Lưu lịch sử train
    history_path = report_dir / "train_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(serialize_obj({
            "best_epoch": best_epoch,
            "best_map": best_map,
            "epoch_0_baseline": val_res_0,
            "history": history,
        }), f, indent=2)

    print("\n" + "=" * 80)
    print(f"🎉 HOÀN THÀNH HUẤN LUYỆN METHOD 6! BEST EPOCH: {best_epoch} (mAP: {best_map*100:.2f}%)")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--lr_sidecar", type=float, default=2e-4)
    parser.add_argument("--lr_gates", type=float, default=1e-3)
    parser.add_argument("--lr_rtdetr", type=float, default=2e-5)
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--all_train_books", action="store_true", default=False)
    parser.add_argument("--num_train_books", type=int, default=50)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    train_method6(args)
