"""Huấn Luyện Chính Thức Method 8: Fast SSD Pre-Resized + Zero-Gated Sidecar RT-DETRv2.

Cải tiến của Method 8:
1. Đọc trực tiếp ảnh 640x640 từ SSD: loại bỏ hoàn toàn độ trễ giải nén CPU.
2. Heatmap 22 kênh memmap sạch từ dự đoán Base RT-DETRv2 (không leak nhãn GT).
3. Độc lập Train/Val theo cuốn sách, lưu checkpoint đầy đủ kèm optimizer/scheduler.
4. Đánh giá COCO mAP@[.50:.95] chuẩn bằng pycocotools.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from methods.method8.common import (
    BASE_MODEL, CACHE_DIR, REPORT_DIR, TRAIN_ANNO, TRAIN_IMG_640,
    TRAIN_IMG_SRC, VAL_ANNO, VAL_IMG_640, VAL_IMG_SRC, WEIGHTS_DIR,
    load_coco,
)
from methods.method8.dataset import Method8FastDataset, collate
from methods.method8.models.rtdetr_method8 import SpatialSidecarRTDetrMethod8
from methods.method8.evaluator import evaluate_method8_coco_map


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện Method 8 (Fast Pre-resized SSD + Sidecar)")
    parser.add_argument("--epochs", type=int, default=5, help="Số epoch huấn luyện")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size trên mỗi GPU step")
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="Tích lũy gradient (effective batch size = batch_size * grad_accum)")
    parser.add_argument("--lr_sidecar", type=float, default=1e-4, help="Learning rate cho Sidecar CNN")
    parser.add_argument("--lr_rtdetr", type=float, default=1e-5, help="Learning rate cho RT-DETR encoder/decoder")
    parser.add_argument("--lr_gates", type=float, default=5e-3, help="Learning rate cho Fusion Gates")
    parser.add_argument("--warmup_epochs", type=int, default=1, help="Số epoch linear warmup")
    parser.add_argument("--num_workers", type=int, default=4, help="Số worker cho DataLoader")
    parser.add_argument("--freeze_backbone", action="store_true", default=True, help="Đóng băng backbone ResNet/HGNet")
    parser.add_argument("--resume", type=str, default="", help="Đường dẫn checkpoint để nạp tiếp tục")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN METHOD 8: FAST PRE-RESIZED SSD + SPATIAL SIDECAR RT-DETR")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Thiết bị: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Nạp annotation & chuẩn bị Train (toàn bộ train.json) và Val (tập val.json chính thức)
    print(f"[Dataset] Nạp annotation tập train từ: {TRAIN_ANNO}")
    train_coco = load_coco(TRAIN_ANNO)
    print(f"[Dataset] Nạp annotation tập validation chính thức từ: {VAL_ANNO}")
    val_coco = load_coco(VAL_ANNO)
    print(f"[Dataset] Quy mô: {len(train_coco['images']):,} trang Train | {len(val_coco['images']):,} trang Val chính thức")

    train_ds = Method8FastDataset(
        train_coco, split="train", img_640_dir=TRAIN_IMG_640, img_raw_dir=TRAIN_IMG_SRC,
        drop_empty=True
    )
    val_ds = Method8FastDataset(
        val_coco, split="val", img_640_dir=VAL_IMG_640, img_raw_dir=VAL_IMG_SRC,
        drop_empty=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    print(f"[DataLoader] Train: {len(train_ds):,} trang ({len(train_loader)} steps/ep) | Val: {len(val_ds):,} trang ({len(val_loader)} steps).")

    # 2. Khởi tạo mô hình
    model = SpatialSidecarRTDetrMethod8(
        pretrained_model_path=BASE_MODEL,
        num_classes=17,
        heatmap_channels=22,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    # 3. Phân nhóm tham số & Optimizer
    param_groups = [
        {"params": model.sidecar_cnn.parameters(), "lr": args.lr_sidecar, "name": "sidecar_cnn"},
        {"params": [model.fusion_gates], "lr": args.lr_gates, "name": "fusion_gates"},
    ]
    rtdetr_params = [
        p for n, p in model.rtdetr.named_parameters()
        if p.requires_grad and "backbone" not in n
    ]
    if rtdetr_params:
        param_groups.append({"params": rtdetr_params, "lr": args.lr_rtdetr, "name": "rtdetr_encoder_decoder"})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    total_steps = len(train_loader) * args.epochs // args.grad_accum_steps
    warmup_steps = len(train_loader) * args.warmup_epochs // args.grad_accum_steps

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    start_epoch = 1
    best_map = 0.0
    history = []
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Resume nếu có checkpoint
    if args.resume and Path(args.resume).exists():
        print(f"[Resume] Đang nạp checkpoint từ: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_map = ckpt.get("best_map", 0.0)
        history = ckpt.get("history", [])
        print(f"[Resume] Tiếp tục từ Epoch {start_epoch}, Best mAP trước đó: {best_map:.4f}")

    # 4. Baseline Evaluation trước khi train
    print("\n--- ĐÁNH GIÁ BASELINE TRƯỚC KHI TRAIN TRÊN TẬP VAL ---")
    val_metrics = evaluate_method8_coco_map(model, val_loader, train_coco, device)
    print(f"[Baseline Val] mAP@[.50:.95]: {val_metrics['map']*100:.2f}% | AP50: {val_metrics['map50']*100:.2f}% | AP75: {val_metrics['map75']*100:.2f}%")
    best_map = max(best_map, val_metrics["map"])

    # 5. Huấn luyện từng Epoch

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*30} EPOCH {epoch}/{args.epochs} {'='*30}")
        model.train()
        total_loss = 0.0
        total_vfl = 0.0
        total_bbox = 0.0
        total_giou = 0.0
        t0 = time.time()
        optimizer.zero_grad()

        for step, (pixel_values, layout_heatmaps, targets) in enumerate(train_loader):
            pixel_values = pixel_values.to(device, non_blocking=True)
            layout_heatmaps = layout_heatmaps.to(device, non_blocking=True)
            labels = [{
                "class_labels": t["class_labels"].to(device),
                "boxes": t["boxes"].to(device),
            } for t in targets]

            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16):
                outputs = model(pixel_values, layout_heatmaps, labels=labels)
                loss = outputs["loss"] / args.grad_accum_steps
                loss_dict = outputs.get("loss_dict", {})

            scaler.scale(loss).backward()
            total_loss += loss.item() * args.grad_accum_steps
            total_vfl += float(loss_dict.get("loss_vfl", 0.0))
            total_bbox += float(loss_dict.get("loss_bbox", 0.0))
            total_giou += float(loss_dict.get("loss_giou", 0.0))

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            if (step + 1) % 100 == 0 or (step + 1) == len(train_loader):
                cur_loss = total_loss / (step + 1)
                cur_vfl = total_vfl / (step + 1)
                cur_bbox = total_bbox / (step + 1)
                cur_giou = total_giou / (step + 1)
                gates = model.fusion_gates.detach().cpu().numpy()
                cur_lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t0
                fps = (step + 1) * args.batch_size / max(elapsed, 1e-4)
                print(f"[Ep {epoch} Step {step+1}/{len(train_loader)}] Loss: {cur_loss:.4f} (vfl:{cur_vfl:.3f}, l1:{cur_bbox:.3f}, giou:{cur_giou:.3f}) | Gates: [{gates[0]:.4f}, {gates[1]:.4f}, {gates[2]:.4f}] | LR: {cur_lr:.2e} | {fps:.1f} FPS")

        avg_loss = total_loss / len(train_loader)
        avg_vfl = total_vfl / len(train_loader)
        avg_bbox = total_bbox / len(train_loader)
        avg_giou = total_giou / len(train_loader)
        epoch_time = time.time() - t0
        gates_val = model.fusion_gates.detach().cpu().tolist()
        peak_vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0

        # Đánh giá Val sau mỗi Epoch
        print(f"\n--- ĐÁNH GIÁ SAU EPOCH {epoch} TRÊN TẬP VAL ---")
        val_metrics = evaluate_method8_coco_map(model, val_loader, val_coco, device)
        cur_map = val_metrics["map"]
        print(f"[Val Ep {epoch}] mAP: {cur_map*100:.2f}% | AP50: {val_metrics['map50']*100:.2f}% | AP75: {val_metrics['map75']*100:.2f}% | VRAM: {peak_vram:.1f} MB")

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_loss_vfl": avg_vfl,
            "train_loss_bbox": avg_bbox,
            "train_loss_giou": avg_giou,
            "epoch_time_sec": epoch_time,
            "fusion_gates": gates_val,
            "peak_vram_mb": peak_vram,
            "val_map": cur_map,
            "val_map50": val_metrics["map50"],
            "val_map75": val_metrics["map75"],
            "val_map_small": val_metrics.get("map_small", 0.0),
            "val_map_medium": val_metrics.get("map_medium", 0.0),
            "val_map_large": val_metrics.get("map_large", 0.0),
            "per_category": val_metrics["per_category"],
        }
        history.append(epoch_record)

        # Lưu checkpoint kỷ lục
        if cur_map > best_map:
            best_map = cur_map
            best_path = WEIGHTS_DIR / "method8_best.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_map": best_map,
                "history": history,
            }, best_path)
            print(f"🔥 KỶ LỤC MỚI! Đã lưu trọng số tốt nhất vào: {best_path.name} (mAP = {best_map*100:.2f}%)")

        # Lưu checkpoint epoch hiện tại
        epoch_path = WEIGHTS_DIR / f"method8_epoch_{epoch}.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "val_map": cur_map,
            "history": history,
        }, epoch_path)

        with open(REPORT_DIR / "train_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print("\n" + "=" * 80)
    print(f"🎉 HOÀN THÀNH HUẤN LUYỆN METHOD 8! Best Val mAP: {best_map*100:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
