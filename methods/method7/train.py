"""Huấn Luyện Chính Thức Method 7: Spatial Sidecar Heatmap RT-DETR (Trung Thực, Không Leak Nhãn).

Các cải tiến cốt lõi so với Method 6:
1. 100% Không Leak Nhãn: Heatmap 22 kênh được sinh từ dự đoán của Base RT-DETRv2 (score >= 0.3).
   - Book Prior được tính qua cửa sổ 64 trang nhưng LOẠI TRỪ trang hiện tại.
   - Page Prior lấy từ dự đoán của trang hiện tại.
2. Tiền xử lý ảnh chuẩn xác theo Preprocessor của Heron (không normalize ImageNet sai lệch).
3. Tập Val hoàn toàn độc lập, tách theo CUỐN SÁCH từ tập train (không đụng vào test.json).
4. Cosine Annealing Learning Rate Scheduler kèm Linear Warmup.
5. Lưu đầy đủ optimizer_state_dict và scheduler_state_dict để resume không bị sốc động lượng.
6. Đánh giá COCO mAP chuẩn bằng pycocotools (top-100 detections, không lọc threshold).
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

from methods.method7.common import (
    BASE_MODEL, CACHE_DIR, REPORT_DIR, TRAIN_ANNO, TRAIN_IMG, WEIGHTS_DIR,
    load_coco, split_train_val_books,
)
from methods.method7.dataset import Method7Dataset, collate
from methods.method7.models.rtdetr_method7 import SpatialSidecarRTDetrMethod7
from methods.method7.evaluator import evaluate_method7_coco_map


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện Method 7 (No Label Leakage)")
    parser.add_argument("--epochs", type=int, default=5, help="Số epoch huấn luyện")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size trên mỗi GPU step")
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="Tích lũy gradient (effective batch size = batch_size * grad_accum)")
    parser.add_argument("--lr_sidecar", type=float, default=1e-4, help="Learning rate cho Sidecar CNN")
    parser.add_argument("--lr_rtdetr", type=float, default=1e-5, help="Learning rate cho RT-DETR encoder/decoder")
    parser.add_argument("--lr_gates", type=float, default=5e-3, help="Learning rate cho Fusion Gates")
    parser.add_argument("--warmup_epochs", type=int, default=1, help="Số epoch linear warmup")
    parser.add_argument("--num_workers", type=int, default=2, help="Số worker cho DataLoader")
    parser.add_argument("--freeze_backbone", action="store_true", default=True, help="Đóng băng backbone ResNet/HGNet")
    parser.add_argument("--resume", type=str, default="", help="Đường dẫn checkpoint để nạp tiếp tục")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN METHOD 7: SPATIAL SIDECAR RT-DETR (TRUNG THỰC - KHÔNG LEAK NHÃN)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Thiết bị: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Nạp annotation & Tách Train/Val theo CUỐN SÁCH
    print(f"[Dataset] Nạp annotation tập train từ: {TRAIN_ANNO}")
    train_coco = load_coco(TRAIN_ANNO)
    train_books, val_books = split_train_val_books(train_coco)
    print(f"[Split] Tách theo sách: {len(train_books):,} sách Train | {len(val_books):,} sách Val (độc lập, không dùng test.json)")

    train_ds = Method7Dataset(train_coco, TRAIN_IMG, split="train", allowed_books=train_books, drop_empty=True)
    val_ds = Method7Dataset(train_coco, TRAIN_IMG, split="train", allowed_books=val_books, drop_empty=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )

    print(f"[DataLoader] Train: {len(train_ds):,} trang ({len(train_loader)} steps/ep) | Val: {len(val_ds):,} trang ({len(val_loader)} steps).")

    # 2. Khởi tạo mô hình
    model = SpatialSidecarRTDetrMethod7(
        pretrained_model_path=BASE_MODEL,
        num_classes=17,
        heatmap_channels=22,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    # 3. Parameter Groups & Optimizer
    param_groups = [
        {"params": model.sidecar_cnn.parameters(), "lr": args.lr_sidecar, "weight_decay": 1e-4},
        {"params": [model.fusion_gates], "lr": args.lr_gates, "weight_decay": 0.0},
        {"params": [p for n, p in model.rtdetr.named_parameters() if p.requires_grad], "lr": args.lr_rtdetr, "weight_decay": 1e-4},
    ]
    optimizer = torch.optim.AdamW(param_groups)

    # Scheduler: Linear Warmup + Cosine Annealing
    total_steps = len(train_loader) * args.epochs // args.grad_accum_steps
    warmup_steps = len(train_loader) * args.warmup_epochs // args.grad_accum_steps

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch = 1
    best_map = 0.0
    history = []

    # 4. Resume nếu có checkpoint
    if args.resume and Path(args.resume).exists():
        print(f"[Resume] Nạp checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("  ✓ Đã nạp lại optimizer_state_dict (bảo toàn động lượng)")
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        if "val_res" in ckpt:
            best_map = ckpt["val_res"].get("map", 0.0)
        print(f"[Resume] Tiếp tục từ Epoch {start_epoch} (Best mAP: {best_map*100:.2f}%)")

    # 5. Đánh giá Baseline trước khi train (Epoch 0)
    print("\n--- ĐÁNH GIÁ BASELINE TRƯỚC HUẤN LUYỆN (EPOCH 0 TRÊN TẬP VAL CHUẨN) ---")
    val_res_0 = evaluate_method7_coco_map(model, val_loader, train_coco, device)
    print(f"✓ Epoch 0 Baseline: mAP@[.50:.95]={val_res_0['map']*100:.2f}% | AP50={val_res_0['map50']*100:.2f}% | AP75={val_res_0['map75']*100:.2f}%")
    print(f"  Fusion Gates ban đầu: {model.fusion_gates.detach().cpu().tolist()}")
    best_map = val_res_0["map"]

    # 6. Huấn luyện các Epochs
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    print(f"[AMP] Chế độ tự động hỗn hợp: {amp_dtype}")

    global_step = 0
    for epoch in range(start_epoch, start_epoch + args.epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        valid_batches = 0
        optimizer.zero_grad()

        print(f"\n>>> BẮT ĐẦU EPOCH {epoch}/{start_epoch + args.epochs - 1}")
        for step, batch in enumerate(train_loader):
            pixel_values, layout_heatmaps, targets = batch
            pixel_values = pixel_values.to(device, non_blocking=True)
            layout_heatmaps = layout_heatmaps.to(device, non_blocking=True)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            try:
                with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=amp_dtype):
                    out = model(pixel_values, layout_heatmaps, labels=targets)
                    loss = out["loss"] / args.grad_accum_steps

                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad()
                    continue

                loss.backward()
                total_loss += loss.item() * args.grad_accum_steps
                valid_batches += 1

                if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

            except Exception as e:
                optimizer.zero_grad()
                continue

            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                gates = model.fusion_gates.detach().cpu().tolist()
                current_lr = scheduler.get_last_lr()[0]
                print(f"  [Ep {epoch}] Step {step+1:04d}/{len(train_loader):04d} | Loss: {loss.item()*args.grad_accum_steps:.4f} | LR: {current_lr:.2e} | Gates: [{gates[0]:+.4f}, {gates[1]:+.4f}, {gates[2]:+.4f}]")

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / max(1, valid_batches)

        # Đánh giá sau epoch
        print(f"\n--- ĐÁNH GIÁ SAU EPOCH {epoch} ({epoch_time:.1f}s) TRÊN TẬP VAL CHUẨN ---")
        val_res = evaluate_method7_coco_map(model, val_loader, train_coco, device)
        gates = model.fusion_gates.detach().cpu().tolist()
        delta_map = (val_res["map"] - val_res_0["map"]) * 100

        print(f"✓ Kết quả Epoch {epoch}:")
        print(f"  - Hungarian Loss:     {avg_loss:.4f}")
        print(f"  - COCO mAP@[.50:.95]: {val_res['map']*100:.2f}% (Thay đổi vs Base: {delta_map:+.2f}%)")
        print(f"  - COCO AP50:          {val_res['map50']*100:.2f}%")
        print(f"  - COCO AP75:          {val_res['map75']*100:.2f}%")
        print(f"  - Fusion Gates:       P3={gates[0]:+.4f}, P4={gates[1]:+.4f}, P5={gates[2]:+.4f}")

        # Lưu checkpoint epoch
        epoch_ckpt = WEIGHTS_DIR / f"method7_epoch_{epoch}.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "fusion_gates": gates,
            "val_res": val_res,
        }, epoch_ckpt)

        if val_res["map"] > best_map:
            best_map = val_res["map"]
            best_ckpt = WEIGHTS_DIR / "method7_best.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "fusion_gates": gates,
                "val_res": val_res,
            }, best_ckpt)
            print(f"  🔥 KỶ LỤC MỚI! Đã lưu checkpoint tại: {best_ckpt.name}")

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_map": val_res["map"],
            "val_map50": val_res["map50"],
            "val_map75": val_res["map75"],
            "gates": gates,
            "time_sec": epoch_time,
        })

        with open(REPORT_DIR / "train_history.json", "w", encoding="utf-8") as f:
            json.dump({
                "baseline_epoch0": val_res_0,
                "best_map": best_map,
                "history": history,
            }, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}\n🎉 HOÀN TẤT HUẤN LUYỆN METHOD 7! BEST VAL mAP: {best_map*100:.2f}%\n{'='*80}")


if __name__ == "__main__":
    main()
