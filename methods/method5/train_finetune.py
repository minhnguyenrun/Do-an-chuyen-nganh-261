"""Fine-tuning Method 5 trên TOÀN BỘ tập dữ liệu Train DocLayNet (328 sách, 10,604 trang).

Khởi đầu từ checkpoint tốt nhất của Phase 1: method5_best_phase1_epoch3.pth (mAP 73.41% / AP75 79.17%).

Cấu hình tối ưu:
- Tập Train: Toàn bộ 328 cuốn sách (10,604 trang) để triệt tiêu hoàn toàn hiện tượng Overfitting.
- Tập Val theo dõi từng epoch: 10 sách (221 trang) giúp kiểm tra mAP nhanh chỉ trong ~20s.
- Learning Rate tinh chỉnh nhẹ: Fusion LR = 5e-5, Base RT-DETR LR = 5e-6 (giữ ổn định tri thức đã học).
- Cấu hình VRAM an toàn: Batch Size = 4, Grad Accum = 2, AMP FP16 -> VRAM chỉ 4.5 GB (không tràn RAM).
- Lưu checkpoint riêng biệt: finetune_best.pth, finetune_epoch_{epoch}.pth.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# Force UTF-8 stdout
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from methods.method5.models.rtdetr_method5 import BookLayoutRTDetrMethod5
from methods.method5.data.doclaynet_book_dataset import DocLayNetBookDataset, collate_book_fn
from methods.method5.models.evaluator import evaluate_coco_map, DOCLAYNET_CATEGORIES


def train_finetune(args):
    print("=" * 80)
    print("🚀 BẮT ĐẦU FINE-TUNING METHOD 5 TRÊN TOÀN BỘ TẬP TRAIN (10,604 TRANG)")
    print("=" * 80)

    device = torch.device(args.device)
    print(f"[Device] Thiết bị: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Dataset & DataLoader
    train_anno = ROOT_DIR / "datasets" / "doclaynet" / "processed" / "annotations" / "train.json"
    train_img = ROOT_DIR / "datasets" / "doclaynet" / "processed" / "images"
    val_anno = ROOT_DIR / "datasets" / "doclaynet_test" / "processed" / "annotations" / "test.json"
    val_img = ROOT_DIR / "datasets" / "doclaynet_test" / "processed" / "images"

    # Val set: dùng 10 sách đã chọn để val nhanh sau mỗi epoch
    selected_books_file = ROOT_DIR / "methods" / "method5" / "data" / "selected_books.json"
    val_allowed_books = None
    if selected_books_file.exists():
        with open(selected_books_file, "r", encoding="utf-8") as f:
            val_allowed_books = json.load(f).get("val_books")

    print("[Dataset] Đang tải TOÀN BỘ tập Train DocLayNet (không giới hạn sách)...")
    train_dataset = DocLayNetBookDataset(
        annotation_file=train_anno,
        images_dir=train_img,
        image_size=640,
        window_size=args.window_size,
        is_train=True,
        allowed_books=None,  # Quét toàn bộ 328 sách
    )

    val_dataset = DocLayNetBookDataset(
        annotation_file=val_anno,
        images_dir=val_img,
        image_size=640,
        window_size=args.window_size,
        is_train=False,
        allowed_books=val_allowed_books,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_book_fn,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_book_fn,
        num_workers=0,
    )

    print(f"[Dataloader] Train: {len(train_dataset):,} trang ({len(train_loader)} batches), Val: {len(val_dataset)} trang ({len(val_loader)} batches).")

    # 2. Khởi tạo mô hình & Nạp checkpoint tốt nhất Phase 1
    model = BookLayoutRTDetrMethod5(
        num_classes=17,
        num_summary_tokens=args.num_summary_tokens,
        freeze_backbone=False,
    ).to(device)

    init_ckpt_path = Path(args.init_checkpoint)
    if init_ckpt_path.exists():
        print(f"[Checkpoint] Nạp trọng số khởi đầu từ: {init_ckpt_path}")
        ckpt = torch.load(init_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print("   ✅ Đã nạp thành công trọng số Phase 1.")
    else:
        print(f"[Cảnh báo] Không tìm thấy file {init_ckpt_path}, dùng pretrained mặc định.")

    # 3. Differential Learning Rate cho Fine-tuning
    rtdetr_lr = args.lr * args.base_lr_factor
    print(f"[Optimization] Fine-tuning LR: Base RT-DETR: {rtdetr_lr:.2e} | Fusion Modules: {args.lr:.2e}")
    param_groups = [
        {"params": [p for p in model.rtdetr.parameters() if p.requires_grad], "lr": rtdetr_lr},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("rtdetr") and p.requires_grad], "lr": args.lr},
    ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("[Optimization] Kích hoạt Automatic Mixed Precision (AMP FP16).")

    output_dir = ROOT_DIR / "methods" / "method5"
    weights_dir = output_dir / "weights"
    report_dir = output_dir / "report"
    weights_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_vfl_loss": [],
        "train_bbox_loss": [],
        "val_map": [],
        "val_map50": [],
        "val_map75": [],
        "lr": [],
        "epoch_time_sec": [],
    }

    best_val_map = -1.0
    accum_steps = max(1, args.grad_accum_steps)

    start_epoch = 1
    if args.resume and Path(args.resume).exists():
        print(f"[Resume] Nạp finetune checkpoint từ: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint and optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_map = checkpoint.get("val_map", -1.0)

        hist_file = report_dir / "finetune_history.json"
        if hist_file.exists():
            try:
                with open(hist_file, "r", encoding="utf-8") as f:
                    old_hist = json.load(f)
                    for k in history:
                        if k in old_hist:
                            history[k] = old_hist[k]
                if history.get("val_map"):
                    best_val_map = max(history["val_map"])
            except Exception as e:
                print(f"[Resume] Cảnh báo khi nạp finetune_history.json: {e}")

        for _ in range(start_epoch - 1):
            scheduler.step()
        print(f"[Resume] Tiếp tục Fine-tuning từ Epoch {start_epoch} (Best mAP: {best_val_map*100:.2f}%)")

    # 4. Vòng lặp huấn luyện
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_train_loss = 0.0
        total_vfl_loss = 0.0
        total_bbox_loss = 0.0
        batch_count = 0

        current_lr = optimizer.param_groups[-1]["lr"]
        print(f"\n==================== Fine-tuning Epoch [{epoch:02d}/{args.epochs:02d}] (Fusion LR: {current_lr:.2e}) ====================")

        optimizer.zero_grad()

        for b_idx, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in l.items()} for l in batch["labels"]]
            context_elements = batch["context_elements"]

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(
                    pixel_values=pixel_values,
                    labels=labels,
                    context_elements=context_elements,
                )
                loss = outputs["loss"]
                loss_dict = outputs["loss_dict"]
                loss_scaled = loss / accum_steps

            # Backward & Step with gradient accumulation
            scaler.scale(loss_scaled).backward()

            if (b_idx + 1) % accum_steps == 0 or (b_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                all_trainable_p = [p for pg in param_groups for p in pg["params"]]
                torch.nn.utils.clip_grad_norm_(all_trainable_p, max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            loss_val = loss.item()
            vfl_val = loss_dict.get("loss_vfl", torch.tensor(0.0)).item()
            bbox_val = (loss_dict.get("loss_bbox", torch.tensor(0.0)) + loss_dict.get("loss_giou", torch.tensor(0.0))).item()

            total_train_loss += loss_val
            total_vfl_loss += vfl_val
            total_bbox_loss += bbox_val
            batch_count += 1

            if (b_idx + 1) % args.log_interval == 0 or (b_idx + 1) == len(train_loader):
                avg_l = total_train_loss / batch_count
                avg_vfl = total_vfl_loss / batch_count
                avg_b = total_bbox_loss / batch_count
                print(
                    f"   Step [{b_idx+1:04d}/{len(train_loader):04d}] "
                    f"Loss: {avg_l:.4f} (VFL: {avg_vfl:.4f}, BBox: {avg_b:.4f})"
                )

        scheduler.step()
        epoch_time = time.time() - t0

        avg_epoch_loss = total_train_loss / max(1, batch_count)
        avg_epoch_vfl = total_vfl_loss / max(1, batch_count)
        avg_epoch_box = total_bbox_loss / max(1, batch_count)

        # 5. Đánh giá Validation COCO mAP@[0.50:0.95]
        print(f"\n📊 Đang tính toán COCO mAP trên 10 sách kiểm định ({len(val_dataset)} trang)...")
        eval_t0 = time.time()
        val_res = evaluate_coco_map(model, val_loader, device=device, conf_thresh=0.2, use_fusion=True)
        eval_time = time.time() - eval_t0

        val_map = val_res["map"]
        val_map50 = val_res["map50"]
        val_map75 = val_res["map75"]

        print(
            f"   ▶ Validation Results: "
            f"mAP@[.50:.95]: {val_map * 100:.2f}% | "
            f"AP50: {val_map50 * 100:.2f}% | "
            f"AP75: {val_map75 * 100:.2f}% "
            f"({eval_time:.1f}s)"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 6. Ghi lại lịch sử
        history["epoch"].append(epoch)
        history["train_loss"].append(round(avg_epoch_loss, 4))
        history["train_vfl_loss"].append(round(avg_epoch_vfl, 4))
        history["train_bbox_loss"].append(round(avg_epoch_box, 4))
        history["val_map"].append(round(val_map, 4))
        history["val_map50"].append(round(val_map50, 4))
        history["val_map75"].append(round(val_map75, 4))
        history["lr"].append(current_lr)
        history["epoch_time_sec"].append(round(epoch_time, 1))

        # Lưu checkpoint
        latest_path = weights_dir / "finetune_latest.pth"
        epoch_path = weights_dir / f"finetune_epoch_{epoch}.pth"
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_map": val_map,
            "val_map50": val_map50,
            "val_map75": val_map75,
            "train_loss": avg_epoch_loss,
        }
        torch.save(ckpt_data, latest_path)
        torch.save(ckpt_data, epoch_path)

        if val_map > best_val_map:
            best_val_map = val_map
            best_path = weights_dir / "finetune_best.pth"
            torch.save(ckpt_data, best_path)
            print(f"   🌟 ĐÃ LƯU BEST FINETUNE MODEL: mAP đạt đỉnh mới: {val_map * 100:.2f}% (Epoch {epoch})")

        # Lưu JSON history
        with open(report_dir / "finetune_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print(f"\n📁 Đã hoàn tất Fine-tuning! Checkpoint lưu tại: {weights_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Method 5 on Full DocLayNet")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate tinh chỉnh")
    parser.add_argument("--base-lr-factor", type=float, default=0.1, help="Hệ số scale LR cho Base RT-DETR (5e-6)")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--num-summary-tokens", type=int, default=8)
    parser.add_argument("--init-checkpoint", type=str, default="methods/method5/weights/method5_best_phase1_epoch3.pth")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, default="", help="Checkpoint để tiếp tục fine-tuning")
    args = parser.parse_args()
    train_finetune(args)
