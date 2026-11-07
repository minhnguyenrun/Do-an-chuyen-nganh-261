"""Huấn Luyện Method 5: 64-Page Book-Layout RT-DETR.

Cấu hình tối ưu:
- Early & Late Book Context & OCR Semantic Fusion (Queries bên trong Decoder được giữ thuần khiết).
- Class-Weighted Varifocal Loss (CW-VFL) khắc phục triệt để mất cân bằng Long-Tail.
- Không đóng băng RT-DETR (Unfreeze) với Differential Learning Rate (Backbone/Decoder: LR 1.5e-5, Fusion: LR 1.5e-4).
- Tuyển chọn 50 sách Train (1,669 trang) & 10 sách Val (221 trang) đa dạng độ dài và phong phú nhãn.
- Tự động đánh giá trên toàn bộ tập Validation sau khi huấn luyện xong.
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


def train_method5(args):
    print("=" * 80)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN METHOD 5 (CLASS-WEIGHTED VFL & UNFROZEN RT-DETR)")
    print("=" * 80)

    device = torch.device(args.device)
    print(f"[Device] Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Dataset & DataLoader
    train_anno = ROOT_DIR / "datasets" / "doclaynet" / "processed" / "annotations" / "train.json"
    train_img = ROOT_DIR / "datasets" / "doclaynet" / "processed" / "images"
    val_anno = ROOT_DIR / "datasets" / "doclaynet_test" / "processed" / "annotations" / "test.json"
    val_img = ROOT_DIR / "datasets" / "doclaynet_test" / "processed" / "images"

    selected_books_file = ROOT_DIR / "methods" / "method5" / "data" / "selected_books.json"
    train_allowed_books = None
    val_allowed_books = None

    if args.use_selected_books and selected_books_file.exists():
        with open(selected_books_file, "r", encoding="utf-8") as f:
            sb_data = json.load(f)
        train_allowed_books = sb_data.get("train_books")
        val_allowed_books = sb_data.get("val_books")
        print(f"[Dataset] Sử dụng {len(train_allowed_books)} sách Train & {len(val_allowed_books)} sách Val đã tuyển chọn.")

    train_dataset = DocLayNetBookDataset(
        annotation_file=train_anno,
        images_dir=train_img,
        image_size=640,
        window_size=args.window_size,
        limit_pages=args.limit_train_pages if args.limit_train_pages > 0 else None,
        is_train=True,
        allowed_books=train_allowed_books,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_book_fn,
        num_workers=0,
    )

    val_dataset = DocLayNetBookDataset(
        annotation_file=val_anno,
        images_dir=val_img,
        image_size=640,
        window_size=args.window_size,
        limit_pages=args.limit_val_pages if args.limit_val_pages > 0 else None,
        is_train=False,
        allowed_books=val_allowed_books,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_book_fn,
        num_workers=0,
    )

    print(f"[Dataloader] Train: {len(train_dataset):,} trang ({len(train_loader)} batches), Val: {len(val_dataset):,} trang ({len(val_loader)} batches).")

    # 2. Khởi tạo Mô hình
    model = BookLayoutRTDetrMethod5(
        pretrained_model_path=None,
        num_classes=17,
        num_summary_tokens=args.num_summary_tokens,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    # 3. Differential Learning Rate
    if not args.freeze_backbone:
        rtdetr_lr = args.lr * args.base_lr_factor
        print(f"[Optimization] Differential LR: Base RT-DETR: {rtdetr_lr:.2e} | Fusion Modules: {args.lr:.2e}")
        param_groups = [
            {"params": [p for p in model.rtdetr.parameters() if p.requires_grad], "lr": rtdetr_lr},
            {"params": [p for n, p in model.named_parameters() if not n.startswith("rtdetr") and p.requires_grad], "lr": args.lr},
        ]
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        param_groups = [{"params": trainable_params, "lr": args.lr}]

    total_trainable = sum(p.numel() for pg in param_groups for p in pg["params"])
    print(f"[Model] Trainable Parameters: {total_trainable:,}")

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

    start_epoch = 1
    if args.resume and Path(args.resume).exists():
        print(f"[Resume] Nạp checkpoint từ: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint and optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        # Nạp lại lịch sử training cũ nếu có
        hist_file = report_dir / "training_history.json"
        if hist_file.exists():
            try:
                with open(hist_file, "r", encoding="utf-8") as f:
                    old_hist = json.load(f)
                    for k in history:
                        if k in old_hist:
                            history[k] = old_hist[k]
                print(f"[Resume] Đã nạp lịch sử huấn luyện từ {len(history['epoch'])} epoch trước.")
                if history.get("val_map"):
                    best_val_map = max(history["val_map"])
            except Exception as e:
                print(f"[Resume] Cảnh báo khi nạp training_history.json: {e}")

        print(f"[Resume] Tiếp tục từ Epoch {start_epoch} (Best mAP lịch sử: {best_val_map*100:.2f}%)")

        # Đồng bộ bước học của scheduler
        for _ in range(start_epoch - 1):
            scheduler.step()

    # 4. Vòng lặp huấn luyện
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_train_loss = 0.0
        total_vfl_loss = 0.0
        total_bbox_loss = 0.0
        batch_count = 0

        current_lr = optimizer.param_groups[-1]["lr"]
        print(f"\n==================== Epoch [{epoch:02d}/{args.epochs:02d}] (Fusion LR: {current_lr:.2e}) ====================")

        optimizer.zero_grad()
        accum_steps = max(1, getattr(args, "grad_accum_steps", 1))

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
        print(f"\n📊 Đang tính toán COCO mAP@[0.50:0.95] trên tập kiểm định ({len(val_dataset)} trang)...")
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

        # Lưu checkpoint latest và epoch checkpoint
        latest_path = weights_dir / "method5_latest.pth"
        epoch_path = weights_dir / f"method5_epoch_{epoch}.pth"
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_map": val_map,
            "val_map50": val_map50,
            "train_loss": avg_epoch_loss,
        }
        torch.save(ckpt_data, latest_path)
        torch.save(ckpt_data, epoch_path)

        # Lưu best checkpoint theo mAP@[0.50:0.95]
        if val_map > best_val_map:
            best_val_map = val_map
            best_path = weights_dir / "method5_best.pth"
            torch.save(ckpt_data, best_path)
            print(f"   🌟 ĐÃ LƯU BEST MODEL: mAP@[.50:.95] đạt đỉnh mới: {val_map * 100:.2f}% (Epoch {epoch})")

        # Lưu CSV history
        csv_file = report_dir / "training_history.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_vfl_loss", "train_bbox_loss", "val_map", "val_map50", "val_map75", "lr", "epoch_time_sec"])
            for i in range(len(history["epoch"])):
                writer.writerow([
                    history["epoch"][i],
                    history["train_loss"][i],
                    history["train_vfl_loss"][i],
                    history["train_bbox_loss"][i],
                    history["val_map"][i],
                    history["val_map50"][i],
                    history["val_map75"][i],
                    history["lr"][i],
                    history["epoch_time_sec"][i],
                ])

        with open(report_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    # 7. Xuất biểu đồ training report
    _plot_report(history, report_dir / "training_report.png")
    print(f"\n📁 Đã hoàn tất huấn luyện! Lịch sử và biểu đồ lưu tại: {report_dir}")

    # 8. ĐÁNH GIÁ CHÍNH THỨC TRÊN TOÀN BỘ TẬP VALIDATION (TEST SET)
    if args.evaluate_full_val:
        print("\n" + "=" * 80)
        print("🏁 ĐÁNH GIÁ ĐỐI CHUẨN CHÍNH THỨC TRÊN TOÀN BỘ TẬP TEST (FULL TEST SET)")
        print("=" * 80)
        # Nạp best weights
        if (weights_dir / "method5_best.pth").exists():
            best_ckpt = torch.load(weights_dir / "method5_best.pth", map_location=device)
            model.load_state_dict(best_ckpt["model_state_dict"])
            print("✓ Đã nạp best checkpoint method5_best.pth")

        full_val_dataset = DocLayNetBookDataset(
            annotation_file=val_anno,
            images_dir=val_img,
            image_size=640,
            window_size=args.window_size,
            is_train=False,
            allowed_books=None,  # Quét toàn bộ tập test
        )
        full_val_loader = DataLoader(
            full_val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_book_fn,
            num_workers=0,
        )
        print(f"[Full Eval] Đang đánh giá trên toàn bộ {len(full_val_dataset):,} trang kiểm thử...")
        full_res = evaluate_coco_map(model, full_val_loader, device=device, conf_thresh=0.2, use_fusion=True)

        print("\n" + "=" * 80)
        print(f"{'Category Name':<18} | {'Method 5 AP@[.50:.95]':<22}")
        print("-" * 80)
        for cat in DOCLAYNET_CATEGORIES:
            cid = cat["id"]
            cname = cat["name"]
            c_ap = full_res["per_category"].get(cid, {}).get("ap", 0.0)
            print(f"{cname:<18} | {c_ap*100:>20.2f}%")
        print("-" * 80)
        print(f"{'OVERALL mAP@[.50:.95]':<18} | {full_res['map']*100:>20.2f}%")
        print(f"{'OVERALL AP50':<18} | {full_res['map50']*100:>20.2f}%")
        print("=" * 80)

        cleaned_per_category = {}
        for cid, metrics in full_res.get("per_category", {}).items():
            cleaned_per_category[str(cid)] = {k: float(v) for k, v in metrics.items()}

        with open(report_dir / "benchmark_full_val.json", "w", encoding="utf-8") as f:
            json.dump({
                "map": round(float(full_res["map"]), 4),
                "map50": round(float(full_res["map50"]), 4),
                "map75": round(float(full_res["map75"]), 4),
                "per_category": cleaned_per_category,
            }, f, indent=2, ensure_ascii=False)
        print(f"📁 Đã lưu kết quả đánh giá toàn bộ tập Val tại: {report_dir / 'benchmark_full_val.json'}")


def _plot_report(history: dict, save_path: Path):
    epochs = history["epoch"]
    if len(epochs) == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(epochs, history["train_loss"], "o-", label="Total Hungarian Loss", color="#1f77b4", linewidth=2)
    ax1.plot(epochs, history["train_vfl_loss"], "^-.", label="Class-Weighted VFL", color="#2ca02c")
    ax1.plot(epochs, history["train_bbox_loss"], "s--", label="BBox L1+GIoU Loss", color="#d62728")
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Loss", fontsize=11, fontweight="bold")
    ax1.set_title("Method 5: Hungarian Loss Convergence", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10)

    # COCO mAP curve
    ax2.plot(epochs, [m * 100 for m in history["val_map"]], "o-", label="mAP@[.50:.95]", color="#9467bd", linewidth=2)
    ax2.plot(epochs, [m * 100 for m in history["val_map50"]], "s--", label="AP50", color="#ff7f0e", linewidth=2)
    ax2.plot(epochs, [m * 100 for m in history["val_map75"]], "d-.", label="AP75", color="#2ca02c", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax2.set_ylabel("COCO Metric (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Validation COCO mAP@[0.50:0.95]", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Method 5: Book-Layout RT-DETR")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Số bước tích lũy gradient để giảm VRAM")
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--base-lr-factor", type=float, default=0.1, help="Hệ số scale LR cho Base RT-DETR (1.5e-5)")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--window-size", type=int, default=64, help="Cửa sổ nhìn nhận số trang sách")
    parser.add_argument("--num-summary-tokens", type=int, default=8)
    parser.add_argument("--limit-train-pages", type=int, default=0)
    parser.add_argument("--limit-val-pages", type=int, default=0)
    parser.add_argument("--freeze-backbone", action="store_true", default=False, help="Mặc định False: Không đóng băng RT-DETR")
    parser.add_argument("--use-selected-books", action="store_true", default=True, help="Train trên 50 sách & val trên 10 sách")
    parser.add_argument("--evaluate-full-val", action="store_true", default=True, help="Đánh giá trên toàn bộ tập test sau khi train xong")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, default="", help="Đường dẫn file checkpoint .pth để tiếp tục train")
    args = parser.parse_args()
    train_method5(args)
