"""Đánh giá toàn diện Method 5 trên TOÀN BỘ tập Validation / Test của DocLayNet.
Tính toán mAP@[0.50:0.95], AP50, AP75 và per-category AP trên 4,999 trang.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Force UTF-8 stdout
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from methods.method5.models.rtdetr_method5 import BookLayoutRTDetrMethod5
from methods.method5.data.doclaynet_book_dataset import DocLayNetBookDataset, collate_book_fn
from methods.method5.models.evaluator import evaluate_coco_map, DOCLAYNET_CATEGORIES


def evaluate_full():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="methods/method5/weights/method5_best.pth")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--conf-thresh", type=float, default=0.2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 80)
    print(f"🏁 ĐÁNH GIÁ CHÍNH THỨC METHOD 5 TRÊN TOÀN BỘ TẬP VALIDATION (DOCLAYNET TEST SET)")
    print(f"   Trọng số: {args.weights}")
    print(f"   Thiết bị: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"   Batch size: {args.batch_size} | Ngưỡng tin cậy (conf_thresh): {args.conf_thresh}")
    print("=" * 80)

    val_anno = ROOT_DIR / "datasets" / "doclaynet_test" / "processed" / "annotations" / "test.json"
    val_img = ROOT_DIR / "datasets" / "doclaynet_test" / "processed" / "images"

    print("[Dataset] Đang tải toàn bộ tập kiểm định...")
    val_dataset = DocLayNetBookDataset(
        annotation_file=val_anno,
        images_dir=val_img,
        image_size=640,
        window_size=args.window_size,
        is_train=False,
        allowed_books=None,  # Quét 100% toàn bộ các cuốn sách
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_book_fn,
        num_workers=0,
    )
    print(f"[Dataset] Đã nạp thành công {len(val_dataset):,} trang ({len(val_loader)} batches).")

    # Nạp mô hình
    print("\n[Model] Đang nạp kiến trúc Method 5...")
    model = BookLayoutRTDetrMethod5(num_classes=17, num_summary_tokens=8).to(device)

    weights_path = Path(args.weights)
    if weights_path.exists():
        print(f"[Model] Nạp trọng số từ: {weights_path} ({weights_path.stat().st_size / (1024*1024):.1f} MB)")
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    else:
        print(f"[Model] ⚠️ Không tìm thấy file {weights_path}")
        return

    model.eval()

    print(f"\n⏳ Bắt đầu suy luận (Inference) trên {len(val_dataset):,} trang...")
    t0 = time.time()
    res = evaluate_coco_map(
        model=model,
        dataloader=val_loader,
        device=device,
        conf_thresh=args.conf_thresh,
        use_fusion=True,
    )
    total_time = time.time() - t0
    fps = len(val_dataset) / total_time

    print(f"\n✓ Hoàn tất đánh giá trong {total_time:.1f}s ({fps:.1f} trang/giây)!")

    # In kết quả
    print("\n" + "=" * 80)
    print(f"{'Category Name':<20} | {'AP@[.50:.95]':<16} | {'AP50':<16}")
    print("-" * 80)
    for cat in DOCLAYNET_CATEGORIES:
        cid = cat["id"]
        cname = cat["name"]
        cat_data = res["per_category"].get(cid, {})
        ap = cat_data.get("ap", 0.0)
        ap50 = cat_data.get("ap50", 0.0)
        ap_str = f"{ap*100:.2f}%" if ap >= 0 else "N/A"
        ap50_str = f"{ap50*100:.2f}%" if ap50 >= 0 else "N/A"
        print(f"{cname:<20} | {ap_str:>14} | {ap50_str:>14}")

    print("-" * 80)
    print(f"{'OVERALL mAP@[.50:.95]':<20} | {res['map']*100:>14.2f}% |")
    print(f"{'OVERALL AP50':<20} | {res['map50']*100:>14.2f}% |")
    print(f"{'OVERALL AP75':<20} | {res['map75']*100:>14.2f}% |")
    print("=" * 80)

    # Convert per_category keys and numpy floats to standard types
    cleaned_per_category = {}
    for cid, metrics in res.get("per_category", {}).items():
        cleaned_per_category[str(cid)] = {k: float(v) for k, v in metrics.items()}

    # Lưu kết quả
    report_file = ROOT_DIR / "methods" / "method5" / "report" / "full_validation_results.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_pages": len(val_dataset),
            "eval_time_sec": round(total_time, 1),
            "fps": round(fps, 1),
            "weights": str(args.weights),
            "map": round(float(res["map"]), 4),
            "map50": round(float(res["map50"]), 4),
            "map75": round(float(res["map75"]), 4),
            "per_category": cleaned_per_category,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Đã lưu kết quả chi tiết tại: {report_file}")


if __name__ == "__main__":
    evaluate_full()
