"""Script so sánh chi tiết hiệu năng giữa:
1. Pretrained Baseline RT-DETRv2 gốc (docling-project/docling-layout-heron)
2. Method 5 Best Model (64-Page Book-Layout RT-DETR với Dual-Stage Fusion & CW-VFL)
trên tập kiểm thử tuyển chọn 10 cuốn sách (221 trang).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Force UTF-8 stdout
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import torch
from torch.utils.data import DataLoader
from transformers import RTDetrV2ForObjectDetection

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from methods.method5.models.rtdetr_method5 import BookLayoutRTDetrMethod5
from methods.method5.data.doclaynet_book_dataset import DocLayNetBookDataset, collate_book_fn
from methods.method5.models.evaluator import evaluate_coco_map, DOCLAYNET_CATEGORIES


def run_comparison():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("🔬 ĐÁNH GIÁ ĐỐI CHUẨN SO SÁNH: BASELINE RT-DETR vs METHOD 5 BEST MODEL")
    print(f"   Thiết bị: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)

    # 1. Dataset 10 sách Val
    val_anno = ROOT_DIR / "datasets" / "doclaynet_test" / "processed" / "annotations" / "test.json"
    val_img = ROOT_DIR / "datasets" / "doclaynet_test" / "processed" / "images"
    selected_books_file = ROOT_DIR / "methods" / "method5" / "data" / "selected_books.json"

    val_allowed_books = None
    if selected_books_file.exists():
        with open(selected_books_file, "r", encoding="utf-8") as f:
            val_allowed_books = json.load(f).get("val_books")

    val_dataset = DocLayNetBookDataset(
        annotation_file=val_anno,
        images_dir=val_img,
        image_size=640,
        window_size=64,
        is_train=False,
        allowed_books=val_allowed_books,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_book_fn,
        num_workers=0,
    )
    print(f"[Dataset] Đã nạp {len(val_dataset)} trang từ 10 cuốn sách kiểm định.")

    # 2. Đánh giá Model Gốc (Baseline Pretrained RT-DETRv2)
    print("\n" + "-" * 80)
    print("1️⃣ Đang đánh giá MODEL GỐC BAN ĐẦU (docling-project/docling-layout-heron)...")
    print("-" * 80)
    base_model = RTDetrV2ForObjectDetection.from_pretrained("docling-project/docling-layout-heron").to(device)
    base_model.eval()

    t0 = time.time()
    base_res = evaluate_coco_map(base_model, val_loader, device=device, conf_thresh=0.2, use_fusion=False)
    base_time = time.time() - t0
    print(f"   Hoàn tất đánh giá Base Model trong {base_time:.1f}s.")
    print(f"   Base Model mAP@[.50:.95]: {base_res['map']*100:.2f}% | AP50: {base_res['map50']*100:.2f}% | AP75: {base_res['map75']*100:.2f}%")

    # Dọn dẹp VRAM
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Đánh giá Method 5 Best Model
    best_weights_path = ROOT_DIR / "methods" / "method5" / "weights" / "method5_best.pth"
    print("\n" + "-" * 80)
    print(f"2️⃣ Đang đánh giá METHOD 5 BEST MODEL ({best_weights_path.name})...")
    print("-" * 80)

    best_ckpt = torch.load(best_weights_path, map_location=device, weights_only=False)
    best_epoch = best_ckpt.get("epoch", "?")
    print(f"   Checkpoint tương ứng với Epoch {best_epoch}.")

    m5_model = BookLayoutRTDetrMethod5(
        num_classes=17,
        num_summary_tokens=8,
        freeze_backbone=False,
    ).to(device)
    m5_model.load_state_dict(best_ckpt["model_state_dict"])
    m5_model.eval()

    t1 = time.time()
    m5_res = evaluate_coco_map(m5_model, val_loader, device=device, conf_thresh=0.2, use_fusion=True)
    m5_time = time.time() - t1
    print(f"   Hoàn tất đánh giá Method 5 trong {m5_time:.1f}s.")
    print(f"   Method 5 Best mAP@[.50:.95]: {m5_res['map']*100:.2f}% | AP50: {m5_res['map50']*100:.2f}% | AP75: {m5_res['map75']*100:.2f}%")

    # 4. In Bảng So Sánh Chi Tiết
    print("\n" + "=" * 80)
    print("📊 BẢNG ĐỐI CHUẨN SO SÁNH: MODEL GỐC vs METHOD 5 (BEST EPOCH)")
    print("=" * 80)
    print(f"{'Category Name':<16} | {'Base Model AP':<15} | {'Method 5 (Best) AP':<20} | {'Cải thiện (Delta)':<18}")
    print("-" * 80)

    per_category_comp = {}
    for cat in DOCLAYNET_CATEGORIES:
        cid = cat["id"]
        cname = cat["name"]
        base_ap = base_res["per_category"].get(cid, {}).get("ap", 0.0)
        m5_ap = m5_res["per_category"].get(cid, {}).get("ap", 0.0)
        delta = m5_ap - base_ap
        sign = "+" if delta >= 0 else ""
        delta_str = f"{sign}{delta*100:.2f}%"

        print(f"{cname:<16} | {base_ap*100:>13.2f}% | {m5_ap*100:>18.2f}% | {delta_str:>16}")
        per_category_comp[cname] = {
            "base_ap": round(float(base_ap), 4),
            "method5_ap": round(float(m5_ap), 4),
            "delta": round(float(delta), 4),
        }

    print("-" * 80)
    overall_delta_map = m5_res["map"] - base_res["map"]
    overall_delta_map50 = m5_res["map50"] - base_res["map50"]
    overall_delta_map75 = m5_res["map75"] - base_res["map75"]

    print(f"{'OVERALL mAP@[.50:.95]':<16} | {base_res['map']*100:>13.2f}% | {m5_res['map']*100:>18.2f}% | {'+' if overall_delta_map >= 0 else ''}{overall_delta_map*100:.2f}%")
    print(f"{'OVERALL AP50':<16} | {base_res['map50']*100:>13.2f}% | {m5_res['map50']*100:>18.2f}% | {'+' if overall_delta_map50 >= 0 else ''}{overall_delta_map50*100:.2f}%")
    print(f"{'OVERALL AP75':<16} | {base_res['map75']*100:>13.2f}% | {m5_res['map75']*100:>18.2f}% | {'+' if overall_delta_map75 >= 0 else ''}{overall_delta_map75*100:.2f}%")
    print("=" * 80)

    # 5. Lưu JSON kết quả so sánh
    out_dir = ROOT_DIR / "methods" / "method5" / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "baseline_vs_method5_comparison.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "eval_pages": len(val_dataset),
            "best_epoch": best_epoch,
            "base_model": {
                "name": "docling-project/docling-layout-heron",
                "map": round(float(base_res["map"]), 4),
                "map50": round(float(base_res["map50"]), 4),
                "map75": round(float(base_res["map75"]), 4),
            },
            "method5_best": {
                "weights": "method5_best.pth",
                "map": round(float(m5_res["map"]), 4),
                "map50": round(float(m5_res["map50"]), 4),
                "map75": round(float(m5_res["map75"]), 4),
            },
            "improvement": {
                "delta_map": round(float(overall_delta_map), 4),
                "delta_map50": round(float(overall_delta_map50), 4),
                "delta_map75": round(float(overall_delta_map75), 4),
            },
            "per_category": per_category_comp,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Đã lưu toàn bộ báo cáo so sánh tại: {out_json}")


if __name__ == "__main__":
    run_comparison()
