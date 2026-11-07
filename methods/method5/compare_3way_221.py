"""Script so sánh đối chuẩn 3 chiều trên tập kiểm định 221 trang (10 cuốn sách):
1. Model Gốc Ban Đầu (Pretrained Base RT-DETR - docling-project/docling-layout-heron)
2. Method 5 Phase 1 Best (Epoch 3: 50 sách, 1,669 trang)
3. Method 5 Phase 2 Fine-tune (Epoch 2: 328 sách, 10,604 trang)
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


def run_3way_comparison():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("🔬 ĐÁNH GIÁ ĐỐI CHUẨN 3 CHIỀU: BASELINE vs PHASE 1 vs PHASE 2 (10K TRANG)")
    print(f"   Thiết bị: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)

    # 1. Dataset 10 sách Val (221 trang)
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

    # -------------------------------------------------------------
    # 2. Đánh giá Model Gốc (Baseline Pretrained RT-DETRv2)
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("1️⃣ Đang đánh giá MODEL GỐC BAN ĐẦU (docling-project/docling-layout-heron)...")
    print("-" * 80)
    base_model = RTDetrV2ForObjectDetection.from_pretrained("docling-project/docling-layout-heron").to(device)
    base_model.eval()

    t0 = time.time()
    base_res = evaluate_coco_map(base_model, val_loader, device=device, conf_thresh=0.2, use_fusion=False)
    base_time = time.time() - t0
    print(f"   Hoàn tất Base Model trong {base_time:.1f}s.")
    print(f"   Base Model mAP@[.50:.95]: {base_res['map']*100:.2f}% | AP50: {base_res['map50']*100:.2f}% | AP75: {base_res['map75']*100:.2f}%")

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # 3. Đánh giá Method 5 Phase 1 Best (Epoch 3)
    # -------------------------------------------------------------
    p1_weights = ROOT_DIR / "methods" / "method5" / "weights" / "method5_best_phase1_epoch3.pth"
    print("\n" + "-" * 80)
    print(f"2️⃣ Đang đánh giá METHOD 5 PHASE 1 BEST (Epoch 3 - 50 sách)...")
    print("-" * 80)

    p1_model = BookLayoutRTDetrMethod5(
        num_classes=17,
        num_summary_tokens=8,
        freeze_backbone=False,
    ).to(device)
    ckpt1 = torch.load(p1_weights, map_location=device, weights_only=False)
    p1_model.load_state_dict(ckpt1["model_state_dict"])
    p1_model.eval()

    t1 = time.time()
    p1_res = evaluate_coco_map(p1_model, val_loader, device=device, conf_thresh=0.2, use_fusion=True)
    p1_time = time.time() - t1
    print(f"   Hoàn tất Phase 1 Best trong {p1_time:.1f}s.")
    print(f"   Phase 1 Best mAP@[.50:.95]: {p1_res['map']*100:.2f}% | AP50: {p1_res['map50']*100:.2f}% | AP75: {p1_res['map75']*100:.2f}%")

    del p1_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # 4. Đánh giá Method 5 Phase 2 Fine-tuned (Epoch 2)
    # -------------------------------------------------------------
    p2_weights = ROOT_DIR / "methods" / "method5" / "weights" / "finetune_epoch_2.pth"
    if not p2_weights.exists():
        p2_weights = ROOT_DIR / "methods" / "method5" / "weights" / "finetune_latest.pth"

    print("\n" + "-" * 80)
    print(f"3️⃣ Đang đánh giá METHOD 5 PHASE 2 (Fine-tune 10k trang - {p2_weights.name})...")
    print("-" * 80)

    p2_model = BookLayoutRTDetrMethod5(
        num_classes=17,
        num_summary_tokens=8,
        freeze_backbone=False,
    ).to(device)
    ckpt2 = torch.load(p2_weights, map_location=device, weights_only=False)
    p2_epoch = ckpt2.get("epoch", 2)
    p2_model.load_state_dict(ckpt2["model_state_dict"])
    p2_model.eval()

    t2 = time.time()
    p2_res = evaluate_coco_map(p2_model, val_loader, device=device, conf_thresh=0.2, use_fusion=True)
    p2_time = time.time() - t2
    print(f"   Hoàn tất Phase 2 (Epoch {p2_epoch}) trong {p2_time:.1f}s.")
    print(f"   Phase 2 mAP@[.50:.95]: {p2_res['map']*100:.2f}% | AP50: {p2_res['map50']*100:.2f}% | AP75: {p2_res['map75']*100:.2f}%")

    del p2_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # 5. Bảng So Sánh 3 Chiều Chi Tiết
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print("📊 BẢNG ĐỐI CHUẨN SO SÁNH 3 CHIỀU TRÊN 221 TRANG (10 CUỐN SÁCH)")
    print("=" * 90)
    print(f"{'Category Name':<16} | {'Base Model':<12} | {'Phase 1 (Epoch 3)':<18} | {'Phase 2 (Epoch 2)':<18} | {'Delta (P2 vs Base)':<18}")
    print("-" * 90)

    per_cat_comp = {}
    for cat in DOCLAYNET_CATEGORIES:
        cid = cat["id"]
        cname = cat["name"]
        b_ap = base_res["per_category"].get(cid, {}).get("ap", 0.0)
        p1_ap = p1_res["per_category"].get(cid, {}).get("ap", 0.0)
        p2_ap = p2_res["per_category"].get(cid, {}).get("ap", 0.0)
        delta_p2_base = p2_ap - b_ap
        sign = "+" if delta_p2_base >= 0 else ""

        print(f"{cname:<16} | {b_ap*100:>10.2f}% | {p1_ap*100:>16.2f}% | {p2_ap*100:>16.2f}% | {sign}{delta_p2_base*100:>15.2f}%")
        per_cat_comp[cname] = {
            "base": round(float(b_ap), 4),
            "phase1": round(float(p1_ap), 4),
            "phase2": round(float(p2_ap), 4),
            "delta_p2_vs_base": round(float(delta_p2_base), 4),
        }

    print("-" * 90)
    d_map = p2_res["map"] - base_res["map"]
    d_map50 = p2_res["map50"] - base_res["map50"]
    d_map75 = p2_res["map75"] - base_res["map75"]

    print(f"{'OVERALL mAP@[.50:.95]':<16} | {base_res['map']*100:>10.2f}% | {p1_res['map']*100:>16.2f}% | {p2_res['map']*100:>16.2f}% | {'+' if d_map >= 0 else ''}{d_map*100:>15.2f}%")
    print(f"{'OVERALL AP50':<16} | {base_res['map50']*100:>10.2f}% | {p1_res['map50']*100:>16.2f}% | {p2_res['map50']*100:>16.2f}% | {'+' if d_map50 >= 0 else ''}{d_map50*100:>15.2f}%")
    print(f"{'OVERALL AP75':<16} | {base_res['map75']*100:>10.2f}% | {p1_res['map75']*100:>16.2f}% | {p2_res['map75']*100:>16.2f}% | {'+' if d_map75 >= 0 else ''}{d_map75*100:>15.2f}%")
    print("=" * 90)

    # 6. Lưu kết quả ra file JSON
    out_file = ROOT_DIR / "methods" / "method5" / "report" / "3way_comparison_221pages.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "eval_pages": len(val_dataset),
            "base_model": {
                "name": "docling-project/docling-layout-heron",
                "map": round(float(base_res["map"]), 4),
                "map50": round(float(base_res["map50"]), 4),
                "map75": round(float(base_res["map75"]), 4),
            },
            "phase1_epoch3": {
                "weights": "method5_best_phase1_epoch3.pth",
                "map": round(float(p1_res["map"]), 4),
                "map50": round(float(p1_res["map50"]), 4),
                "map75": round(float(p1_res["map75"]), 4),
            },
            "phase2_epoch2": {
                "weights": str(p2_weights.name),
                "epoch": p2_epoch,
                "map": round(float(p2_res["map"]), 4),
                "map50": round(float(p2_res["map50"]), 4),
                "map75": round(float(p2_res["map75"]), 4),
            },
            "per_category": per_cat_comp,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Đã lưu kết quả so sánh 3 chiều tại: {out_file}")


if __name__ == "__main__":
    run_3way_comparison()
