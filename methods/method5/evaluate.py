"""Evaluation and Benchmark: Method 5 vs Base RT-DETRv2.
Tính mAP@[0.50:0.95], AP50, AP75 và per-category metrics.
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


def evaluate_method5(args):
    print("=" * 80)
    print("📊 BẮT ĐẦU ĐỐI CHUẨN COCO mAP@[0.50:0.95]: BASE RT-DETR vs METHOD 5")
    print("=" * 80)

    device = torch.device(args.device)
    val_anno = ROOT_DIR / "datasets" / "doclaynet_test" / "processed" / "annotations" / "test.json"
    val_img = ROOT_DIR / "datasets" / "doclaynet_test" / "processed" / "images"

    dataset = DocLayNetBookDataset(
        annotation_file=val_anno,
        images_dir=val_img,
        image_size=640,
        window_size=args.window_size,
        limit_pages=args.limit_pages if args.limit_pages > 0 else None,
        is_train=False,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_book_fn)
    print(f"[Dataset] Đang đánh giá trên {len(dataset):,} trang kiểm thử.")

    # Nạp mô hình
    model = BookLayoutRTDetrMethod5(num_classes=17, num_summary_tokens=args.num_summary_tokens).to(device)
    weights_path = Path(args.weights)
    if weights_path.exists():
        print(f"[Model] Nạp trọng số Method 5 từ: {weights_path}")
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    else:
        print(f"[Model] ⚠️ Không tìm thấy file {weights_path}, sử dụng checkpoint mặc định.")

    model.eval()

    # 1. Đánh giá Base RT-DETRv2 (Single page, tắt Fusion)
    print("\n🔍 Đang đánh giá Base RT-DETRv2 (Single Page)...")
    t0 = time.time()
    base_res = evaluate_coco_map(model, dataloader, device=device, conf_thresh=args.conf_thresh, use_fusion=False)
    print(f"   Base: mAP@[.50:.95]: {base_res['map']*100:.2f}% | AP50: {base_res['map50']*100:.2f}% ({time.time()-t0:.1f}s)")

    # 2. Đánh giá Method 5 (Có Semantic Fusion)
    print("\n🚀 Đang đánh giá Method 5 (Book-Layout Semantic Fusion)...")
    t1 = time.time()
    m5_res = evaluate_coco_map(model, dataloader, device=device, conf_thresh=args.conf_thresh, use_fusion=True)
    print(f"   Method 5: mAP@[.50:.95]: {m5_res['map']*100:.2f}% | AP50: {m5_res['map50']*100:.2f}% ({time.time()-t1:.1f}s)")

    # 3. Bảng so sánh chi tiết
    print("\n" + "=" * 80)
    print(f"{'Category Name':<18} | {'Base AP@[.50:.95]':<18} | {'Method 5 AP@[.50:.95]':<22} | {'Improvement':<12}")
    print("-" * 80)

    cat_rows = []
    for cat in DOCLAYNET_CATEGORIES:
        cid = cat["id"]
        cname = cat["name"]
        b_ap = base_res["per_category"].get(cid, {}).get("ap", 0.0)
        m_ap = m5_res["per_category"].get(cid, {}).get("ap", 0.0)
        delta = (m_ap - b_ap) * 100
        sign = "+" if delta > 0 else ""
        print(f"{cname:<18} | {b_ap*100:>16.2f}% | {m_ap*100:>20.2f}% | {sign}{delta:>9.2f}%")
        cat_rows.append({
            "id": cid,
            "name": cname,
            "base_ap": round(b_ap, 4),
            "method5_ap": round(m_ap, 4),
            "delta_ap_pct": round(delta, 2),
        })

    delta_overall = (m5_res["map"] - base_res["map"]) * 100
    sign_o = "+" if delta_overall > 0 else ""
    print("-" * 80)
    print(f"{'OVERALL (mAP@[.50:.95])':<18} | {base_res['map']*100:>16.2f}% | {m5_res['map']*100:>20.2f}% | {sign_o}{delta_overall:>9.2f}%")
    print(f"{'OVERALL (AP50)':<18} | {base_res['map50']*100:>16.2f}% | {m5_res['map50']*100:>20.2f}% | {sign_o}{(m5_res['map50']-base_res['map50'])*100:>9.2f}%")
    print("=" * 80)

    out_file = ROOT_DIR / "methods" / "method5" / "report" / "benchmark_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "categories": cat_rows,
            "overall": {
                "base_map": round(base_res["map"], 4),
                "base_map50": round(base_res["map50"], 4),
                "method5_map": round(m5_res["map"], 4),
                "method5_map50": round(m5_res["map50"], 4),
                "delta_map_pct": round(delta_overall, 2),
            }
        }, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Đã lưu kết quả so sánh tại: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="methods/method5/weights/method5_best.pth")
    parser.add_argument("--limit-pages", type=int, default=100)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--num-summary-tokens", type=int, default=8)
    parser.add_argument("--conf-thresh", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    evaluate_method5(args)
