"""Evaluation and Benchmark Script: Book-Layout RT-DETR vs Base RT-DETRv2.

So sánh chi tiết:
1. Base RT-DETRv2 (Không có ngữ cảnh sách, trang đơn).
2. Book-Layout RT-DETR (Có cơ chế Semantic Fusion ngữ cảnh cuốn sách E_book).
3. Đánh giá chi tiết trên từng lớp (Class-level Breakdown): F1-score, Precision, Recall.
4. Báo cáo bảng đối chiếu định lượng và lưu file JSON / CSV kết quả.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

# Force UTF-8 stdout
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from methods.method_rtdetr_fusion.models.rtdetr_network import BookLayoutRTDetr
from methods.method_rtdetr_fusion.data.doclaynet_book_dataset import DocLayNetBookDataset, collate_book_fn


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """Tính IoU giữa 2 box dạng [cx, cy, w, h] chuẩn hóa."""
    x1_min = box1[0] - box1[2] / 2.0
    y1_min = box1[1] - box1[3] / 2.0
    x1_max = box1[0] + box1[2] / 2.0
    y1_max = box1[1] + box1[3] / 2.0

    x2_min = box2[0] - box2[2] / 2.0
    y2_min = box2[1] - box2[3] / 2.0
    x2_max = box2[0] + box2[2] / 2.0
    y2_max = box2[1] + box2[3] / 2.0

    inter_xmin = max(float(x1_min), float(x2_min))
    inter_ymin = max(float(y1_min), float(y2_min))
    inter_xmax = min(float(x1_max), float(x2_max))
    inter_ymax = min(float(y1_max), float(y2_max))

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = float(box1[2] * box1[3])
    area2 = float(box2[2] * box2[3])
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def evaluate_model(args):
    print("=" * 70)
    print("📊 BẮT ĐẦU ĐÁNH GIÁ SO SÁNH: BASE RT-DETR vs BOOK-LAYOUT RT-DETR")
    print("=" * 70)

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
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_book_fn)
    print(f"[Dataset] Đang đánh giá trên {len(dataset):,} trang kiểm thử.")

    # Nạp mô hình
    model = BookLayoutRTDetr(num_classes=17, num_summary_tokens=args.num_summary_tokens).to(device)
    weights_path = Path(args.weights)
    if weights_path.exists():
        print(f"[Model] Nạp trọng số fine-tuned từ: {weights_path}")
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    else:
        print(f"[Model] ⚠️ Không tìm thấy file trọng số {weights_path}, sử dụng checkpoint mặc định của Docling Heron.")

    model.eval()

    # Nhãn quan tâm trong DocLayNet
    CLASS_NAMES = {
        0: "Caption",
        1: "Footnote",
        2: "Formula",
        3: "List-item",
        4: "Page-footer",
        5: "Page-header",
        6: "Picture",
        7: "Section-header",
        8: "Table",
        9: "Text",
        10: "Title",
    }

    # Thống kê TP, FP, FN cho từng mô hình
    metrics = {
        "base": {"tp": defaultdict(int), "fp": defaultdict(int), "fn": defaultdict(int)},
        "fusion": {"tp": defaultdict(int), "fp": defaultdict(int), "fn": defaultdict(int)},
    }

    t0 = time.time()
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(device)
            gt_labels_list = batch["labels"][0]["class_labels"].to(device)
            gt_boxes_list = batch["labels"][0]["boxes"].to(device)
            ctx = batch["context_elements"][0]

            # 1. Dự đoán Base RT-DETR (Single Page, không có ngữ cảnh)
            base_out = model.rtdetr(pixel_values=pixel_values)
            base_logits = base_out.logits[0]  # [300, 17]
            base_boxes = base_out.pred_boxes[0]  # [300, 4]

            # 2. Dự đoán Book-Layout RT-DETR (Có Semantic Fusion)
            e_book = model.book_encoder(ctx, device=device)
            fusion_out = model(pixel_values, e_book=e_book)
            fused_logits = fusion_out["logits"][0]
            fused_boxes = fusion_out["pred_boxes"][0]

            # Đánh giá cho Base
            _eval_page(base_logits, base_boxes, gt_labels_list, gt_boxes_list, metrics["base"], conf_thresh=args.conf_thresh, iou_thresh=args.iou_thresh)
            # Đánh giá cho Fusion
            _eval_page(fused_logits, fused_boxes, gt_labels_list, gt_boxes_list, metrics["fusion"], conf_thresh=args.conf_thresh, iou_thresh=args.iou_thresh)

            if (idx + 1) % 50 == 0 or (idx + 1) == len(dataloader):
                print(f"   Đã kiểm tra [{idx + 1:04d}/{len(dataloader):04d}] trang ({time.time() - t0:.1f}s)...", flush=True)

    # Tổng hợp bảng so sánh
    print("\n" + "=" * 90)
    print(f"{'Class Name':<16} | {'Base Precision':<14} {'Base Recall':<12} {'Base F1':<10} | {'Fusion Prec':<12} {'Fusion Rec':<11} {'Fusion F1':<10}")
    print("-" * 90)

    summary_rows = []
    base_all_tp, base_all_fp, base_all_fn = 0, 0, 0
    fuse_all_tp, fuse_all_fp, fuse_all_fn = 0, 0, 0

    for cid, cname in CLASS_NAMES.items():
        # Base
        b_tp = metrics["base"]["tp"][cid]
        b_fp = metrics["base"]["fp"][cid]
        b_fn = metrics["base"]["fn"][cid]
        b_prec = b_tp / max(1, b_tp + b_fp)
        b_rec = b_tp / max(1, b_tp + b_fn)
        b_f1 = (2 * b_prec * b_rec) / max(1e-6, b_prec + b_rec)

        # Fusion
        f_tp = metrics["fusion"]["tp"][cid]
        f_fp = metrics["fusion"]["fp"][cid]
        f_fn = metrics["fusion"]["fn"][cid]
        f_prec = f_tp / max(1, f_tp + f_fp)
        f_rec = f_tp / max(1, f_tp + f_fn)
        f_f1 = (2 * f_prec * f_rec) / max(1e-6, f_prec + f_rec)

        base_all_tp += b_tp
        base_all_fp += b_fp
        base_all_fn += b_fn
        fuse_all_tp += f_tp
        fuse_all_fp += f_fp
        fuse_all_fn += f_fn

        print(f"{cname:<16} | {b_prec*100:>12.2f}% {b_rec*100:>10.2f}% {b_f1*100:>8.2f}% | {f_prec*100:>10.2f}% {f_rec*100:>9.2f}% {f_f1*100:>8.2f}%")
        summary_rows.append({
            "class": cname,
            "base_precision": round(b_prec, 4),
            "base_recall": round(b_rec, 4),
            "base_f1": round(b_f1, 4),
            "fusion_precision": round(f_prec, 4),
            "fusion_recall": round(f_rec, 4),
            "fusion_f1": round(f_f1, 4),
        })

    # Macro và Micro Average
    b_micro_p = base_all_tp / max(1, base_all_tp + base_all_fp)
    b_micro_r = base_all_tp / max(1, base_all_tp + base_all_fn)
    b_micro_f1 = (2 * b_micro_p * b_micro_r) / max(1e-6, b_micro_p + b_micro_r)

    f_micro_p = fuse_all_tp / max(1, fuse_all_tp + fuse_all_fp)
    f_micro_r = fuse_all_tp / max(1, fuse_all_tp + fuse_all_fn)
    f_micro_f1 = (2 * f_micro_p * f_micro_r) / max(1e-6, f_micro_p + f_micro_r)

    print("-" * 90)
    print(f"{'OVERALL (Micro)':<16} | {b_micro_p*100:>12.2f}% {b_micro_r*100:>10.2f}% {b_micro_f1*100:>8.2f}% | {f_micro_p*100:>10.2f}% {f_micro_r*100:>9.2f}% {f_micro_f1*100:>8.2f}%")
    print("=" * 90)

    # Lưu kết quả so sánh
    out_dir = ROOT_DIR / "methods" / "method_rtdetr_fusion" / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "benchmark_comparison.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "classes": summary_rows,
            "overall": {
                "base_micro_f1": round(b_micro_f1, 4),
                "fusion_micro_f1": round(f_micro_f1, 4),
                "f1_improvement": round((f_micro_f1 - b_micro_f1) * 100, 2),
            }
        }, f, indent=2, ensure_ascii=False)
    print(f"📁 Đã lưu file so sánh chi tiết tại: {report_file}")


def _eval_page(logits, boxes, gt_labels, gt_boxes, metrics_dict, conf_thresh=0.4, iou_thresh=0.5):
    probs = F.softmax(logits, dim=-1)
    max_probs, pred_classes = probs.max(dim=-1)
    valid_idx = (max_probs >= conf_thresh).nonzero(as_tuple=True)[0]

    matched_gt = set()
    for idx in valid_idx:
        p_cls = int(pred_classes[idx].item())
        p_box = boxes[idx]

        best_iou = 0.0
        best_gt_idx = -1
        for g_idx in range(len(gt_labels)):
            if g_idx in matched_gt:
                continue
            if int(gt_labels[g_idx].item()) == p_cls:
                iou = compute_iou(p_box, gt_boxes[g_idx])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

        if best_iou >= iou_thresh and best_gt_idx != -1:
            metrics_dict["tp"][p_cls] += 1
            matched_gt.add(best_gt_idx)
        else:
            metrics_dict["fp"][p_cls] += 1

    for g_idx in range(len(gt_labels)):
        if g_idx not in matched_gt:
            g_cls = int(gt_labels[g_idx].item())
            metrics_dict["fn"][g_cls] += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Book-Layout RT-DETR vs Base")
    parser.add_argument("--weights", default="methods/method_rtdetr_fusion/weights/book_rtdetr_best.pth")
    parser.add_argument("--limit-pages", type=int, default=100)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--num-summary-tokens", type=int, default=8)
    parser.add_argument("--conf-thresh", type=float, default=0.35)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    evaluate_model(args)
