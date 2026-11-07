"""Evaluation and Benchmark: 3-Way Comparison.
1. Base RT-DETRv2 (Pretrained Single Page)
2. Book-Layout RT-DETR Epoch 3 (Checkpoint Best Acc)
3. Book-Layout RT-DETR Epoch 10 (Checkpoint Final Convergence - Restored)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

# Force UTF-8 stdout
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from methods.method_rtdetr_fusion.models.rtdetr_network import BookLayoutRTDetr
from methods.method_rtdetr_fusion.data.doclaynet_book_dataset import DocLayNetBookDataset, collate_book_fn

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

CLASS_COLORS = {
    0: "#e6194b",  # Caption - red
    1: "#3cb44b",  # Footnote - green
    2: "#ffe119",  # Formula - yellow
    3: "#4363d8",  # List-item - blue
    4: "#f58231",  # Page-footer - orange
    5: "#911eb4",  # Page-header - purple
    6: "#42d4f4",  # Picture - cyan
    7: "#f032e6",  # Section-header - magenta
    8: "#bfef45",  # Table - lime
    9: "#469990",  # Text - teal
    10: "#9a6324", # Title - brown
}


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """Tính IoU giữa 2 box dạng [cx, cy, w, h] chuẩn hóa [0, 1]."""
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


def eval_detections(logits, boxes, gt_labels, gt_boxes, metrics_dict, conf_thresh=0.35, iou_thresh=0.5):
    # Dùng sigmoid cho RT-DETRv2 logits
    probs = torch.sigmoid(logits)
    max_probs, pred_classes = probs.max(dim=-1)
    valid_idx = (max_probs >= conf_thresh).nonzero(as_tuple=True)[0]

    matched_gt = set()
    for idx in valid_idx:
        p_cls = int(pred_classes[idx].item())
        if p_cls not in CLASS_NAMES:
            continue
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
        g_cls = int(gt_labels[g_idx].item())
        if g_cls in CLASS_NAMES and g_idx not in matched_gt:
            metrics_dict["fn"][g_cls] += 1


def draw_boxes(image: Image.Image, boxes: list, labels: list, title: str) -> Image.Image:
    img_draw = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img_draw)
    w, h = img_draw.size

    for box, lbl in zip(boxes, labels):
        cx, cy, bw, bh = box
        xmin = (cx - bw / 2.0) * w
        ymin = (cy - bh / 2.0) * h
        xmax = (cx + bw / 2.0) * w
        ymax = (cy + bh / 2.0) * h

        color = CLASS_COLORS.get(lbl, "#ffffff")
        name = CLASS_NAMES.get(lbl, str(lbl))
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
        draw.text((xmin + 2, ymin + 2), name, fill=color)

    # Title banner
    draw.rectangle([0, 0, w, 24], fill="#111111")
    draw.text((8, 4), title, fill="#ffffff")
    return img_draw


def run_benchmark_3way(limit_pages=150, conf_thresh=0.35, iou_thresh=0.5):
    print("=" * 80)
    print("🚀 BẮT ĐẦU ĐỐI CHUẨN 3 MÔ HÌNH: BASE RT-DETR vs EPOCH 3 vs EPOCH 10")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using device: {device}")

    # 1. Dataset
    val_anno = ROOT_DIR / "datasets" / "doclaynet_test" / "processed" / "annotations" / "test.json"
    val_img = ROOT_DIR / "datasets" / "doclaynet_test" / "processed" / "images"

    dataset = DocLayNetBookDataset(
        annotation_file=val_anno,
        images_dir=val_img,
        image_size=640,
        window_size=4,
        limit_pages=limit_pages if limit_pages > 0 else None,
        is_train=False,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_book_fn)
    print(f"[Dataset] Đang đánh giá trên {len(dataset):,} trang kiểm thử.")

    # 2. Nạp Models
    print("[Models] Khởi tạo và nạp trọng số:")
    # Base
    m_base = BookLayoutRTDetr().to(device)
    m_base.eval()
    print("  ✓ Base RT-DETRv2 Pretrained")

    # Ep 3
    m_ep3 = BookLayoutRTDetr().to(device)
    c3 = torch.load(ROOT_DIR / "methods/method_rtdetr_fusion/weights/book_rtdetr_epoch3.pth", map_location=device)
    m_ep3.load_state_dict(c3["model_state_dict"])
    m_ep3.eval()
    print(f"  ✓ Epoch 3 Checkpoint (Val Acc: {c3.get('val_acc', 0):.4f})")

    # Ep 10
    m_ep10 = BookLayoutRTDetr().to(device)
    c10 = torch.load(ROOT_DIR / "methods/method_rtdetr_fusion/weights/book_rtdetr_epoch10.pth", map_location=device)
    m_ep10.load_state_dict(c10["model_state_dict"])
    m_ep10.eval()
    print(f"  ✓ Epoch 10 Checkpoint (Final Converged Loss: {c10.get('val_loss', 0):.4f})")

    metrics = {
        "base": {"tp": defaultdict(int), "fp": defaultdict(int), "fn": defaultdict(int)},
        "ep3":  {"tp": defaultdict(int), "fp": defaultdict(int), "fn": defaultdict(int)},
        "ep10": {"tp": defaultdict(int), "fp": defaultdict(int), "fn": defaultdict(int)},
    }

    vis_dir = ROOT_DIR / "methods" / "method_rtdetr_fusion" / "report" / "vis_samples"
    vis_dir.mkdir(parents=True, exist_ok=True)
    saved_vis_count = 0

    t0 = time.time()
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(device)
            gt_labels_list = batch["labels"][0]["class_labels"].to(device)
            gt_boxes_list = batch["labels"][0]["boxes"].to(device)
            ctx = batch["context_elements"][0]
            img_path = val_img / batch["file_names"][0]

            # 1. Base inference
            base_out = m_base.rtdetr(pixel_values=pixel_values)
            b_logits = base_out.logits[0]
            b_boxes = base_out.pred_boxes[0]

            # 2. Ep 3 inference
            e3_book = m_ep3.book_encoder(ctx, device=device)
            ep3_out = m_ep3(pixel_values, e_book=e3_book)
            ep3_logits = ep3_out["logits"][0]
            ep3_boxes = ep3_out["pred_boxes"][0]

            # 3. Ep 10 inference
            e10_book = m_ep10.book_encoder(ctx, device=device)
            ep10_out = m_ep10(pixel_values, e_book=e10_book)
            ep10_logits = ep10_out["logits"][0]
            ep10_boxes = ep10_out["pred_boxes"][0]

            # Evaluation
            eval_detections(b_logits, b_boxes, gt_labels_list, gt_boxes_list, metrics["base"], conf_thresh, iou_thresh)
            eval_detections(ep3_logits, ep3_boxes, gt_labels_list, gt_boxes_list, metrics["ep3"], conf_thresh, iou_thresh)
            eval_detections(ep10_logits, ep10_boxes, gt_labels_list, gt_boxes_list, metrics["ep10"], conf_thresh, iou_thresh)

            # Lưu ảnh visual cho 3 trang đầu
            if saved_vis_count < 3 and Path(img_path).exists():
                orig_img = Image.open(img_path).resize((640, 640))
                # GT
                gt_boxes_np = [gt_boxes_list[i].tolist() for i in range(len(gt_labels_list))]
                gt_labels_np = [int(gt_labels_list[i].item()) for i in range(len(gt_labels_list))]
                img_gt = draw_boxes(orig_img, gt_boxes_np, gt_labels_np, f"Sample #{idx+1} - Ground Truth")

                # Ep10 preds
                probs10 = torch.sigmoid(ep10_logits)
                max_probs10, pred_cls10 = probs10.max(dim=-1)
                mask10 = (max_probs10 >= conf_thresh)
                pred_boxes10_np = [ep10_boxes[i].tolist() for i in mask10.nonzero(as_tuple=True)[0]]
                pred_labels10_np = [int(pred_cls10[i].item()) for i in mask10.nonzero(as_tuple=True)[0]]
                img_ep10 = draw_boxes(orig_img, pred_boxes10_np, pred_labels10_np, f"Book-Layout RT-DETR (Epoch 10)")

                # Ep3 preds
                probs3 = torch.sigmoid(ep3_logits)
                max_probs3, pred_cls3 = probs3.max(dim=-1)
                mask3 = (max_probs3 >= conf_thresh)
                pred_boxes3_np = [ep3_boxes[i].tolist() for i in mask3.nonzero(as_tuple=True)[0]]
                pred_labels3_np = [int(pred_cls3[i].item()) for i in mask3.nonzero(as_tuple=True)[0]]
                img_ep3 = draw_boxes(orig_img, pred_boxes3_np, pred_labels3_np, f"Book-Layout RT-DETR (Epoch 3)")

                # Ghép 3 ảnh thành 1 dải ngang so sánh
                combined = Image.new("RGB", (640 * 3, 640))
                combined.paste(img_gt, (0, 0))
                combined.paste(img_ep3, (640, 0))
                combined.paste(img_ep10, (1280, 0))
                save_fp = vis_dir / f"compare_sample_{saved_vis_count+1}.png"
                combined.save(save_fp)
                saved_vis_count += 1

            if (idx + 1) % 25 == 0 or (idx + 1) == len(dataloader):
                print(f"   Đã kiểm tra [{idx + 1:04d}/{len(dataloader):04d}] trang ({time.time() - t0:.1f}s)...", flush=True)

    # Tổng hợp bảng so sánh 3-way
    def calc_metrics(m_dict):
        rows = {}
        all_tp, all_fp, all_fn = 0, 0, 0
        for cid, cname in CLASS_NAMES.items():
            tp = m_dict["tp"][cid]
            fp = m_dict["fp"][cid]
            fn = m_dict["fn"][cid]
            p = tp / max(1, tp + fp)
            r = tp / max(1, tp + fn)
            f1 = (2 * p * r) / max(1e-6, p + r)
            rows[cname] = {"p": round(p, 4), "r": round(r, 4), "f1": round(f1, 4)}
            all_tp += tp
            all_fp += fp
            all_fn += fn
        micro_p = all_tp / max(1, all_tp + all_fp)
        micro_r = all_tp / max(1, all_tp + all_fn)
        micro_f1 = (2 * micro_p * micro_r) / max(1e-6, micro_p + micro_r)
        return rows, {"p": round(micro_p, 4), "r": round(micro_r, 4), "f1": round(micro_f1, 4)}

    b_res, b_micro = calc_metrics(metrics["base"])
    ep3_res, ep3_micro = calc_metrics(metrics["ep3"])
    ep10_res, ep10_micro = calc_metrics(metrics["ep10"])

    print("\n" + "=" * 95)
    print(f"{'Class Name':<16} | {'Base F1':<10} | {'Epoch 3 F1':<12} | {'Epoch 10 F1':<12} | {'Improvement (vs Base)':<20}")
    print("-" * 95)

    comparison_data = []
    for cname in CLASS_NAMES.values():
        bf1 = b_res[cname]["f1"]
        e3f1 = ep3_res[cname]["f1"]
        e10f1 = ep10_res[cname]["f1"]
        delta = round((e10f1 - bf1) * 100, 2)
        sign = "+" if delta > 0 else ""
        print(f"{cname:<16} | {bf1*100:>8.2f}% | {e3f1*100:>10.2f}% | {e10f1*100:>10.2f}% | {sign}{delta:>6.2f}%")
        comparison_data.append({
            "class": cname,
            "base_precision": b_res[cname]["p"],
            "base_recall": b_res[cname]["r"],
            "base_f1": bf1,
            "ep3_precision": ep3_res[cname]["p"],
            "ep3_recall": ep3_res[cname]["r"],
            "ep3_f1": e3f1,
            "ep10_precision": ep10_res[cname]["p"],
            "ep10_recall": ep10_res[cname]["r"],
            "ep10_f1": e10f1,
            "delta_f1_pct": delta,
        })

    print("-" * 95)
    delta_micro = round((ep10_micro["f1"] - b_micro["f1"]) * 100, 2)
    sign_m = "+" if delta_micro > 0 else ""
    print(f"{'OVERALL (Micro)':<16} | {b_micro['f1']*100:>8.2f}% | {ep3_micro['f1']*100:>10.2f}% | {ep10_micro['f1']*100:>10.2f}% | {sign_m}{delta_micro:>6.2f}%")
    print("=" * 95)

    # Lưu kết quả json
    out_file = ROOT_DIR / "methods" / "method_rtdetr_fusion" / "report" / "benchmark_comparison_3way.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "classes": comparison_data,
            "overall": {
                "base_micro": b_micro,
                "ep3_micro": ep3_micro,
                "ep10_micro": ep10_micro,
                "delta_micro_f1_pct": delta_micro,
            }
        }, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Đã lưu file so sánh đối chuẩn 3 mô hình tại: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-pages", type=int, default=150)
    parser.add_argument("--conf-thresh", type=float, default=0.35)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    args = parser.parse_args()
    run_benchmark_3way(args.limit_pages, args.conf_thresh, args.iou_thresh)
