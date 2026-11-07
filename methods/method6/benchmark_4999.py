"""Benchmark toàn diện Method 6 trên toàn bộ 4,999 trang DocLayNet Test Set.

Đo đạc và đối chuẩn với:
1. Base Pretrained Model (RT-DETRv2 Docling Heron)
2. Method 5 Phase 1 (64-Page Book Context Dual-Stage OCR Fusion)
3. Method 6 (Spatial Sidecar Heatmap RT-DETR)

Đầy đủ các tiêu chí chuẩn:
- COCO mAP@[.50:.95], AP50, AP75, AP small/med/large, AR1, AR10, AR100
- Micro & Macro Precision, Recall, F1 @ IoU 0.50
- Pure BBox Precision, Recall, F1, Mean IoU
- Chi tiết 11 lớp văn bản DocLayNet
- Profiling: Thời gian (s), Độ trễ mỗi trang (ms), FPS, Peak VRAM, Peak RAM
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

# Force UTF-8 stdout
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.common.metrics import coco_map, evaluate, evaluate_boxes
from src.common.profiling import peak_ram_mb, peak_vram_mb, sync_device
from methods.method6.models.rtdetr_method6 import SpatialSidecarRTDetrMethod6
from methods.method6.data.doclaynet_spatial_dataset import DocLayNetSpatialDataset, collate_spatial_fn

DOCLAYNET_CATEGORIES = [
    {"id": 1, "name": "Caption"},
    {"id": 2, "name": "Footnote"},
    {"id": 3, "name": "Formula"},
    {"id": 4, "name": "List-item"},
    {"id": 5, "name": "Page-footer"},
    {"id": 6, "name": "Page-header"},
    {"id": 7, "name": "Picture"},
    {"id": 8, "name": "Section-header"},
    {"id": 9, "name": "Table"},
    {"id": 10, "name": "Text"},
    {"id": 11, "name": "Title"},
]
CATEGORY_NAMES = {c["id"]: c["name"] for c in DOCLAYNET_CATEGORIES}
HERON_TO_DOCLAYNET_ID = {
    0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11
}


def serialize_obj(obj):
    if isinstance(obj, (torch.Tensor, np.integer)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, dict):
        return {str(k): serialize_obj(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_obj(v) for v in obj]
    return obj


def evaluate_method6_4999(checkpoint_path: Path, conf_thresh: float = 0.2, no_cache: bool = False):
    print("=" * 80)
    print("🚀 BẮT ĐẦU BENCHMARK TOÀN DIỆN METHOD 6 TRÊN 4,999 TRANG DOCLAYNET TEST")
    print(f"  Checkpoint: {checkpoint_path}")
    print("=" * 80)

    cache_path = ROOT_DIR / "methods" / "method6" / "report" / f"eval_cache_{checkpoint_path.stem}.pt"
    legacy_cache_path = ROOT_DIR / "methods" / "method6" / "report" / "eval_cache.pt"

    chosen_cache = cache_path if cache_path.exists() else (legacy_cache_path if (legacy_cache_path.exists() and "best" in checkpoint_path.name) else None)

    if chosen_cache and not no_cache:
        print(f"[Cache] Tìm thấy cache suy luận tại {chosen_cache}! Đang nạp lại để tính chỉ số...")
        cached_data = torch.load(chosen_cache, weights_only=False)
        gt_images = cached_data["gt_images"]
        gt_annotations = cached_data["gt_annotations"]
        gt_by_image = cached_data["gt_by_image"]
        pred_by_image = cached_data["pred_by_image"]
        predictions_list = cached_data["predictions_list"]
        total_time = cached_data["total_time"]
        peak_vram = cached_data["peak_vram"]
        peak_ram = cached_data["peak_ram"]
        ms_per_page = cached_data["ms_per_page"]
        fps = cached_data["fps"]
        eval_pages = cached_data["eval_pages"]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Device] Thiết bị: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

        test_anno = ROOT_DIR / "datasets" / "doclaynet_test" / "processed" / "annotations" / "test.json"
        test_img = ROOT_DIR / "datasets" / "doclaynet_test" / "processed" / "images"

        test_ds = DocLayNetSpatialDataset(
            annotation_file=test_anno,
            images_dir=test_img,
            image_size=640,
            heatmap_size=80,
            window_size=64,
            is_train=False,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=8,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_spatial_fn,
            pin_memory=True,
        )

        print(f"[Dataset] Sẵn sàng đánh giá {len(test_ds):,} trang (Batch size = {test_loader.batch_size}, {len(test_loader)} batches).")

        # Load Model
        print(f"[Model] Nạp trọng số Method 6 từ: {checkpoint_path}")
        model = SpatialSidecarRTDetrMethod6(
            pretrained_model_path="docling-project/docling-layout-heron",
            num_classes=17,
            heatmap_channels=22,
            freeze_backbone=True,
        ).to(device)

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        print(f"[Model] Fusion Gates hiện tại: {model.fusion_gates.detach().cpu().tolist()}")

        # Profiling start
        sync_device()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        gc.collect()

        start_time = time.perf_counter()

        gt_images = []
        gt_annotations = []
        gt_by_image = {}
        pred_by_image = {}
        predictions_list = []
        ann_id_counter = 1

        print("\n>>> Đang chạy suy luận trên 4,999 trang...")

        with torch.no_grad():
            for b_idx, batch in enumerate(test_loader):
                pixel_values = batch["pixel_values"].to(device)
                layout_heatmaps = batch["layout_heatmaps"].to(device)
                targets = batch["targets"]

                with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                    out = model(pixel_values, layout_heatmaps)
                    logits = out["logits"]
                    pred_boxes = out["pred_boxes"]

                probs = torch.sigmoid(logits)
                bs = pixel_values.size(0)

                for i in range(bs):
                    tgt = targets[i]
                    img_id = tgt["image_id"]
                    h_orig, w_orig = tgt["orig_size"].tolist()

                    # GT
                    gt_images.append({
                        "id": img_id,
                        "width": int(w_orig),
                        "height": int(h_orig),
                    })
                    gt_boxes_for_img = []
                    gt_b = tgt["boxes"]
                    gt_c = tgt["class_labels"]
                    for b_sub in range(len(gt_b)):
                        cx, cy, w, h = gt_b[b_sub].tolist()
                        x1 = (cx - w / 2.0) * w_orig
                        y1 = (cy - h / 2.0) * h_orig
                        bw = w * w_orig
                        bh = h * h_orig
                        heron_cid = int(gt_c[b_sub].item())
                        doc_cid = HERON_TO_DOCLAYNET_ID.get(heron_cid, 10)

                        gt_annotations.append({
                            "id": ann_id_counter,
                            "image_id": img_id,
                            "category_id": doc_cid,
                            "bbox": [x1, y1, bw, bh],
                            "area": bw * bh,
                            "iscrowd": 0,
                        })
                        gt_boxes_for_img.append({
                            "category_id": doc_cid,
                            "bbox": [x1, y1, bw, bh],
                        })
                        ann_id_counter += 1
                    gt_by_image[img_id] = gt_boxes_for_img

                    # Pred
                    p_probs = probs[i]
                    p_boxes = pred_boxes[i]
                    max_scores, max_cids = p_probs[:, :11].max(dim=-1)
                    keep_mask = max_scores >= conf_thresh

                    v_scores = max_scores[keep_mask]
                    v_classes = max_cids[keep_mask]
                    v_boxes = p_boxes[keep_mask]

                    pred_boxes_for_img = []
                    for score, cls_id, box in zip(v_scores, v_classes, v_boxes):
                        c_heron = cls_id.item()
                        if c_heron not in HERON_TO_DOCLAYNET_ID:
                            continue
                        doc_cat_id = HERON_TO_DOCLAYNET_ID[c_heron]

                        cx, cy, bw, bh = box.tolist()
                        abs_x = (cx - bw / 2.0) * w_orig
                        abs_y = (cy - bh / 2.0) * h_orig
                        abs_w = bw * w_orig
                        abs_h = bh * h_orig
                        bbox_xywh = [float(abs_x), float(abs_y), float(abs_w), float(abs_h)]
                        score_val = float(score.item())

                        pred_item = {
                            "category_id": doc_cat_id,
                            "bbox": bbox_xywh,
                            "score": score_val,
                        }
                        pred_boxes_for_img.append(pred_item)
                        predictions_list.append({
                            "image_id": img_id,
                            "category_id": doc_cat_id,
                            "bbox": bbox_xywh,
                            "score": score_val,
                        })
                    pred_by_image[img_id] = pred_boxes_for_img

                if (b_idx + 1) % 50 == 0 or (b_idx + 1) == len(test_loader):
                    elapsed = time.perf_counter() - start_time
                    done_pages = min((b_idx + 1) * test_loader.batch_size, len(test_ds))
                    fps = done_pages / max(1e-4, elapsed)
                    print(f"  [Progress] {done_pages:,}/{len(test_ds):,} trang ({done_pages/len(test_ds)*100:.1f}%) | {elapsed:.1f}s | {fps:.1f} FPS")

        sync_device()
        total_time = time.perf_counter() - start_time
        peak_vram = peak_vram_mb()
        peak_ram = peak_ram_mb()
        eval_pages = len(test_ds)
        ms_per_page = (total_time / eval_pages) * 1000.0
        fps = eval_pages / total_time

        print(f"\n✓ Hoàn thành suy luận {eval_pages:,} trang trong {total_time:.2f}s ({fps:.1f} FPS, {ms_per_page:.2f} ms/trang).")
        print(f"  Peak VRAM: {peak_vram:.1f} MB | Peak RAM: {peak_ram:.1f} MB")

        # Cache raw outputs to disk
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "gt_images": gt_images,
            "gt_annotations": gt_annotations,
            "gt_by_image": gt_by_image,
            "pred_by_image": pred_by_image,
            "predictions_list": predictions_list,
            "total_time": total_time,
            "peak_vram": peak_vram,
            "peak_ram": peak_ram,
            "ms_per_page": ms_per_page,
            "fps": fps,
            "eval_pages": eval_pages,
        }, cache_path)
        print(f"[Cache] Đã lưu kết quả suy luận thô vào: {cache_path}")

    # 1. COCO Benchmark
    print("\n--- Đang tính toán COCO mAP@[.50:.95]... ---")
    gt_coco = {
        "images": gt_images,
        "annotations": gt_annotations,
        "categories": DOCLAYNET_CATEGORIES,
    }
    coco_res = coco_map(gt_coco, predictions_list)

    # 2. Strict Classification & BBox Localization
    print("--- Đang tính toán Strict Metrics & Box Localization... ---")
    cls_res = evaluate(gt_by_image, pred_by_image, CATEGORY_NAMES, iou_threshold=0.5)
    box_res = evaluate_boxes(gt_by_image, pred_by_image, iou_threshold=0.5)

    macro_prec = float(np.mean([cls_res["per_category"].get(c, {}).get("precision", 0.0) for c in CATEGORY_NAMES]))
    macro_rec = float(np.mean([cls_res["per_category"].get(c, {}).get("recall", 0.0) for c in CATEGORY_NAMES]))
    macro_f1 = float(np.mean([cls_res["per_category"].get(c, {}).get("f1", 0.0) for c in CATEGORY_NAMES]))

    m6_results = {
        "model_name": "Method 6 (Spatial Sidecar Heatmap RT-DETR)",
        "runtime": {
            "total_time_sec": round(total_time, 2),
            "fps": round(fps, 1),
            "ms_per_page": round(ms_per_page, 2),
            "peak_vram_mb": round(peak_vram, 1),
            "peak_ram_mb": round(peak_ram, 1),
        },
        "coco_map": {
            "map": round(coco_res["overall"]["map"], 4),
            "map50": round(coco_res["overall"]["map50"], 4),
            "map75": round(coco_res["overall"]["map75"], 4),
            "map_small": round(coco_res["overall"]["map_small"], 4),
            "map_medium": round(coco_res["overall"]["map_medium"], 4),
            "map_large": round(coco_res["overall"]["map_large"], 4),
            "ar1": round(coco_res["overall"]["ar1"], 4),
            "ar10": round(coco_res["overall"]["ar10"], 4),
            "ar100": round(coco_res["overall"]["ar100"], 4),
        },
        "per_category_coco": {
            str(cid): {
                "name": CATEGORY_NAMES[cid],
                "ap": round(coco_res["per_category"].get(cid, {}).get("ap", 0.0), 4),
                "ap50": round(coco_res["per_category"].get(cid, {}).get("ap50", 0.0), 4),
            }
            for cid in sorted(CATEGORY_NAMES.keys())
        },
        "classification": {
            "micro": {
                "precision": round(cls_res["overall"]["precision"], 4),
                "recall": round(cls_res["overall"]["recall"], 4),
                "f1": round(cls_res["overall"]["f1"], 4),
                "mean_iou_matched": round(cls_res["overall"]["mean_iou_matched"], 4),
            },
            "macro": {
                "precision": round(macro_prec, 4),
                "recall": round(macro_rec, 4),
                "f1": round(macro_f1, 4),
            },
            "per_category": {
                str(cid): {
                    "name": CATEGORY_NAMES[cid],
                    "precision": round(cls_res["per_category"].get(cid, {}).get("precision", 0.0), 4),
                    "recall": round(cls_res["per_category"].get(cid, {}).get("recall", 0.0), 4),
                    "f1": round(cls_res["per_category"].get(cid, {}).get("f1", 0.0), 4),
                    "support": cls_res["per_category"].get(cid, {}).get("support", 0),
                    "tp": cls_res["per_category"].get(cid, {}).get("tp", 0),
                    "fp": cls_res["per_category"].get(cid, {}).get("fp", 0),
                    "fn": cls_res["per_category"].get(cid, {}).get("fn", 0),
                }
                for cid in sorted(CATEGORY_NAMES.keys())
            },
        },
        "box_localization": {
            "precision": round(box_res["precision"], 4),
            "recall": round(box_res["recall"], 4),
            "f1": round(box_res["f1"], 4),
            "mean_iou_matched": round(box_res["mean_iou_matched"], 4),
            "total_gt": box_res["total_gt"],
            "total_pred": box_res["total_pred"],
        },
    }

    # Nạp kết quả đối chuẩn trước đó (Method 5 & Baseline)
    m5_benchmark_path = ROOT_DIR / "methods" / "method5" / "report" / "full_4999_benchmark_comparison.json"
    with open(m5_benchmark_path, "r", encoding="utf-8") as f:
        prev_data = json.load(f)

    comparison_report = {
        "eval_pages": eval_pages,
        "base_model": prev_data["base_model"],
        "method5_phase1": prev_data["method5_phase1"],
        "method6": m6_results,
    }

    out_json = ROOT_DIR / "methods" / "method6" / "report" / "full_4999_benchmark_comparison.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(serialize_obj(comparison_report), f, indent=2)

    print(f"\n[OK] Đã lưu báo cáo so sánh 3 mô hình trên 4,999 trang tại: {out_json}")
    return comparison_report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark Method 6 trên 4,999 trang DocLayNet Test")
    parser.add_argument("--checkpoint", type=str, default=str(ROOT_DIR / "methods" / "method6" / "weights" / "method6_epoch_15.pth"), help="Đường dẫn file trọng số .pth")
    parser.add_argument("--no_cache", action="store_true", help="Bỏ qua cache cũ và chạy suy luận lại từ đầu")
    parser.add_argument("--conf_thresh", type=float, default=0.2, help="Confidence threshold")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        fallback = ROOT_DIR / "methods" / "method6" / "weights" / "method6_best.pth"
        print(f"[Warning] Không tìm thấy {ckpt}, chuyển sang dùng {fallback}")
        ckpt = fallback

    evaluate_method6_4999(ckpt, conf_thresh=args.conf_thresh, no_cache=args.no_cache)
