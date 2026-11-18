"""Benchmark toàn diện trên toàn bộ 4,999 trang DocLayNet Test Set:
So sánh:
1. Model Gốc Ban Đầu (Pretrained RT-DETRv2 - Docling Heron)
2. Method 5 Phase 1 Best (Epoch 3: 50 sách, 1,669 trang - method5_best_phase1_epoch3.pth)

Đánh giá theo đầy đủ các tiêu chuẩn của src.common.metrics và src.common.profiling:
- COCO mAP@[.50:.95], AP50, AP75, AP_s, AP_m, AP_l, AR1, AR10, AR100
- Micro & Macro Precision, Recall, F1 @ IoU 0.50
- Class-agnostic BBox Precision, Recall, F1, Mean IoU
- Chi tiết 11 lớp văn bản DocLayNet
- Profiling tốc độ: Tổng thời gian (s), Độ trễ mỗi trang (ms/page), FPS, Peak VRAM, Peak RAM
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

# Force UTF-8 stdout
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import RTDetrV2ForObjectDetection

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.common.metrics import coco_map, evaluate, evaluate_boxes
from src.common.profiling import peak_ram_mb, peak_vram_mb, sync_device
from methods.method5.models.rtdetr_method5 import BookLayoutRTDetrMethod5
from methods.method5.data.doclaynet_book_dataset import DocLayNetBookDataset, collate_book_fn

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


def evaluate_model_on_dataset(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_fusion: bool = False,
    conf_thresh: float = 0.2,
    model_name: str = "Model",
) -> dict:
    model.eval()
    print(f"\n>>> Đang chạy suy luận cho {model_name} trên {len(dataloader.dataset):,} trang (Batch size = {dataloader.batch_size})...")

    gt_images = []
    gt_annotations = []
    gt_by_image = {}
    pred_by_image = {}
    predictions_list = []
    ann_id_counter = 1

    sync_device()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()

    use_amp = (device.type == "cuda")

    with torch.no_grad():
        for b_idx, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(device)
            ctx_elements = batch["context_elements"]
            image_ids = batch["image_ids"]
            file_names = batch["file_names"]
            orig_sizes = batch["orig_sizes"]
            batch_annotations = batch["annotations"]
            bs = len(pixel_values)

            # 1. Forward
            with torch.amp.autocast("cuda", enabled=use_amp):
                if use_fusion:
                    out = model(pixel_values, context_elements=ctx_elements)
                    logits = out["logits"]
                    pred_boxes = out["pred_boxes"]
                else:
                    out = model(pixel_values=pixel_values)
                    logits = out.logits
                    pred_boxes = out.pred_boxes

            probs = torch.sigmoid(logits)

            for i in range(bs):
                img_id = image_ids[i]
                orig_w, orig_h = orig_sizes[i]

                # GT
                gt_images.append({
                    "id": img_id,
                    "file_name": file_names[i],
                    "width": orig_w,
                    "height": orig_h,
                })
                gt_boxes_for_img = []
                for ann in batch_annotations[i]:
                    cid = ann["category_id"]
                    gt_annotations.append({
                        "id": ann.get("id", ann_id_counter),
                        "image_id": img_id,
                        "category_id": cid,
                        "bbox": ann["bbox"],
                        "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                        "iscrowd": ann.get("iscrowd", 0),
                    })
                    gt_boxes_for_img.append({
                        "category_id": cid,
                        "bbox": ann["bbox"],
                    })
                    ann_id_counter += 1
                gt_by_image[img_id] = gt_boxes_for_img

                # Pred
                sample_probs = probs[i]
                sample_boxes = pred_boxes[i]
                max_scores, pred_classes = sample_probs.max(dim=-1)
                valid_mask = (max_scores >= conf_thresh)

                v_scores = max_scores[valid_mask]
                v_classes = pred_classes[valid_mask]
                v_boxes = sample_boxes[valid_mask]

                pred_boxes_for_img = []
                for score, cls_id, box in zip(v_scores, v_classes, v_boxes):
                    c_heron = cls_id.item()
                    if c_heron not in HERON_TO_DOCLAYNET_ID:
                        continue
                    doc_cat_id = HERON_TO_DOCLAYNET_ID[c_heron]

                    cx, cy, bw, bh = box.tolist()
                    abs_x = (cx - bw / 2.0) * orig_w
                    abs_y = (cy - bh / 2.0) * orig_h
                    abs_w = bw * orig_w
                    abs_h = bh * orig_h
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

            if (b_idx + 1) % 50 == 0 or (b_idx + 1) == len(dataloader):
                elapsed = time.perf_counter() - start_time
                done_pages = (b_idx + 1) * dataloader.batch_size
                fps = min(done_pages, len(dataloader.dataset)) / max(1e-4, elapsed)
                print(f"   [{b_idx+1:03d}/{len(dataloader):03d}] {min(done_pages, len(dataloader.dataset)):,}/{len(dataloader.dataset):,} trang ({elapsed:.1f}s - {fps:.1f} trang/s)")

    sync_device()
    total_time = time.perf_counter() - start_time
    total_pages = len(dataloader.dataset)
    fps = total_pages / max(1e-4, total_time)
    ms_per_page = (total_time / total_pages) * 1000.0
    vram_peak = peak_vram_mb() or 0.0
    ram_peak = peak_ram_mb() or 0.0

    print(f"\n[Timing] Hoàn tất {total_pages:,} trang trong {total_time:.1f}s ({fps:.1f} FPS, {ms_per_page:.1f} ms/trang). Peak VRAM: {vram_peak:.1f} MB | Peak RAM: {ram_peak:.1f} MB")

    # 2. Tính COCO mAP chuẩn pycocotools
    print("   -> Đang tính COCO mAP (pycocotools)...")
    gt_coco = {
        "images": gt_images,
        "annotations": gt_annotations,
        "categories": DOCLAYNET_CATEGORIES,
    }
    coco_res = coco_map(gt_coco, predictions_list)

    # 3. Tính Classification Precision / Recall / F1 @ IoU 0.50
    print("   -> Đang tính Precision / Recall / F1 (IoU=0.50)...")
    clf_res = evaluate(gt_by_image, pred_by_image, CATEGORY_NAMES, iou_threshold=0.5)

    # 4. Tính Class-Agnostic Localization Box Metrics
    print("   -> Đang tính Pure BBox Localization Metrics...")
    box_res = evaluate_boxes(gt_by_image, pred_by_image, iou_threshold=0.5)

    # Macro F1
    valid_macro_cats = [c["id"] for c in DOCLAYNET_CATEGORIES if clf_res["per_category"].get(c["id"], {}).get("support", 0) > 0]
    macro_p = np.mean([clf_res["per_category"][c]["precision"] for c in valid_macro_cats])
    macro_r = np.mean([clf_res["per_category"][c]["recall"] for c in valid_macro_cats])
    macro_f1 = np.mean([clf_res["per_category"][c]["f1"] for c in valid_macro_cats])

    return {
        "model_name": model_name,
        "runtime": {
            "total_time_sec": round(total_time, 2),
            "fps": round(fps, 2),
            "ms_per_page": round(ms_per_page, 2),
            "peak_vram_mb": round(vram_peak, 1),
            "peak_ram_mb": round(ram_peak, 1),
        },
        "coco_map": {
            "map": round(float(coco_res["overall"]["map"]), 4),
            "map50": round(float(coco_res["overall"]["map50"]), 4),
            "map75": round(float(coco_res["overall"]["map75"]), 4),
            "map_small": round(float(coco_res["overall"].get("map_small", 0.0)), 4),
            "map_medium": round(float(coco_res["overall"].get("map_medium", 0.0)), 4),
            "map_large": round(float(coco_res["overall"].get("map_large", 0.0)), 4),
            "ar1": round(float(coco_res["overall"].get("ar1", 0.0)), 4),
            "ar10": round(float(coco_res["overall"].get("ar10", 0.0)), 4),
            "ar100": round(float(coco_res["overall"].get("ar100", 0.0)), 4),
        },
        "per_category_coco": {
            str(cid): {
                "name": CATEGORY_NAMES[cid],
                "ap": round(float(coco_res["per_category"].get(cid, {}).get("ap", 0.0)), 4),
                "ap50": round(float(coco_res["per_category"].get(cid, {}).get("ap50", 0.0)), 4),
            }
            for cid in CATEGORY_NAMES
        },
        "classification": {
            "micro": {
                "precision": round(float(clf_res["overall"]["precision"]), 4),
                "recall": round(float(clf_res["overall"]["recall"]), 4),
                "f1": round(float(clf_res["overall"]["f1"]), 4),
                "mean_iou_matched": round(float(clf_res["overall"]["mean_iou_matched"]), 4),
            },
            "macro": {
                "precision": round(float(macro_p), 4),
                "recall": round(float(macro_r), 4),
                "f1": round(float(macro_f1), 4),
            },
            "per_category": {
                str(cid): {
                    "name": CATEGORY_NAMES[cid],
                    "precision": round(float(clf_res["per_category"].get(cid, {}).get("precision", 0.0)), 4),
                    "recall": round(float(clf_res["per_category"].get(cid, {}).get("recall", 0.0)), 4),
                    "f1": round(float(clf_res["per_category"].get(cid, {}).get("f1", 0.0)), 4),
                    "support": clf_res["per_category"].get(cid, {}).get("support", 0),
                    "tp": clf_res["per_category"].get(cid, {}).get("tp", 0),
                    "fp": clf_res["per_category"].get(cid, {}).get("fp", 0),
                    "fn": clf_res["per_category"].get(cid, {}).get("fn", 0),
                }
                for cid in CATEGORY_NAMES
            },
        },
        "box_localization": {
            "precision": round(float(box_res["precision"]), 4),
            "recall": round(float(box_res["recall"]), 4),
            "f1": round(float(box_res["f1"]), 4),
            "mean_iou_matched": round(float(box_res["mean_iou_matched"]), 4),
            "total_gt": box_res["total_gt"],
            "total_pred": box_res["total_pred"],
        },
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 85)
    print("🔬 BENCHMARK TOÀN DIỆN 4,999 TRANG DOCLAYNET: BASELINE vs METHOD 5 PHASE 1 BEST")
    print(f"   Thiết bị: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 85)

    val_anno = ROOT_DIR / "datasets" / "doclaynet_test" / "processed" / "annotations" / "test.json"
    val_img = ROOT_DIR / "datasets" / "doclaynet_test" / "processed" / "images"

    dataset = DocLayNetBookDataset(
        annotation_file=val_anno,
        images_dir=val_img,
        image_size=640,
        window_size=64,
        is_train=False,
        allowed_books=None,  # Quét toàn bộ 4,999 trang
    )
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_book_fn,
        num_workers=0,
    )
    print(f"[Dataset] Đã nạp {len(dataset):,} trang từ tập Test DocLayNet ({len(dataloader)} batches).")

    # 1. Đánh giá Model Gốc (Baseline)
    print("\n" + "=" * 85)
    print("1️⃣ ĐÁNH GIÁ MODEL GỐC BAN ĐẦU (docling-project/docling-layout-heron)")
    print("=" * 85)
    base_model = RTDetrV2ForObjectDetection.from_pretrained("docling-project/docling-layout-heron").to(device)
    base_results = evaluate_model_on_dataset(
        model=base_model,
        dataloader=dataloader,
        device=device,
        use_fusion=False,
        conf_thresh=0.2,
        model_name="Base Pretrained RT-DETR",
    )

    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Đánh giá Method 5 Phase 1 Best Model
    print("\n" + "=" * 85)
    print("2️⃣ ĐÁNH GIÁ METHOD 5 PHASE 1 BEST (method5_best_phase1_epoch3.pth)")
    print("=" * 85)
    m5_weights = ROOT_DIR / "methods" / "method5" / "weights" / "method5_best_phase1_epoch3.pth"
    m5_model = BookLayoutRTDetrMethod5(
        num_classes=17,
        num_summary_tokens=8,
        freeze_backbone=False,
    ).to(device)
    ckpt = torch.load(m5_weights, map_location=device, weights_only=False)
    m5_model.load_state_dict(ckpt["model_state_dict"])

    m5_results = evaluate_model_on_dataset(
        model=m5_model,
        dataloader=dataloader,
        device=device,
        use_fusion=True,
        conf_thresh=0.2,
        model_name="Method 5 Phase 1 Best",
    )

    del m5_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Tổng hợp Báo Cáo So Sánh
    report_data = {
        "eval_pages": len(dataset),
        "base_model": base_results,
        "method5_phase1": m5_results,
    }

    out_file = ROOT_DIR / "methods" / "method5" / "report" / "full_4999_benchmark_comparison.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Đã lưu kết quả JSON đầy đủ tại: {out_file}")

    # 4. In Bảng Tổng Kết Định Lượng Trực Quan
    print("\n" + "=" * 95)
    print("📊 BẢNG ĐỐI CHUẨN ĐỊNH LƯỢNG TOÀN DIỆN (4,999 TRANG DOCLAYNET TEST SET)")
    print("=" * 95)
    print(f"{'Tiêu chí đánh giá':<30} | {'Model Gốc (Baseline)':<20} | {'Method 5 (Phase 1)':<20} | {'Mức độ cải thiện':<18}")
    print("-" * 95)

    # Runtime
    b_time = base_results["runtime"]
    m_time = m5_results["runtime"]
    print(f"{'Thời gian suy luận (giây)':<30} | {b_time['total_time_sec']:>17.1f}s | {m_time['total_time_sec']:>17.1f}s | {m_time['total_time_sec'] - b_time['total_time_sec']:>+15.1f}s")
    print(f"{'Tốc độ xử lý (FPS)':<30} | {b_time['fps']:>17.1f}  | {m_time['fps']:>17.1f}  | {m_time['fps'] - b_time['fps']:>+15.1f}")
    print(f"{'Độ trễ trung bình (ms/trang)':<30} | {b_time['ms_per_page']:>15.1f}ms | {m_time['ms_per_page']:>15.1f}ms | {m_time['ms_per_page'] - b_time['ms_per_page']:>+13.1f}ms")
    print(f"{'Peak VRAM (MB)':<30} | {b_time['peak_vram_mb']:>15.1f}MB | {m_time['peak_vram_mb']:>15.1f}MB | {m_time['peak_vram_mb'] - b_time['peak_vram_mb']:>+13.1f}MB")
    print("-" * 95)

    # COCO mAP
    b_coco = base_results["coco_map"]
    m_coco = m5_results["coco_map"]
    d_map = m_coco["map"] - b_coco["map"]
    d_map50 = m_coco["map50"] - b_coco["map50"]
    d_map75 = m_coco["map75"] - b_coco["map75"]

    print(f"{'COCO mAP@[0.50:0.95]':<30} | {b_coco['map']*100:>19.2f}% | {m_coco['map']*100:>19.2f}% | {'+' if d_map>=0 else ''}{d_map*100:>15.2f}%")
    print(f"{'COCO AP50 (IoU=0.50)':<30} | {b_coco['map50']*100:>19.2f}% | {m_coco['map50']*100:>19.2f}% | {'+' if d_map50>=0 else ''}{d_map50*100:>15.2f}%")
    print(f"{'COCO AP75 (IoU=0.75 - Strict)':<30} | {b_coco['map75']*100:>19.2f}% | {m_coco['map75']*100:>19.2f}% | {'+' if d_map75>=0 else ''}{d_map75*100:>15.2f}%")
    print(f"{'COCO Average Recall (AR100)':<30} | {b_coco['ar100']*100:>19.2f}% | {m_coco['ar100']*100:>19.2f}% | {'+' if m_coco['ar100']>=b_coco['ar100'] else ''}{(m_coco['ar100']-b_coco['ar100'])*100:>15.2f}%")
    print("-" * 95)

    # F1 / Precision / Recall
    b_clf = base_results["classification"]
    m_clf = m5_results["classification"]
    d_f1 = m_clf["macro"]["f1"] - b_clf["macro"]["f1"]
    print(f"{'Macro F1-score (IoU=0.50)':<30} | {b_clf['macro']['f1']*100:>19.2f}% | {m_clf['macro']['f1']*100:>19.2f}% | {'+' if d_f1>=0 else ''}{d_f1*100:>15.2f}%")
    print(f"{'Macro Precision':<30} | {b_clf['macro']['precision']*100:>19.2f}% | {m_clf['macro']['precision']*100:>19.2f}% | {'+' if m_clf['macro']['precision']>=b_clf['macro']['precision'] else ''}{(m_clf['macro']['precision']-b_clf['macro']['precision'])*100:>15.2f}%")
    print(f"{'Macro Recall':<30} | {b_clf['macro']['recall']*100:>19.2f}% | {m_clf['macro']['recall']*100:>19.2f}% | {'+' if m_clf['macro']['recall']>=b_clf['macro']['recall'] else ''}{(m_clf['macro']['recall']-b_clf['macro']['recall'])*100:>15.2f}%")
    print(f"{'Mean IoU of Matched Boxes':<30} | {b_clf['micro']['mean_iou_matched']*100:>19.2f}% | {m_clf['micro']['mean_iou_matched']*100:>19.2f}% | {'+' if m_clf['micro']['mean_iou_matched']>=b_clf['micro']['mean_iou_matched'] else ''}{(m_clf['micro']['mean_iou_matched']-b_clf['micro']['mean_iou_matched'])*100:>15.2f}%")
    print("-" * 95)

    # Pure Localization
    b_box = base_results["box_localization"]
    m_box = m5_results["box_localization"]
    print(f"{'Localization-Only Box F1':<30} | {b_box['f1']*100:>19.2f}% | {m_box['f1']*100:>19.2f}% | {'+' if m_box['f1']>=b_box['f1'] else ''}{(m_box['f1']-b_box['f1'])*100:>15.2f}%")
    print("=" * 95)

    # Per-category COCO AP
    print("\n" + "=" * 95)
    print("📋 CHI TIẾT HIỆU NĂNG THEO TỪNG LỚP (PER-CATEGORY AP@[0.50:0.95] TRÊN 4,999 TRANG)")
    print("=" * 95)
    print(f"{'Phân loại (Category)':<20} | {'Model Gốc AP':<18} | {'Method 5 (Phase 1) AP':<24} | {'Chênh lệch (Delta)':<18}")
    print("-" * 95)

    for cat in DOCLAYNET_CATEGORIES:
        cid = str(cat["id"])
        cname = cat["name"]
        b_ap = base_results["per_category_coco"][cid]["ap"]
        m_ap = m5_results["per_category_coco"][cid]["ap"]
        delta = m_ap - b_ap
        sign = "+" if delta >= 0 else ""
        print(f"{cname:<20} | {b_ap*100:>15.2f}% | {m_ap*100:>21.2f}% | {sign}{delta*100:>15.2f}%")

    print("=" * 95)


if __name__ == "__main__":
    main()
