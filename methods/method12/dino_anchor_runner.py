"""Stage 03.1 & Stage 04: DINO + M2Doc Anchor Runner (3-5 pages/book)."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm.auto import tqdm

from methods.method12.common import ROOT_DIR, book_of, select_anchor_pages

DINO_DIR = ROOT_DIR / "methods" / "dino_m2doc"
if str(DINO_DIR / "src") not in sys.path:
    sys.path.insert(0, str(DINO_DIR / "src"))


def load_dino_m2doc_from_cached_json(
    coco: dict,
    anchors_by_book: Dict[str, List[int]],
    pred_dir: Path,
    score_thresh: float = 0.35,
) -> Dict[str, List[Tuple[List[List[float]], List[int], List[float]]]]:
    """Load DINO+M2Doc anchor predictions from existing JSON files (e.g., doclaynet_test)."""
    img_by_id = {im["id"]: im for im in coco["images"]}
    book_anchors: Dict[str, List[Tuple[List[List[float]], List[int], List[float]]]] = {}

    for bname, page_ids in anchors_by_book.items():
        pages_data = []
        for pid in page_ids:
            im = img_by_id.get(pid)
            if im is None:
                continue
            h = float(im.get("height", 1025))
            w = float(im.get("width", 1025))
            jf = pred_dir / f"{pid}.json"
            boxes, classes, scores = [], [], []
            if jf.exists():
                with open(jf, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                preds = raw.get("detections", []) if isinstance(raw, dict) else raw
                for p in preds:
                    sc = float(p.get("score", 0.0))
                    if sc < score_thresh:
                        continue
                    bx, by, bw, bh = p["bbox"]
                    if bw <= 0 or bh <= 0:
                        continue
                    cx = max(0.0, min(1.0, (bx + bw / 2.0) / w))
                    cy = max(0.0, min(1.0, (by + bh / 2.0) / h))
                    nw = max(1e-4, min(1.0, bw / w))
                    nh = max(1e-4, min(1.0, bh / h))
                    boxes.append([cx, cy, nw, nh])
                    classes.append(int(p["category_id"]) - 1)
                    scores.append(sc)
            pages_data.append((boxes, classes, scores))
        book_anchors[bname] = pages_data
    return book_anchors


def run_dino_m2doc_on_anchors(
    coco: dict,
    anchors_by_book: Dict[str, List[int]],
    dataset_name: str,
    img_raw_dir: Path,
    img_640_dir: Path,
    device: str = "cuda",
    score_thresh: float = 0.35,
    ocr_cache: Optional[Dict[int, Tuple[List[List[float]], List[str]]]] = None,
    save_cache_path: Optional[Path] = None,
) -> Tuple[Dict[str, List[Tuple[List[List[float]], List[int], List[float]]]], List[dict]]:
    """Run DINO + M2Doc live on selected anchor pages (3-5 pages/book).

    Returns:
      - book_anchors: normalized anchors per book for Stage 03.2 / 04
      - raw_coco_dets: COCO detections for those anchor pages (for Overall Book Eval in Stage 04)
    """
    from mmcv_shim import apply_mmcv_shims
    apply_mmcv_shims()

    from ocr_provider import OCRProvider
    from run_inference import (
        DINO_M2DOC_TO_DOCLAYNET,
        build_transform,
        load_dino_m2doc_model,
        resolve_checkpoint_path,
        tokenize_ocr_for_m2doc,
    )
    from PIL import Image

    ckpt_path = resolve_checkpoint_path()
    model = load_dino_m2doc_model(ckpt_path, device=device)
    ocr = OCRProvider(dataset=dataset_name) if ocr_cache is None else None
    transform = build_transform()

    img_by_id = {im["id"]: im for im in coco["images"]}
    book_anchors: Dict[str, List[Tuple[List[List[float]], List[int], List[float]]]] = {}
    raw_coco_dets: List[dict] = []

    all_tasks = [(bname, pid) for bname, pids in anchors_by_book.items() for pid in pids]
    per_book_pages: Dict[str, List[Tuple[List[List[float]], List[int], List[float]]]] = defaultdict(list)

    for bname, pid in tqdm(all_tasks, desc=f"DINO+M2Doc Anchors ({dataset_name})"):
        im = img_by_id.get(pid)
        if im is None:
            continue
        orig_w = int(im.get("width", 1025))
        orig_h = int(im.get("height", 1025))
        praw = img_raw_dir / im["file_name"]
        p640 = img_640_dir / f"{Path(im['file_name']).stem}.jpg"
        img_path = str(praw if praw.exists() else p640)

        pil_img = Image.open(img_path).convert("RGB")
        img_tensor, (new_h, new_w) = transform(pil_img)
        img_tensor = img_tensor.to(device)

        if ocr_cache is not None and pid in ocr_cache:
            ocr_boxes_orig, ocr_texts = ocr_cache[pid]
            sx, sy = new_w / max(orig_w, 1), new_h / max(orig_h, 1)
            ocr_boxes = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in ocr_boxes_orig]
        else:
            ocr_boxes, ocr_texts = ocr.get_ocr_for_image(
                image_path=img_path,
                image_id=pid,
                target_size=(new_w, new_h),
                orig_size=(orig_w, orig_h),
            )

        input_ids, attention_mask, text_boxes = tokenize_ocr_for_m2doc(
            ocr_boxes, ocr_texts, (new_h, new_w), device=device
        )

        with torch.no_grad():
            outputs = model.forward_dummy(
                img=img_tensor.unsqueeze(0),
                img_metas=[{"img_shape": (new_h, new_w, 3), "scale_factor": 1.0, "ori_shape": (orig_h, orig_w, 3)}],
                input_ids=input_ids,
                attention_mask=attention_mask,
                text_boxes=text_boxes,
            )

        bboxes, labels, scores = outputs[0]
        sx_inv = orig_w / max(new_w, 1)
        sy_inv = orig_h / max(new_h, 1)

        p_boxes, p_classes, p_scores = [], [], []
        for box, lbl, sc in zip(bboxes.cpu().tolist(), labels.cpu().tolist(), scores.cpu().tolist()):
            if sc < score_thresh:
                continue
            cat_id = DINO_M2DOC_TO_DOCLAYNET.get(int(lbl))
            if cat_id is None:
                continue
            x1, y1, x2, y2 = box
            ox1, oy1, ox2, oy2 = x1 * sx_inv, y1 * sy_inv, x2 * sx_inv, y2 * sy_inv
            bw, bh = max(0.0, ox2 - ox1), max(0.0, oy2 - oy1)
            if bw <= 0 or bh <= 0:
                continue

            raw_coco_dets.append({
                "image_id": int(pid),
                "category_id": int(cat_id),
                "bbox": [round(ox1, 2), round(oy1, 2), round(bw, 2), round(bh, 2)],
                "score": round(float(sc), 5),
            })

            cx = max(0.0, min(1.0, (ox1 + bw / 2.0) / max(orig_w, 1)))
            cy = max(0.0, min(1.0, (oy1 + bh / 2.0) / max(orig_h, 1)))
            nw = max(1e-4, min(1.0, bw / max(orig_w, 1)))
            nh = max(1e-4, min(1.0, bh / max(orig_h, 1)))
            p_boxes.append([cx, cy, nw, nh])
            p_classes.append(int(cat_id) - 1)
            p_scores.append(float(sc))

        per_book_pages[bname].append((p_boxes, p_classes, p_scores))

    book_anchors = dict(per_book_pages)
    if save_cache_path is not None:
        save_cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"book_anchors": book_anchors, "raw_coco_dets": raw_coco_dets}, save_cache_path)
        print(f"[Stage 03.1] Saved DINO+M2Doc anchor cache to {save_cache_path}")

    return book_anchors, raw_coco_dets
