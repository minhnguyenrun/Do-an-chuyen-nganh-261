"""Fast YOLO Datasets, OCR Grid Precomputation, and COCO/F1 Evaluators for Method 12."""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from ultralytics.utils.nms import non_max_suppression

from methods.method12.common import (
    HEATMAP_SIZE,
    IMAGE_SIZE,
    NUM_DLN_CLASSES,
    book_of,
    coco_eval,
    compute_strict_loose_f1,
    load_image_640_tensor,
)
from methods.method12.ocr_utils import (
    build_book_anchor_heatmap,
    pack_anchor_memory,
    render_ocr_grid,
)


def _extract_yolo_boxes(anns: List[dict], img_w: float, img_h: float) -> torch.Tensor:
    rows = []
    for a in anns:
        bx, by, bw, bh = a["bbox"]
        if bw <= 0 or bh <= 0:
            continue
        cx = max(0.0, min(1.0, (bx + bw / 2.0) / img_w))
        cy = max(0.0, min(1.0, (by + bh / 2.0) / img_h))
        nw = max(1e-4, min(1.0, bw / img_w))
        nh = max(1e-4, min(1.0, bh / img_h))
        rows.append([float(int(a["category_id"]) - 1), cx, cy, nw, nh])
    return torch.tensor(rows, dtype=torch.float32) if rows else torch.zeros((0, 5), dtype=torch.float32)


def _collate_yolo_targets(boxes_list: List[torch.Tensor]) -> dict[str, torch.Tensor]:
    batch_idx, cls_list, bbox_list = [], [], []
    for i, b in enumerate(boxes_list):
        if len(b) > 0:
            batch_idx.append(torch.full((len(b),), i, dtype=torch.long))
            cls_list.append(b[:, 0:1])
            bbox_list.append(b[:, 1:5])
    if cls_list:
        return {
            "batch_idx": torch.cat(batch_idx),
            "cls": torch.cat(cls_list),
            "bboxes": torch.cat(bbox_list),
        }
    return {
        "batch_idx": torch.zeros((0,), dtype=torch.long),
        "cls": torch.zeros((0, 1), dtype=torch.float32),
        "bboxes": torch.zeros((0, 4), dtype=torch.float32),
    }


class Method12YOLODataset(Dataset):
    """Single-page fast YOLO dataset reading 640x640 SSD images."""

    def __init__(
        self,
        coco: dict,
        img_640_dir: Path,
        img_raw_dir: Optional[Path] = None,
        max_samples: Optional[int] = None,
    ):
        self.img_640_dir = img_640_dir
        self.img_raw_dir = img_raw_dir
        anns_by_img = defaultdict(list)
        for a in coco.get("annotations", []):
            anns_by_img[a["image_id"]].append(a)

        self.items = []
        for im in coco["images"]:
            self.items.append((im, anns_by_img.get(im["id"], [])))
            if max_samples is not None and len(self.items) >= max_samples:
                break

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        im, anns = self.items[idx]
        stem = Path(im["file_name"]).stem
        pix = load_image_640_tensor(
            self.img_640_dir / f"{stem}.jpg",
            (self.img_raw_dir / im["file_name"]) if self.img_raw_dir else None,
            augment=False,
        )
        h = float(im.get("height", 1025))
        w = float(im.get("width", 1025))
        boxes_t = _extract_yolo_boxes(anns, w, h)
        meta = (int(im["id"]), int(h), int(w))
        return pix, boxes_t, meta


def collate_fn_yolo(batch):
    pixs = torch.stack([b[0] for b in batch], dim=0)
    targets = _collate_yolo_targets([b[1] for b in batch])
    metas = [b[2] for b in batch]
    return pixs, targets, metas


def extract_gt_anchors_by_book(
    coco: dict,
    anchors_by_book: Dict[str, List[int]],
) -> Dict[str, List[Tuple[List[List[float]], List[int], List[float]]]]:
    """Extract Ground-Truth boxes from selected anchor pages (used in Stage 02)."""
    img_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_img = defaultdict(list)
    for a in coco.get("annotations", []):
        anns_by_img[a["image_id"]].append(a)

    book_anchors: Dict[str, List[Tuple[List[List[float]], List[int], List[float]]]] = {}
    for bname, page_ids in anchors_by_book.items():
        pages_data = []
        for pid in page_ids:
            im = img_by_id.get(pid)
            if im is None:
                continue
            h = float(im.get("height", 1025))
            w = float(im.get("width", 1025))
            boxes, classes, scores = [], [], []
            for a in anns_by_img.get(pid, []):
                bx, by, bw, bh = a["bbox"]
                if bw <= 0 or bh <= 0:
                    continue
                boxes.append([
                    max(0.0, min(1.0, (bx + bw / 2.0) / w)),
                    max(0.0, min(1.0, (by + bh / 2.0) / h)),
                    max(1e-4, min(1.0, bw / w)),
                    max(1e-4, min(1.0, bh / h)),
                ])
                classes.append(int(a["category_id"]) - 1)
                scores.append(1.0)
            pages_data.append((boxes, classes, scores))
        book_anchors[bname] = pages_data
    return book_anchors


def precompute_ocr_grids(
    coco: dict,
    dataset_name: str,
    img_raw_dir: Path,
    img_640_dir: Path,
    cache_path: Path,
    max_samples: Optional[int] = None,
) -> Dict[int, np.ndarray]:
    """Precompute 8-channel M2Doc-style OCR grids [8, 80, 80] (float16) for fast training."""
    if cache_path.exists():
        data = torch.load(cache_path, weights_only=False)
        print(f"[OCR Cache] Loaded {len(data)} OCR grids from {cache_path}")
        return data

    from methods.dino_m2doc.src.ocr_provider import OCRProvider

    ocr = OCRProvider(dataset=dataset_name)
    grids: Dict[int, np.ndarray] = {}
    images = coco["images"][:max_samples] if max_samples is not None else coco["images"]

    for im in tqdm(images, desc=f"Precompute OCR grids ({dataset_name})"):
        iid = im["id"]
        h = float(im.get("height", 1025))
        w = float(im.get("width", 1025))
        praw = img_raw_dir / im["file_name"]
        p640 = img_640_dir / f"{Path(im['file_name']).stem}.jpg"
        img_path = str(praw if praw.exists() else p640)

        boxes, texts = ocr.get_ocr_for_image(image_path=img_path, image_id=iid, target_size=(w, h), orig_size=(w, h))
        grids[iid] = render_ocr_grid(boxes, texts, src_size=(w, h), grid=HEATMAP_SIZE).astype(np.float16)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(grids, cache_path)
    print(f"[OCR Cache] Saved {len(grids)} OCR grids to {cache_path}")
    return grids


class Method12YOLOFusionDataset(Dataset):
    """Dataset providing 640x640 image + 19ch [Book Heatmap, OCR Grid] + Anchor Memory."""

    def __init__(
        self,
        coco: dict,
        img_640_dir: Path,
        img_raw_dir: Optional[Path],
        book_anchors: Dict[str, List[Tuple[List[List[float]], List[int], List[float]]]],
        exclude_image_ids: Optional[Set[int]] = None,
        ocr_grids: Optional[Dict[int, np.ndarray]] = None,
        max_anchors: int = 64,
        max_samples: Optional[int] = None,
    ):
        self.img_640_dir = img_640_dir
        self.img_raw_dir = img_raw_dir
        self.ocr_grids = ocr_grids or {}
        self.max_anchors = max_anchors

        self.book_heatmaps: Dict[str, np.ndarray] = {}
        self.book_memories: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for bname, anchor_pages in book_anchors.items():
            self.book_heatmaps[bname] = build_book_anchor_heatmap(anchor_pages, grid=HEATMAP_SIZE)
            self.book_memories[bname] = pack_anchor_memory(anchor_pages, max_anchors=max_anchors)

        anns_by_img = defaultdict(list)
        for a in coco.get("annotations", []):
            anns_by_img[a["image_id"]].append(a)

        self.items = []
        for im in coco["images"]:
            if exclude_image_ids is not None and im["id"] in exclude_image_ids:
                continue
            self.items.append((im, anns_by_img.get(im["id"], [])))
            if max_samples is not None and len(self.items) >= max_samples:
                break

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        im, anns = self.items[idx]
        iid = im["id"]
        bname = book_of(im)
        stem = Path(im["file_name"]).stem
        pix = load_image_640_tensor(
            self.img_640_dir / f"{stem}.jpg",
            (self.img_raw_dir / im["file_name"]) if self.img_raw_dir else None,
            augment=False,
        )

        book_hm = self.book_heatmaps.get(bname, np.zeros((NUM_DLN_CLASSES, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32))
        ocr_g = self.ocr_grids.get(iid)
        ocr_g = np.zeros((8, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32) if ocr_g is None else np.asarray(ocr_g, dtype=np.float32)
        early_19 = torch.from_numpy(np.concatenate([book_hm, ocr_g], axis=0))

        mem, mask = self.book_memories.get(
            bname,
            (np.zeros((self.max_anchors, 4 + NUM_DLN_CLASSES + 1), dtype=np.float32), np.zeros((self.max_anchors,), dtype=bool)),
        )
        h = float(im.get("height", 1025))
        w = float(im.get("width", 1025))
        boxes_t = _extract_yolo_boxes(anns, w, h)
        meta = (int(iid), int(h), int(w))
        return pix, early_19, torch.from_numpy(mem), torch.from_numpy(mask), boxes_t, meta


def collate_fn_yolo_fusion(batch):
    return (
        torch.stack([b[0] for b in batch], dim=0),
        torch.stack([b[1] for b in batch], dim=0),
        torch.stack([b[2] for b in batch], dim=0),
        torch.stack([b[3] for b in batch], dim=0),
        _collate_yolo_targets([b[4] for b in batch]),
        [b[5] for b in batch],
    )


def _decode_yolo_preds_to_coco(
    raw_preds: torch.Tensor,
    metas: List[Tuple[int, int, int]],
    conf_thres: float = 0.001,
    iou_thres: float = 0.65,
) -> list[dict]:
    """Decode [B, 15, 8400] YOLO output via NMS into COCO detections."""
    nms_out = non_max_suppression(raw_preds.float(), conf_thres=conf_thres, iou_thres=iou_thres, max_det=100)
    dets = []
    for i, det in enumerate(nms_out):
        if det is None or len(det) == 0:
            continue
        img_id, orig_h, orig_w = metas[i]
        sx, sy = orig_w / float(IMAGE_SIZE), orig_h / float(IMAGE_SIZE)
        for row in det.cpu().tolist():
            x1, y1, x2, y2, sc, cls_id = row[:6]
            ox1 = max(0.0, x1 * sx)
            oy1 = max(0.0, y1 * sy)
            bw = max(0.0, (x2 - x1) * sx)
            bh = max(0.0, (y2 - y1) * sy)
            dets.append({
                "image_id": int(img_id),
                "category_id": int(cls_id) + 1,
                "bbox": [round(ox1, 2), round(oy1, 2), round(bw, 2), round(bh, 2)],
                "score": round(float(sc), 5),
            })
    return dets


@torch.no_grad()
def evaluate_yolo_model(
    model,
    data_loader,
    coco_gt: dict,
    device: torch.device,
    is_fusion: bool = False,
    use_amp: bool = True,
    score_thresh_f1: float = 0.35,
    desc: str = "YOLO Eval",
) -> dict:
    """Evaluate YOLO (Stage 01 or Fusion) -> COCO mAP + Strict/Loose F1 + FPS."""
    model.eval()
    all_dets = []
    t0 = time.time()
    n_imgs = 0

    for batch in tqdm(data_loader, desc=desc, leave=False):
        if is_fusion:
            pixs, early_19, anchor_mem, anchor_mask, _, metas = batch
            pixs = pixs.to(device, non_blocking=True)
            early_19 = early_19.to(device, non_blocking=True)
            anchor_mem = anchor_mem.to(device, non_blocking=True)
            anchor_mask = anchor_mask.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(use_amp and device.type == "cuda")):
                out = model(pixs, early_19, anchor_mem, anchor_mask)
        else:
            pixs, _, metas = batch
            pixs = pixs.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(use_amp and device.type == "cuda")):
                out = model(pixs)

        raw = out["preds"][0] if isinstance(out["preds"], (tuple, list)) else out["preds"]
        all_dets.extend(_decode_yolo_preds_to_coco(raw, metas, conf_thres=0.001, iou_thres=0.65))
        n_imgs += pixs.shape[0]

    infer_sec = max(time.time() - t0, 1e-6)
    coco_m = coco_eval(all_dets, coco_gt)
    f1_m = compute_strict_loose_f1(all_dets, coco_gt, score_thresh=score_thresh_f1, iou_thresh=0.50)

    return {
        "mAP": coco_m.get("map", 0.0),
        "AP50": coco_m.get("map50", 0.0),
        "AP75": coco_m.get("map75", 0.0),
        "AP_small": coco_m.get("map_small", 0.0),
        "AP_medium": coco_m.get("map_medium", 0.0),
        "AP_large": coco_m.get("map_large", 0.0),
        "per_category": coco_m.get("per_category", {}),
        "f1": f1_m,
        "infer_time_sec": round(infer_sec, 2),
        "fps": round(n_imgs / infer_sec, 2),
        "ms_per_page": round(1000.0 * infer_sec / max(n_imgs, 1), 2),
        "num_images": n_imgs,
        "detections": all_dets,
    }
