"""Dataset and Evaluation utilities for Method 12."""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from methods.method12.common import (
    IMAGE_SIZE,
    coco_eval,
    compute_strict_loose_f1,
    load_image_640_tensor,
    to_coco_dets,
    topk_detections,
)


class DocLayNet640Dataset(Dataset):
    """Fast single-page dataset loading pre-resized 640x640 images from SSD."""

    def __init__(
        self,
        coco: dict,
        img_640_dir: Path,
        img_raw_dir: Optional[Path] = None,
        allowed_image_ids: Optional[Set[int]] = None,
        augment: bool = False,
        max_samples: Optional[int] = None,
    ):
        self.img_640_dir = img_640_dir
        self.img_raw_dir = img_raw_dir
        self.augment = augment

        anns_by_img: Dict[int, List[dict]] = defaultdict(list)
        for a in coco.get("annotations", []):
            anns_by_img[a["image_id"]].append(a)

        self.items = []
        for im in coco["images"]:
            if allowed_image_ids is not None and im["id"] not in allowed_image_ids:
                continue
            self.items.append((im, anns_by_img.get(im["id"], [])))
            if max_samples is not None and len(self.items) >= max_samples:
                break

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        im, anns = self.items[idx]
        stem = Path(im["file_name"]).stem
        p640 = self.img_640_dir / f"{stem}.jpg"
        praw = (self.img_raw_dir / im["file_name"]) if self.img_raw_dir is not None else None
        pix = load_image_640_tensor(p640, praw, augment=self.augment)

        h = float(im.get("height", 1025))
        w = float(im.get("width", 1025))

        boxes, classes = [], []
        for a in anns:
            bx, by, bw, bh = a["bbox"]
            if bw <= 0 or bh <= 0:
                continue
            cx = max(0.0, min(1.0, (bx + bw / 2.0) / w))
            cy = max(0.0, min(1.0, (by + bh / 2.0) / h))
            nw = max(1e-4, min(1.0, bw / w))
            nh = max(1e-4, min(1.0, bh / h))
            boxes.append([cx, cy, nw, nh])
            classes.append(int(a["category_id"]) - 1)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "class_labels": torch.tensor(classes, dtype=torch.long) if classes else torch.zeros((0,), dtype=torch.long),
            "image_id": torch.tensor([im["id"]], dtype=torch.long),
            "orig_size": torch.tensor([int(h), int(w)], dtype=torch.long),
        }
        return pix, target


def collate_fn_det(batch):
    pixs = torch.stack([b[0] for b in batch], dim=0)
    labels = [b[1] for b in batch]
    return pixs, labels


@torch.no_grad()
def evaluate_stage01_model(
    model,
    data_loader,
    coco_gt: dict,
    device: torch.device,
    use_amp: bool = True,
    score_thresh_f1: float = 0.35,
    desc: str = "Eval",
) -> dict:
    """Evaluate COCO mAP + Strict/Loose F1 + inference FPS."""
    model.eval()
    all_dets = []
    t0 = time.time()
    n_imgs = 0

    for pixs, labels in tqdm(data_loader, desc=desc, leave=False):
        pixs = pixs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(use_amp and device.type == "cuda")):
            out = model(pixs)

        probs = out["logits"].float().sigmoid()
        boxes = out["pred_boxes"].float()

        for i in range(pixs.shape[0]):
            img_id = int(labels[i]["image_id"].item())
            orig_h = int(labels[i]["orig_size"][0].item())
            orig_w = int(labels[i]["orig_size"][1].item())
            cls_ids, scores, det_boxes = topk_detections(probs[i], boxes[i], k=100)
            all_dets.extend(to_coco_dets(img_id, cls_ids, scores, det_boxes, orig_h, orig_w))
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
    }
