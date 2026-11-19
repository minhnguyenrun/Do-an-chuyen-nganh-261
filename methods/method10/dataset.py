import json
import random
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from tqdm.auto import tqdm

from methods.method10.common import (
    CACHE_DIR,
    TOTAL_FEAT_ELEMS,
    book_of,
    coco_eval,
    load_image_640_tensor,
    to_coco_dets,
    topk_detections,
    unflatten_feats,
)


def precompute_backbone_book_context(
    images_list: List[dict],
    img_640_dir: Path,
    img_raw_dir: Optional[Path],
    model,
    split: str,
    batch_size: int = 16,
    prefix: str = "",
    cache_dir: Path = CACHE_DIR,
    device: Optional[torch.device] = None,
) -> Tuple[Path, Dict[str, dict]]:
    """Precompute per-book mean frozen backbone features into a float16 memmap.

    Groups pages by book so only a single float32 accumulator (8.6 MB) is kept
    in RAM regardless of dataset size.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    memmap_path = cache_dir / f"{prefix}book_ctx_{split}.npy"
    index_path = cache_dir / f"{prefix}book_ctx_index_{split}.json"

    book_to_imgs: Dict[str, List[dict]] = defaultdict(list)
    for img in images_list:
        book_to_imgs[book_of(img)].append(img)
    book_names = sorted(book_to_imgs.keys())

    if memmap_path.exists() and index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            existing_index = json.load(f)
        if set(existing_index.keys()) == set(book_names):
            print(
                f"[Method10 Cache] Loaded existing {split} book context cache: "
                f"{len(book_names)} books -> {memmap_path}"
            )
            return memmap_path, existing_index

    if device is None:
        device = next(model.parameters()).device

    num_books = len(book_names)
    tmp_memmap_path = cache_dir / f"{prefix}book_ctx_{split}.tmp.npy"
    mmap = np.lib.format.open_memmap(
        tmp_memmap_path,
        mode="w+",
        dtype=np.float16,
        shape=(num_books, TOTAL_FEAT_ELEMS),
    )

    # Flatten (book_idx, img_dict) in book-sorted order for O(1) RAM accumulation
    flat_items: List[Tuple[int, dict]] = []
    for b_idx, b_name in enumerate(book_names):
        for img in book_to_imgs[b_name]:
            flat_items.append((b_idx, img))

    model.eval()
    index_dict: Dict[str, dict] = {}
    curr_b_idx = -1
    curr_sum = np.zeros(TOTAL_FEAT_ELEMS, dtype=np.float32)
    curr_count = 0

    t0 = time.time()
    pbar = tqdm(
        range(0, len(flat_items), batch_size),
        desc=f"Precompute {split} book ctx ({num_books} books, {len(flat_items)} pages)",
        leave=True,
    )

    with torch.no_grad():
        for start in pbar:
            batch_items = flat_items[start : start + batch_size]
            pix_list = []
            for _, img in batch_items:
                fn = img["file_name"]
                p640 = img_640_dir / fn
                praw = (img_raw_dir / fn) if img_raw_dir is not None else None
                pix_list.append(load_image_640_tensor(p640, praw))

            pix_batch = torch.stack(pix_list, dim=0).to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                feats = model.extract_frozen_feats(pix_batch)

            flat_batch = torch.cat([f.flatten(1) for f in feats], dim=1).float().cpu().numpy()

            for i, (b_idx, _) in enumerate(batch_items):
                if b_idx != curr_b_idx:
                    if curr_b_idx >= 0:
                        mmap[curr_b_idx] = (curr_sum / max(curr_count, 1)).astype(np.float16)
                        index_dict[book_names[curr_b_idx]] = {
                            "row": int(curr_b_idx),
                            "count": int(curr_count),
                        }
                    curr_b_idx = b_idx
                    curr_sum.fill(0.0)
                    curr_count = 0
                curr_sum += flat_batch[i]
                curr_count += 1

    if curr_b_idx >= 0:
        mmap[curr_b_idx] = (curr_sum / max(curr_count, 1)).astype(np.float16)
        index_dict[book_names[curr_b_idx]] = {
            "row": int(curr_b_idx),
            "count": int(curr_count),
        }

    mmap.flush()
    del mmap

    if memmap_path.exists():
        memmap_path.unlink()
    tmp_memmap_path.rename(memmap_path)

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_dict, f)

    dt = time.time() - t0
    print(
        f"[Method10 Cache] Saved {split} book context: {num_books} books, "
        f"{len(flat_items)} pages in {dt:.1f}s -> {memmap_path}"
    )
    return memmap_path, index_dict


class Method10FastDataset(Dataset):
    """Dataset returning 640x640 image, precomputed float16 book context, book count, and DETR labels."""

    def __init__(
        self,
        coco_dict: dict,
        img_640_dir: Path,
        img_raw_dir: Optional[Path],
        memmap_path: Path,
        book_index: Dict[str, dict],
    ):
        self.images = coco_dict["images"]
        self.img_640_dir = Path(img_640_dir)
        self.img_raw_dir = Path(img_raw_dir) if img_raw_dir is not None else None
        self.memmap_path = Path(memmap_path)
        self.book_index = book_index
        self._mmap = None
        self._row_cache: Optional[OrderedDict] = None
        self._max_cached_books = 8

        self.ann_by_img = defaultdict(list)
        for ann in coco_dict.get("annotations", []):
            self.ann_by_img[ann["image_id"]].append(ann)

    @property
    def ctx_mmap(self):
        if self._mmap is None:
            self._mmap = np.load(self.memmap_path, mmap_mode="r")
        return self._mmap

    def _get_book_ctx(self, row: int) -> torch.Tensor:
        if self._row_cache is None:
            self._row_cache = OrderedDict()
        cached = self._row_cache.get(row)
        if cached is not None:
            self._row_cache.move_to_end(row)
            return cached
        tensor = torch.from_numpy(np.array(self.ctx_mmap[row], copy=True))
        self._row_cache[row] = tensor
        if len(self._row_cache) > self._max_cached_books:
            self._row_cache.popitem(last=False)
        return tensor

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        fn = img["file_name"]
        p640 = self.img_640_dir / fn
        praw = (self.img_raw_dir / fn) if self.img_raw_dir is not None else None
        pix = load_image_640_tensor(p640, praw)

        b_name = book_of(img)
        meta = self.book_index[b_name]
        row = meta["row"]
        count = meta["count"]
        flat_ctx = self._get_book_ctx(row)

        w_img = float(img["width"])
        h_img = float(img["height"])
        boxes = []
        classes = []
        for ann in self.ann_by_img.get(img["id"], []):
            x, y, bw, bh = ann["bbox"]
            if bw <= 0 or bh <= 0:
                continue
            cx = (x + bw * 0.5) / w_img
            cy = (y + bh * 0.5) / h_img
            nw = bw / w_img
            nh = bh / h_img
            boxes.append([cx, cy, nw, nh])
            classes.append(int(ann["category_id"]) - 1)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "class_labels": torch.tensor(classes, dtype=torch.long) if classes else torch.zeros((0,), dtype=torch.long),
            "image_id": torch.tensor([img["id"]], dtype=torch.long),
            "orig_size": torch.tensor([int(h_img), int(w_img)], dtype=torch.long),
        }
        return pix, flat_ctx, count, target


def collate_fn_m10(batch):
    pixs = torch.stack([b[0] for b in batch], dim=0)
    flat_ctxs = torch.stack([b[1] for b in batch], dim=0)
    book_means = unflatten_feats(flat_ctxs)
    book_counts = torch.tensor([b[2] for b in batch], dtype=torch.long)
    labels = [b[3] for b in batch]
    return pixs, book_means, book_counts, labels


class BookGroupedBatchSampler(Sampler[List[int]]):
    """BatchSampler that shuffles at the BOOK level and keeps pages of the same book adjacent.

    This allows Method10FastDataset's in-RAM LRU book cache to reuse the 4.3 MB book_ctx
    across all pages of the same book with minimal SSD I/O.
    """

    def __init__(
        self,
        dataset: Method10FastDataset,
        batch_size: int,
        seed: int = 42,
        shuffle_books: bool = True,
        shuffle_pages_in_book: bool = True,
    ):
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.shuffle_books = shuffle_books
        self.shuffle_pages_in_book = shuffle_pages_in_book

        book_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, img in enumerate(dataset.images):
            book_to_indices[book_of(img)].append(idx)

        self.book_groups = [book_to_indices[b] for b in sorted(book_to_indices.keys())]
        self.num_samples = len(dataset.images)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        groups = [list(g) for g in self.book_groups]
        if self.shuffle_books:
            rng.shuffle(groups)
        ordered_indices: List[int] = []
        for g in groups:
            if self.shuffle_pages_in_book:
                rng.shuffle(g)
            ordered_indices.extend(g)

        for i in range(0, len(ordered_indices), self.batch_size):
            yield ordered_indices[i : i + self.batch_size]

    def __len__(self) -> int:
        return (self.num_samples + self.batch_size - 1) // self.batch_size


@torch.no_grad()
def evaluate_method10_coco_map(
    model,
    data_loader,
    coco_gt: dict,
    device: torch.device,
    use_amp: bool = True,
    desc: str = "COCO Eval",
) -> dict:
    """Run exact COCO evaluation over every image in data_loader."""
    model.eval()
    all_dets = []
    t0 = time.time()

    for pixs, book_means, book_counts, labels in tqdm(data_loader, desc=desc, leave=False):
        pixs = pixs.to(device, non_blocking=True)
        book_means = [m.to(device, non_blocking=True) for m in book_means]
        book_counts = book_counts.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(use_amp and device.type == "cuda")):
            out = model(pixs, book_means, book_counts)

        probs = out["logits"].float().sigmoid()
        boxes = out["pred_boxes"].float()

        for i in range(pixs.shape[0]):
            img_id = int(labels[i]["image_id"].item())
            orig_h = int(labels[i]["orig_size"][0].item())
            orig_w = int(labels[i]["orig_size"][1].item())
            cls_ids, scores, det_boxes = topk_detections(probs[i], boxes[i], k=100)
            all_dets.extend(to_coco_dets(img_id, cls_ids, scores, det_boxes, orig_h, orig_w))

    metrics = coco_eval(all_dets, coco_gt)
    metrics["mAP"] = metrics["map"]
    metrics["AP50"] = metrics["map50"]
    metrics["AP75"] = metrics["map75"]
    metrics["AP_small"] = metrics["map_small"]
    metrics["AP_medium"] = metrics["map_medium"]
    metrics["AP_large"] = metrics["map_large"]
    metrics["per_class"] = {k: v["ap"] for k, v in metrics.get("per_category", {}).items()}
    metrics["eval_time_sec"] = round(time.time() - t0, 2)
    metrics["num_images"] = len(coco_gt["images"])
    metrics["num_detections"] = len(all_dets)
    return metrics
