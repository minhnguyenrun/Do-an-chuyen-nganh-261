"""Method 7 - Dataset: ảnh (tiền xử lý đúng Heron) + heatmap 22 kênh từ DỰ ĐOÁN Base (memmap).

GT chỉ dùng để tạo target cho loss. Heatmap đọc từ cache sinh bởi precompute_base_preds.py.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from methods.method7.common import CACHE_DIR, book_of, load_image_tensor


class Method7Dataset(Dataset):
    def __init__(self, coco: dict, img_dir: Path, split: str, allowed_books: set[str] | None = None,
                 drop_empty: bool = False):
        self.img_dir = img_dir
        self.split = split
        with open(CACHE_DIR / f"heatmap_index_{split}.json") as f:
            self.hm_index = {int(k): v for k, v in json.load(f).items()}
        self._hm = None  # mở memmap lười trong từng worker

        anns = defaultdict(list)
        for a in coco["annotations"]:
            anns[a["image_id"]].append(a)
        self.items = []
        for im in coco["images"]:
            if allowed_books is not None and book_of(im) not in allowed_books:
                continue
            if drop_empty and not anns.get(im["id"]):
                continue
            self.items.append((im, anns.get(im["id"], [])))

    @property
    def hm(self):
        if self._hm is None:
            self._hm = np.load(CACHE_DIR / f"heatmaps_{self.split}.npy", mmap_mode="r")
        return self._hm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        im, anns = self.items[i]
        pix, (h, w) = load_image_tensor(self.img_dir / im["file_name"])
        heat = torch.from_numpy(np.asarray(self.hm[self.hm_index[im["id"]]], dtype=np.float32))
        boxes, cls = [], []
        for a in anns:
            bx, by, bw, bh = a["bbox"]
            if bw < 1 or bh < 1 or not (1 <= a["category_id"] <= 11):
                continue
            cx = min(max((bx + bw / 2) / w, 0.0), 1.0)
            cy = min(max((by + bh / 2) / h, 0.0), 1.0)
            boxes.append([cx, cy, min(bw / w, 1.0), min(bh / h, 1.0)])
            cls.append(a["category_id"] - 1)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "class_labels": torch.tensor(cls, dtype=torch.long),
            "image_id": im["id"],
            "orig_size": torch.tensor([h, w], dtype=torch.float32),
        }
        return pix, heat, target


def collate(batch):
    return (torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch]), [b[2] for b in batch])
