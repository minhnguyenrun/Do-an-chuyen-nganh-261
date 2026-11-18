"""DocLayNet Book Dataset for Method 5 with 64-Page Window & OCR Features.

Hỗ trợ:
- Cửa sổ ngữ cảnh sách mở rộng lên tới 64 trang (window_size=64).
- Trích xuất OCR statistical features (8-dim) cho từng bounding box.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

LABEL_TO_HERON_ID = {
    1: 0,   # Caption
    2: 1,   # Footnote
    3: 2,   # Formula
    4: 3,   # List-item
    5: 4,   # Page-footer
    6: 5,   # Page-header
    7: 6,   # Picture
    8: 7,   # Section-header
    9: 8,   # Table
    10: 9,  # Text
    11: 10, # Title
}


class DocLayNetBookDataset(Dataset):
    def __init__(
        self,
        annotation_file: str | Path,
        images_dir: str | Path,
        image_size: int = 640,
        window_size: int = 64,
        limit_pages: int | None = None,
        is_train: bool = True,
        allowed_books: list[str] | set[str] | None = None,
    ):
        super().__init__()
        self.annotation_file = Path(annotation_file)
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.window_size = window_size
        self.is_train = is_train
        self.allowed_books = set(allowed_books) if allowed_books is not None else None

        print(f"[DocLayNetBookDataset] Loading annotations from: {self.annotation_file}")
        with open(self.annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}

        annotations_by_image = defaultdict(list)
        for ann in data.get("annotations", []):
            annotations_by_image[ann["image_id"]].append(ann)

        books_dict = defaultdict(list)
        for img in data.get("images", []):
            orig_name = img.get("original_filename")
            if not orig_name:
                orig_name = "_".join(img["file_name"].split("_")[:-1])

            if self.allowed_books is not None and orig_name not in self.allowed_books:
                continue

            page_no = img.get("page_no", 0)

            img_entry = {
                "id": img["id"],
                "file_name": img["file_name"],
                "width": img["width"],
                "height": img["height"],
                "page_no": page_no,
                "annotations": annotations_by_image.get(img["id"], []),
            }
            books_dict[orig_name].append(img_entry)

        # Tạo chuỗi sliding window lên tới 64 trang
        self.samples = []
        for book_name, pages in books_dict.items():
            pages.sort(key=lambda x: x["page_no"])
            for idx, p in enumerate(pages):
                start_idx = max(0, idx - self.window_size // 2)
                end_idx = min(len(pages), idx + self.window_size // 2 + 1)
                context_pages = pages[start_idx:end_idx]

                self.samples.append({
                    "target_page": p,
                    "context_pages": context_pages,
                    "book_name": book_name,
                    "target_idx_in_context": idx - start_idx,
                })

        if limit_pages is not None and limit_pages > 0:
            self.samples = self.samples[:limit_pages]

        print(f"[DocLayNetBookDataset] Loaded {len(self.samples):,} pages across {len(books_dict):,} books (Window size: {self.window_size}).")

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def _extract_page_layout_elements(self, page_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
        w_orig = max(1.0, float(page_dict["width"]))
        h_orig = max(1.0, float(page_dict["height"]))

        boxes = []
        cls_logits = []
        ocr_stats = []

        for ann in page_dict["annotations"]:
            cid = ann["category_id"]
            if cid not in LABEL_TO_HERON_ID:
                continue
            heron_id = LABEL_TO_HERON_ID[cid]

            bx, by, bw, bh = ann["bbox"]
            cx = (bx + bw / 2.0) / w_orig
            cy = (by + bh / 2.0) / h_orig
            nw = bw / w_orig
            nh = bh / h_orig

            boxes.append([cx, cy, nw, nh])

            one_hot = [0.0] * 17
            one_hot[heron_id] = 1.0
            cls_logits.append(one_hot)

            # 8-dim OCR / Text features
            area_norm = nw * nh
            aspect_r = nw / max(1e-4, nh)
            is_wide = 1.0 if aspect_r > 2.0 else 0.0
            is_tall = 1.0 if aspect_r < 0.5 else 0.0
            pos_y_rel = cy
            pos_x_rel = cx
            is_header_zone = 1.0 if cy < 0.15 else 0.0
            is_footer_zone = 1.0 if cy > 0.85 else 0.0

            ocr_stats.append([
                area_norm, aspect_r, is_wide, is_tall,
                pos_y_rel, pos_x_rel, is_header_zone, is_footer_zone,
            ])

        if not boxes:
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "cls_logits": torch.zeros((0, 17), dtype=torch.float32),
                "ocr_stats": torch.zeros((0, 8), dtype=torch.float32),
                "page_rel_idx": page_dict.get("page_no", 0),
            }

        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32).clamp(0.0, 1.0),
            "cls_logits": torch.tensor(cls_logits, dtype=torch.float32),
            "ocr_stats": torch.tensor(ocr_stats, dtype=torch.float32),
            "page_rel_idx": page_dict.get("page_no", 0),
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[idx]
        target_info = item["target_page"]
        context_infos = item["context_pages"]

        # 1. Ảnh chính
        img_path = self.images_dir / target_info["file_name"]
        if not img_path.exists():
            image = Image.new("RGB", (self.image_size, self.image_size), color=(255, 255, 255))
        else:
            image = Image.open(img_path).convert("RGB")

        pixel_values = self.transform(image)

        # 2. Ground truth labels
        w_orig = max(1.0, float(target_info["width"]))
        h_orig = max(1.0, float(target_info["height"]))

        gt_boxes = []
        gt_classes = []

        for ann in target_info["annotations"]:
            cid = ann["category_id"]
            if cid not in LABEL_TO_HERON_ID:
                continue
            heron_id = LABEL_TO_HERON_ID[cid]

            bx, by, bw, bh = ann["bbox"]
            cx = (bx + bw / 2.0) / w_orig
            cy = (by + bh / 2.0) / h_orig
            nw = bw / w_orig
            nh = bh / h_orig

            gt_boxes.append([cx, cy, nw, nh])
            gt_classes.append(heron_id)

        if not gt_boxes:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            classes_tensor = torch.zeros((0,), dtype=torch.long)
        else:
            boxes_tensor = torch.tensor(gt_boxes, dtype=torch.float32).clamp(0.0, 1.0)
            classes_tensor = torch.tensor(gt_classes, dtype=torch.long)

        # 3. Context elements lên tới 64 trang
        context_elements = [
            self._extract_page_layout_elements(p) for p in context_infos
        ]

        return {
            "pixel_values": pixel_values,
            "labels": {
                "class_labels": classes_tensor,
                "boxes": boxes_tensor,
            },
            "context_elements": context_elements,
            "file_name": target_info["file_name"],
            "page_no": target_info["page_no"],
            "image_id": target_info["id"],
            "orig_size": (target_info["width"], target_info["height"]),
            "annotations": target_info["annotations"],
        }


def collate_book_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = [b["labels"] for b in batch]
    context_elements = [b["context_elements"] for b in batch]
    file_names = [b["file_name"] for b in batch]
    image_ids = [b["image_id"] for b in batch]
    orig_sizes = [b["orig_size"] for b in batch]
    annotations = [b["annotations"] for b in batch]

    return {
        "pixel_values": pixel_values,
        "labels": labels,
        "context_elements": context_elements,
        "file_names": file_names,
        "image_ids": image_ids,
        "orig_sizes": orig_sizes,
        "annotations": annotations,
    }
