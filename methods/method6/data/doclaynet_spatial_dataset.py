"""DocLayNet Spatial Heatmap Dataset for Method 6.

Xây dựng bản đồ bố cục 2D (Spatial Layout Heatmap) kích thước 80x80 gồm 22 kênh:
- Kênh 0 - 10: Book Prior (Tần suất xuất hiện của 11 class qua 64 trang ngữ cảnh của sách).
- Kênh 11 - 21: Page Prior (Bố cục các vùng phát hiện của trang hiện tại).
"""

from __future__ import annotations

import json
import sys
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

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


def render_page_heatmap(
    annotations: list[dict[str, Any]],
    orig_w: float,
    orig_h: float,
    grid_size: int = 80,
    num_classes: int = 11,
) -> np.ndarray:
    """Vẽ bounding box của trang lên grid_size x grid_size heatmap cho 11 class."""
    heatmap = np.zeros((num_classes, grid_size, grid_size), dtype=np.float32)
    orig_w = max(1.0, float(orig_w))
    orig_h = max(1.0, float(orig_h))

    for ann in annotations:
        cid = ann.get("category_id")
        if cid not in LABEL_TO_HERON_ID:
            continue
        c_idx = LABEL_TO_HERON_ID[cid]

        bx, by, bw, bh = ann["bbox"]
        x1 = int(np.clip((bx / orig_w) * grid_size, 0, grid_size - 1))
        y1 = int(np.clip((by / orig_h) * grid_size, 0, grid_size - 1))
        x2 = int(np.clip(((bx + bw) / orig_w) * grid_size, 0, grid_size))
        y2 = int(np.clip(((by + bh) / orig_h) * grid_size, 0, grid_size))

        if x2 > x1 and y2 > y1:
            heatmap[c_idx, y1:y2, x1:x2] = 1.0

    return heatmap


class DocLayNetSpatialDataset(Dataset):
    def __init__(
        self,
        annotation_file: str | Path,
        images_dir: str | Path,
        image_size: int = 640,
        heatmap_size: int = 80,
        window_size: int = 64,
        limit_pages: int | None = None,
        is_train: bool = True,
        allowed_books: list[str] | set[str] | None = None,
    ):
        super().__init__()
        self.annotation_file = Path(annotation_file)
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.window_size = window_size
        self.is_train = is_train
        self.allowed_books = set(allowed_books) if allowed_books is not None else None

        print(f"[DocLayNetSpatialDataset] Nạp annotation từ: {self.annotation_file}")
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

        # Tạo chuỗi sliding window 64 trang
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

        if self.is_train:
            self.samples = [s for s in self.samples if len(s["target_page"]["annotations"]) > 0]

        if limit_pages is not None and limit_pages > 0:
            self.samples = self.samples[:limit_pages]

        print(f"[DocLayNetSpatialDataset] Sẵn sàng: {len(self.samples):,} trang từ {len(books_dict):,} cuốn sách (Cửa sổ: {self.window_size} trang).")

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[idx]
        target_info = item["target_page"]
        context_infos = item["context_pages"]

        # 1. Ảnh chính sạch
        img_path = self.images_dir / target_info["file_name"]
        if not img_path.exists():
            image = Image.new("RGB", (self.image_size, self.image_size), color=(255, 255, 255))
        else:
            image = Image.open(img_path).convert("RGB")

        pixel_values = self.transform(image)

        # 2. Tạo 2D Layout Heatmap (22 kênh: 11 Book Prior + 11 Page Prior)
        # 2.1 Book Prior: Trung bình mật độ layout của các trang trong cửa sổ sách
        book_heatmap = np.zeros((11, self.heatmap_size, self.heatmap_size), dtype=np.float32)
        for ctx_p in context_infos:
            p_map = render_page_heatmap(
                ctx_p["annotations"],
                ctx_p["width"],
                ctx_p["height"],
                grid_size=self.heatmap_size,
                num_classes=11,
            )
            book_heatmap += p_map
        if len(context_infos) > 0:
            book_heatmap /= float(len(context_infos))

        # 2.2 Page Prior: Bản đồ layout riêng của trang hiện tại
        page_heatmap = render_page_heatmap(
            target_info["annotations"],
            target_info["width"],
            target_info["height"],
            grid_size=self.heatmap_size,
            num_classes=11,
        )

        # Ghép 22 kênh
        layout_heatmap = np.concatenate([book_heatmap, page_heatmap], axis=0)  # [22, 80, 80]
        layout_heatmap_tensor = torch.from_numpy(layout_heatmap)

        # 3. Ground truth labels cho Hungarian Matcher
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
            if bw < 2.0 or bh < 2.0:
                continue

            cx = (bx + bw / 2.0) / w_orig
            cy = (by + bh / 2.0) / h_orig
            nw = float(np.clip(bw / w_orig, 1e-3, 1.0))
            nh = float(np.clip(bh / h_orig, 1e-3, 1.0))
            cx = float(np.clip(cx, 1e-3, 1.0 - 1e-3))
            cy = float(np.clip(cy, 1e-3, 1.0 - 1e-3))

            gt_boxes.append([cx, cy, nw, nh])
            gt_classes.append(heron_id)

        if not gt_boxes:
            target_boxes = torch.zeros((0, 4), dtype=torch.float32)
            target_classes = torch.zeros((0,), dtype=torch.long)
        else:
            target_boxes = torch.tensor(gt_boxes, dtype=torch.float32).clamp(0.0, 1.0)
            target_classes = torch.tensor(gt_classes, dtype=torch.long)

        target = {
            "boxes": target_boxes,
            "class_labels": target_classes,
            "image_id": target_info["id"],
            "orig_size": torch.tensor([h_orig, w_orig], dtype=torch.float32),
        }

        return {
            "pixel_values": pixel_values,
            "layout_heatmap": layout_heatmap_tensor,
            "target": target,
        }


def collate_spatial_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0)
    layout_heatmaps = torch.stack([item["layout_heatmap"] for item in batch], dim=0)
    targets = [item["target"] for item in batch]

    return {
        "pixel_values": pixel_values,
        "layout_heatmaps": layout_heatmaps,
        "targets": targets,
    }
