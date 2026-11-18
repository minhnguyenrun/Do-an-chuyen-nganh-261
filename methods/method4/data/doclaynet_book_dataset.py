"""Dataset and DataLoader for Book Layout RT-DETR.

Gom các trang thuộc cùng 1 tài liệu (original_filename / book) thành chuỗi trang liên tục.
Hỗ trợ:
- Lấy ảnh trang hiện tại (Target Page) và nhãn Ground Truth (COCO format -> xywh chuẩn hóa [0, 1]).
- Lấy layout các trang lân cận (Context Window) để đưa vào BookLayoutSemanticEncoder.
- Chuẩn hóa tensor ảnh cho RT-DETR [3, 640, 640] và tạo candidate layout từ RT-DETRv2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from collections import defaultdict
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class DocLayNetBookDataset(Dataset):
    """
    Dataset tài liệu nhiều trang theo phong cách Book-Level cho RT-DETRv2.
    """

    def __init__(
        self,
        annotation_file: str | Path,
        images_dir: str | Path,
        image_size: int = 640,
        window_size: int = 4,
        limit_pages: int | None = None,
        is_train: bool = True,
    ):
        super().__init__()
        self.annotation_file = Path(annotation_file)
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.window_size = window_size
        self.is_train = is_train

        print(f"[DocLayNetBookDataset] Loading annotations from: {self.annotation_file}")
        with open(self.annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}

        # Ánh xạ ID từ DocLayNet (1-11) sang RT-DETR heron class (0-16)
        # DocLayNet: 1: Caption, 2: Footnote, 3: Formula, 4: List-item, 5: Page-footer, 
        #            6: Page-header, 7: Picture, 8: Section-header, 9: Table, 10: Text, 11: Title
        # Heron label2id:
        #   'caption': 0, 'footnote': 1, 'formula': 2, 'list_item': 3, 'page_footer': 4,
        #   'page_header': 5, 'picture': 6, 'section_header': 7, 'table': 8, 'text': 9, 'title': 10
        self.doclaynet_to_heron_id = {
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

        # Gom nhóm annotations theo image_id
        annotations_by_img = defaultdict(list)
        for ann in data.get("annotations", []):
            if ann.get("iscrowd", 0) == 0:
                annotations_by_img[ann["image_id"]].append(ann)

        # Gom nhóm ảnh theo tài liệu (original_filename) và sắp xếp theo page_no
        books = defaultdict(list)
        images = data.get("images", [])
        if limit_pages and limit_pages > 0:
            images = images[:limit_pages]

        for img in images:
            img_id = img["id"]
            img_anns = annotations_by_img[img_id]
            doc_key = img.get("original_filename") or f"doc_{img_id // 100}"
            page_no = img.get("page_no", img_id)
            books[doc_key].append({
                "id": img_id,
                "file_name": img["file_name"],
                "width": img["width"],
                "height": img["height"],
                "page_no": page_no,
                "annotations": img_anns,
            })

        # Sắp xếp các trang trong mỗi tài liệu
        self.flat_pages = []
        for doc_key, page_list in books.items():
            page_list.sort(key=lambda p: p["page_no"])
            for idx, p in enumerate(page_list):
                # Lưu ngữ cảnh các trang lân cận (trang trước và trang sau trong cùng tài liệu)
                start_w = max(0, idx - self.window_size // 2)
                end_w = min(len(page_list), idx + self.window_size // 2 + 1)
                context_pages = page_list[start_w:end_w]

                self.flat_pages.append({
                    "target_page": p,
                    "context_pages": context_pages,
                })

        print(f"[DocLayNetBookDataset] Loaded {len(self.flat_pages):,} pages across {len(books):,} books.")

    def __len__(self) -> int:
        return len(self.flat_pages)

    def _extract_page_layout_elements(self, page_info: dict) -> dict[str, torch.Tensor]:
        """Tạo đặc trưng layout của một trang để đưa vào BookLayoutSemanticEncoder."""
        anns = page_info["annotations"]
        w_orig = max(1.0, float(page_info["width"]))
        h_orig = max(1.0, float(page_info["height"]))

        boxes_list = []
        class_list = []

        for ann in anns:
            bbox = ann["bbox"]  # [x, y, w, h]
            cat_id = ann["category_id"]
            heron_id = self.doclaynet_to_heron_id.get(cat_id, 9)  # default: text

            x, y, bw, bh = bbox
            # Chuẩn hóa về [cx, cy, w, h] trong [0, 1]
            cx = (x + bw / 2.0) / w_orig
            cy = (y + bh / 2.0) / h_orig
            nw = bw / w_orig
            nh = bh / h_orig

            boxes_list.append([cx, cy, nw, nh])
            class_list.append(heron_id)

        if not boxes_list:
            # Trang trống: tạo một box giả nhỏ
            return {
                "bboxes": torch.tensor([[0.5, 0.5, 0.01, 0.01]], dtype=torch.float32),
                "class_logits": torch.zeros(1, 17, dtype=torch.float32),
                "visual_feats": torch.zeros(1, 256, dtype=torch.float32),
            }

        boxes_t = torch.tensor(boxes_list, dtype=torch.float32).clamp(0.0, 1.0)
        class_t = torch.tensor(class_list, dtype=torch.long)
        one_hot = torch.zeros(len(class_list), 17, dtype=torch.float32)
        one_hot.scatter_(1, class_t.unsqueeze(1), 1.0)

        # Giả lập đặc trưng thị giác ban đầu (pseudo visual embed)
        vis_feats = torch.zeros(len(class_list), 256, dtype=torch.float32)

        return {
            "bboxes": boxes_t,
            "class_logits": one_hot,
            "visual_feats": vis_feats,
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.flat_pages[idx]
        target_info = item["target_page"]
        context_infos = item["context_pages"]

        # 1. Đọc và resize ảnh trang hiện tại
        img_path = self.images_dir / target_info["file_name"]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            # Resize về kích thước chuẩn của RT-DETR (640x640)
            img_resized = img.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            pixel_values = TF.to_tensor(img_resized)  # [3, 640, 640], range [0, 1]

        # 2. Xử lý Ground Truth cho RT-DETR loss
        w_orig = max(1.0, float(target_info["width"]))
        h_orig = max(1.0, float(target_info["height"]))
        gt_boxes = []
        gt_classes = []

        for ann in target_info["annotations"]:
            x, y, bw, bh = ann["bbox"]
            cat_id = ann["category_id"]
            heron_id = self.doclaynet_to_heron_id.get(cat_id, 9)

            # Format [cx, cy, w, h] chuẩn hóa [0, 1] cho RT-DETR
            cx = (x + bw / 2.0) / w_orig
            cy = (y + bh / 2.0) / h_orig
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

        # 3. Trích xuất Layout các trang trong cửa sổ sách
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
        }


def collate_book_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collate_fn cho RT-DETR và Book Context."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = [b["labels"] for b in batch]
    context_elements = [b["context_elements"] for b in batch]
    file_names = [b["file_name"] for b in batch]
    image_ids = [b["image_id"] for b in batch]

    return {
        "pixel_values": pixel_values,
        "labels": labels,
        "context_elements": context_elements,
        "file_names": file_names,
        "image_ids": image_ids,
    }
