"""Dataset and Data Processing for BBox Multimodal Transformer (Pipeline B)."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# 11 nhãn chuẩn của tập dữ liệu DocLayNet
CATEGORY_NAMES = {
    1: "Caption",
    2: "Footnote",
    3: "Formula",
    4: "List-item",
    5: "Page-footer",
    6: "Page-header",
    7: "Picture",
    8: "Section-header",
    9: "Table",
    10: "Text",
    11: "Title",
}


class DocLayNetBBoxDataset(Dataset):
    """Dataset nạp trang tài liệu và trích xuất đồng thời 4 phương thức đặc trưng cho từng BBox."""

    def __init__(
        self,
        annotation_file: Path,
        images_dir: Path,
        max_boxes_per_page: int = 64,
        crop_size: Tuple[int, int] = (64, 128),  # (H, W)
        text_embedder: Optional[Any] = None,
        limit_samples: Optional[int] = None,
    ):
        self.images_dir = Path(images_dir)
        self.max_boxes = max_boxes_per_page
        self.crop_size = crop_size
        self.text_embedder = text_embedder

        # Transform chuẩn hóa ảnh crop cho visual backbone
        self.transform = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Nạp annotations COCO
        with open(annotation_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images = coco.get("images", [])
        if limit_samples:
            self.images = self.images[:limit_samples]

        self.anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
        for ann in coco.get("annotations", []):
            self.anns_by_img.setdefault(ann["image_id"], []).append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_info = self.images[idx]
        img_id = img_info["id"]
        img_path = self.images_dir / img_info["file_name"]

        base_img = Image.open(img_path).convert("RGB")
        img_w, img_h = base_img.size

        anns = self.anns_by_img.get(img_id, [])[:self.max_boxes]
        num_boxes = len(anns)

        crops_list = []
        spatial_list = []
        labels_list = []
        text_strings = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            w = max(1.0, float(w))
            h = max(1.0, float(h))
            x0 = max(0.0, float(x))
            y0 = max(0.0, float(y))
            x1 = min(float(img_w), x0 + w)
            y1 = min(float(img_h), y0 + h)

            # 1. Trích xuất và tiền xử lý ảnh crop
            crop = base_img.crop((int(x0), int(y0), max(int(x0) + 1, int(x1)), max(int(y0) + 1, int(y1))))
            crop_tensor = self.transform(crop)
            crops_list.append(crop_tensor)

            # 2. Đặc trưng hình học không gian 2D (Spatial Coordinates)
            # Chuẩn hóa về [0, 1] đối chiếu với toàn trang
            norm_x0 = x0 / img_w
            norm_y0 = y0 / img_h
            norm_x1 = x1 / img_w
            norm_y1 = y1 / img_h
            norm_w = w / img_w
            norm_h = h / img_h
            aspect_ratio = min(5.0, w / max(h, 1e-4))
            area_ratio = (w * h) / (img_w * img_h)

            spatial_vec = [norm_x0, norm_y0, norm_x1, norm_y1, norm_w, norm_h, aspect_ratio, area_ratio]
            spatial_list.append(spatial_vec)

            # 3. Nhãn phân loại (0 đến 10 tương ứng với 11 categories)
            cat_id = ann.get("category_id", 10)  # Mặc định Text
            labels_list.append(max(0, min(10, cat_id - 1)))

            # 4. Trích xuất text string nếu có
            text_strings.append(ann.get("text", ""))

        # 5. Mã hóa văn bản thành dense vectors
        if self.text_embedder is not None and text_strings:
            text_embeds = self.text_embedder(text_strings)  # (num_boxes, text_dim)
        else:
            # Fallback vector nếu chạy offline / chưa nạp NLP model
            text_embeds = np.zeros((num_boxes, 384), dtype=np.float32)

        return {
            "image_id": img_id,
            "file_name": img_info["file_name"],
            "width": img_w,
            "height": img_h,
            "num_boxes": num_boxes,
            "crops": torch.stack(crops_list) if crops_list else torch.zeros(0, 3, *self.crop_size),
            "spatial": torch.tensor(spatial_list, dtype=torch.float32) if spatial_list else torch.zeros(0, 8),
            "text_embeds": torch.tensor(text_embeds, dtype=torch.float32) if isinstance(text_embeds, np.ndarray) else text_embeds,
            "labels": torch.tensor(labels_list, dtype=torch.long) if labels_list else torch.zeros(0, dtype=torch.long),
            "raw_bboxes": [ann["bbox"] for ann in anns],
        }


def pad_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collator gom batch nhiều trang tài liệu với số lượng hộp khác nhau."""
    batch_size = len(batch)
    max_n = max(max(b["num_boxes"] for b in batch), 1)

    crops_padded = torch.zeros(batch_size, max_n, 3, 64, 128, dtype=torch.float32)
    spatial_padded = torch.zeros(batch_size, max_n, 8, dtype=torch.float32)
    text_padded = torch.zeros(batch_size, max_n, 384, dtype=torch.float32)
    labels_padded = torch.full((batch_size, max_n), -100, dtype=torch.long)  # -100 để bỏ qua trong CrossEntropyLoss
    mask = torch.ones(batch_size, max_n, dtype=torch.bool)  # True = padded slot (bị che)

    for i, b in enumerate(batch):
        n = b["num_boxes"]
        if n > 0:
            crops_padded[i, :n] = b["crops"]
            spatial_padded[i, :n] = b["spatial"]
            text_padded[i, :n] = b["text_embeds"]
            labels_padded[i, :n] = b["labels"]
            mask[i, :n] = False  # False = hộp hợp lệ

    return {
        "crops": crops_padded,
        "spatial": spatial_padded,
        "text_embeds": text_padded,
        "labels": labels_padded,
        "mask": mask,
        "batch_meta": batch,
    }
