"""Pipeline 1: Frozen Docling Layout Scout for Full-Book Structural Extraction.

Chạy Docling (RT-DETRv2 pretrained trên DocLayNet) ở chế độ đóng băng (Frozen / No-Grad)
qua toàn bộ các trang P_1, P_2, ..., P_N của cuốn sách hoặc tài liệu để trích xuất:
- Bounding boxes chuẩn hóa [x_min, y_min, x_max, y_max] thuộc [0, 1]
- Nhãn dự đoán sơ bộ (11 classes DocLayNet)
- Confidence scores
- Đặc trưng hình học [x_min, y_min, x_max, y_max, w, h, aspect_ratio, area_ratio]
Hỗ trợ cơ chế lưu và đọc từ bộ nhớ đệm (Offline JSON Cache) để tiết kiệm thời gian và VRAM.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch

DOCLING_LABEL_TO_ID = {
    "caption": 1,
    "footnote": 2,
    "formula": 3,
    "list_item": 4,
    "page_footer": 5,
    "page_header": 6,
    "picture": 7,
    "section_header": 8,
    "table": 9,
    "text": 10,
    "title": 11,
}

DOCLAYNET_ID_TO_NAME = {
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


class DoclingPageScout:
    """Pipeline 1: Mô hình Docling cơ sở (Frozen) quét nhanh toàn bộ các trang tài liệu."""

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        existing_predictions_dir: Optional[Union[str, Path]] = None,
        lazy_init: bool = True,
    ):
        """Khởi tạo Docling Scout.

        Args:
            cache_dir: Thư mục lưu cache kết quả quét của Pipeline 1.
            existing_predictions_dir: Thư mục chứa JSON dự đoán sẵn của Docling (nếu đã chạy trước đó).
            lazy_init: Chỉ khởi tạo DocumentConverter khi cần chạy thực tế trên ảnh chưa có cache.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.existing_predictions_dir = Path(existing_predictions_dir) if existing_predictions_dir else None
        self._converter = None
        if not lazy_init:
            self._init_converter()

    def _init_converter(self):
        """Khởi tạo DocumentConverter của Docling ở chế độ tối giản (chỉ chạy Layout Model)."""
        if self._converter is not None:
            return
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, ImageFormatOption

        options = PdfPipelineOptions()
        options.do_ocr = False
        options.do_table_structure = False
        options.do_picture_classification = False
        options.do_picture_description = False
        options.do_chart_extraction = False
        options.do_code_enrichment = False
        options.do_formula_enrichment = False
        options.generate_page_images = False
        options.layout_options.keep_empty_clusters = True

        self._converter = DocumentConverter(
            allowed_formats=[InputFormat.IMAGE, InputFormat.PDF],
            format_options={
                InputFormat.IMAGE: ImageFormatOption(pipeline_options=options),
            },
        )

    @staticmethod
    def compute_spatial_features(
        boxes_xywh: List[List[float]],
        page_width: float,
        page_height: float,
    ) -> np.ndarray:
        """Chuyển đổi danh sách bbox [x, y, w, h] sang vector hình học 8 chiều chuẩn hóa [0, 1].

        Vector 8 chiều cho mỗi vùng j:
            [x_min, y_min, x_max, y_max, w_norm, h_norm, aspect_ratio, area_ratio]
        """
        if not boxes_xywh:
            return np.zeros((0, 8), dtype=np.float32)

        W = max(float(page_width), 1.0)
        H = max(float(page_height), 1.0)
        feats = []
        for bbox in boxes_xywh:
            x, y, w, h = [float(v) for v in bbox]
            w = max(1.0, w)
            h = max(1.0, h)
            x_min = np.clip(x / W, 0.0, 1.0)
            y_min = np.clip(y / H, 0.0, 1.0)
            x_max = np.clip((x + w) / W, 0.0, 1.0)
            y_max = np.clip((y + h) / H, 0.0, 1.0)
            w_norm = np.clip(w / W, 0.0, 1.0)
            h_norm = np.clip(h / H, 0.0, 1.0)
            aspect = min(5.0, w / max(h, 1e-4)) / 5.0
            area = np.clip((w * h) / (W * H), 0.0, 1.0)
            feats.append([x_min, y_min, x_max, y_max, w_norm, h_norm, aspect, area])
        return np.asarray(feats, dtype=np.float32)

    def extract_page(
        self,
        image_path: Union[str, Path],
        page_index: int = 0,
        total_pages: int = 1,
        fallback_to_model: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Phân tích một trang đơn lẻ bằng Pipeline 1 (hoặc nạp từ cache nếu đã tồn tại)."""
        image_path = Path(image_path)
        stem = image_path.stem

        # 1. Kiểm tra trong existing_predictions_dir hoặc cache_dir trước
        for candidate_dir in (self.existing_predictions_dir, self.cache_dir):
            if candidate_dir is not None:
                cached_file = candidate_dir / f"{stem}.json"
                if cached_file.is_file():
                    try:
                        raw = json.loads(cached_file.read_text(encoding="utf-8"))
                        return self._format_page_representation(raw, page_index, total_pages)
                    except Exception:
                        pass

        # 2. Nếu chưa có cache và cho phép chạy trực tiếp Docling Layout Model
        if not fallback_to_model:
            return None

        self._init_converter()
        page = self._converter.convert(str(image_path)).pages[0]
        width, height = float(page.size.width), float(page.size.height)

        detections = []
        for element in page.assembled.elements:
            cluster = element.cluster
            label_str = cluster.label.value
            cat_id = DOCLING_LABEL_TO_ID.get(label_str, 10)
            bbox = cluster.bbox.to_top_left_origin(height)
            detections.append(
                {
                    "category_id": cat_id,
                    "category_name": DOCLAYNET_ID_TO_NAME.get(cat_id, label_str),
                    "bbox": [
                        float(bbox.l),
                        float(bbox.t),
                        float(bbox.r - bbox.l),
                        float(bbox.b - bbox.t),
                    ],
                    "score": float(cluster.confidence),
                }
            )

        raw_result = {
            "image_id": int(stem) if stem.isdigit() else stem,
            "file_name": image_path.name,
            "width": width,
            "height": height,
            "detections": detections,
        }

        if self.cache_dir is not None:
            out_cache = self.cache_dir / f"{stem}.json"
            out_cache.write_text(json.dumps(raw_result, ensure_ascii=False, indent=2), encoding="utf-8")

        return self._format_page_representation(raw_result, page_index, total_pages)

    def _format_page_representation(
        self,
        raw_data: Dict[str, Any],
        page_index: int,
        total_pages: int,
    ) -> Dict[str, Any]:
        """Chuẩn hóa dữ liệu dự đoán của trang thành biểu diễn R_i cho Book-Level Encoder."""
        width = float(raw_data.get("width", 1025.0))
        height = float(raw_data.get("height", 1025.0))
        dets = raw_data.get("detections", [])

        boxes_xywh = [d["bbox"] for d in dets]
        labels = [max(0, min(10, int(d.get("category_id", 10)) - 1)) for d in dets]
        scores = [float(d.get("score", 1.0)) for d in dets]
        spatial_feats = self.compute_spatial_features(boxes_xywh, width, height)

        return {
            "page_id": raw_data.get("image_id", page_index),
            "file_name": raw_data.get("file_name", f"{page_index}.png"),
            "page_index": page_index,
            "total_pages": max(1, total_pages),
            "width": width,
            "height": height,
            "boxes_xywh": np.asarray(boxes_xywh, dtype=np.float32) if boxes_xywh else np.zeros((0, 4), dtype=np.float32),
            "spatial_features": spatial_feats,  # (M_i, 8)
            "labels": np.asarray(labels, dtype=np.int64),  # (M_i,) in 0..10
            "scores": np.asarray(scores, dtype=np.float32),  # (M_i,) in [0, 1]
            "raw_detections": dets,
        }

    def extract_book(self, page_image_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """Quét toàn bộ chuỗi trang P_1, ..., P_N của một cuốn sách."""
        total = len(page_image_paths)
        book_representations = []
        for idx, path in enumerate(page_image_paths):
            rep = self.extract_page(path, page_index=idx, total_pages=total)
            book_representations.append(rep)
        return book_representations
