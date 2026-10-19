"""OCR text provider for M2Doc multi-modal layout analysis.

Supports:
1. Fast lookup from DocLayNet raw parquet annotations (pdf_cells)
2. RapidOCR / PaddleOCR dynamic on-the-fly text extraction for arbitrary images
3. Fallback dummy text representations when OCR is unavailable.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np


class OCRProvider:
    """Provides OCR text bounding boxes and text content for document images."""

    def __init__(self, raw_data_dir: Optional[str] = None):
        self.raw_data_dir = (
            Path(raw_data_dir)
            if raw_data_dir
            else Path(__file__).resolve().parents[3] / "datasets" / "doclaynet" / "raw" / "data"
        )
        self._parquet_cache = {}
        self._index_built = False
        self._rapid_ocr = None

    def _build_parquet_index(self):
        """Indexes available parquet files to map image_id and filename to pdf_cells."""
        if self._index_built:
            return

        if not self.raw_data_dir.exists():
            self._index_built = True
            return

        try:
            import pyarrow.parquet as pq

            parquet_files = sorted(self.raw_data_dir.glob("*.parquet"))
            for pq_file in parquet_files:
                table = pq.read_table(pq_file, columns=["metadata", "pdf_cells"])
                meta_list = table["metadata"].to_pylist()
                cells_list = table["pdf_cells"]

                for idx, meta in enumerate(meta_list):
                    img_id = meta.get("image_id")
                    orig_fname = meta.get("original_filename", "")
                    cells = cells_list[idx].as_py()

                    flat_boxes = []
                    flat_texts = []
                    if cells:
                        for grp in cells:
                            for cell in grp:
                                b = cell.get("bbox")
                                t = cell.get("text", "").strip()
                                if b and t:
                                    # bbox is [x, y, w, h] -> convert to [x1, y1, x2, y2]
                                    flat_boxes.append([b[0], b[1], b[0] + b[2], b[1] + b[3]])
                                    flat_texts.append(t)

                    entry = (flat_boxes, flat_texts)
                    if img_id is not None:
                        self._parquet_cache[img_id] = entry
                    if orig_fname:
                        self._parquet_cache[orig_fname] = entry

            self._index_built = True
        except Exception as e:
            print(f"[OCRProvider] Notice: could not index parquet files: {e}")
            self._index_built = True

    def get_ocr_for_image(
        self,
        image_path: str,
        image_id: Optional[int] = None,
        target_size: Tuple[int, int] = (1025, 1025),
    ) -> Tuple[List[List[float]], List[str]]:
        """Retrieves scaled OCR text bounding boxes [x1, y1, x2, y2] and text strings.

        Args:
            image_path: Path to document image
            image_id: Optional integer ID matching DocLayNet
            target_size: (target_width, target_height) for model input scaling

        Returns:
            Tuple of (scaled_bboxes, texts)
        """
        self._build_parquet_index()
        target_w, target_h = target_size

        try:
            data = np.fromfile(str(image_path), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            img = None
        if img is None:
            return [], []
        orig_h, orig_w = img.shape[:2]
        scale_x = target_w / max(orig_w, 1)
        scale_y = target_h / max(orig_h, 1)

        # 1. Try parquet lookup
        filename = Path(image_path).name
        raw_entry = None
        if image_id is not None and image_id in self._parquet_cache:
            raw_entry = self._parquet_cache[image_id]
        elif filename in self._parquet_cache:
            raw_entry = self._parquet_cache[filename]

        if raw_entry and len(raw_entry[0]) > 0:
            boxes, texts = raw_entry
            scaled_boxes = [
                [
                    round(b[0] * scale_x, 2),
                    round(b[1] * scale_y, 2),
                    round(b[2] * scale_x, 2),
                    round(b[3] * scale_y, 2),
                ]
                for b in boxes
            ]
            return scaled_boxes, texts

        # 2. Try RapidOCR fallback
        try:
            if self._rapid_ocr is None:
                from rapidocr import RapidOCR

                self._rapid_ocr = RapidOCR()

            ocr_res = self._rapid_ocr(img)
            boxes = None
            texts = None

            # Handle new RapidOCROutput object
            if hasattr(ocr_res, "boxes") and hasattr(ocr_res, "txts"):
                boxes = ocr_res.boxes
                texts = ocr_res.txts
            elif isinstance(ocr_res, tuple) and len(ocr_res) > 0:
                raw_list = ocr_res[0]
                if raw_list:
                    boxes = [item[0] for item in raw_list]
                    texts = [item[1] for item in raw_list]
            elif isinstance(ocr_res, list):
                boxes = [item[0] for item in ocr_res]
                texts = [item[1] for item in ocr_res]

            if boxes is not None and texts is not None and len(boxes) > 0:
                scaled_boxes = []
                valid_texts = []
                for pts, text in zip(boxes, texts):
                    if not text:
                        continue
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    scaled_boxes.append(
                        [
                            round(float(x1) * scale_x, 2),
                            round(float(y1) * scale_y, 2),
                            round(float(x2) * scale_x, 2),
                            round(float(y2) * scale_y, 2),
                        ]
                    )
                    valid_texts.append(str(text))
                return scaled_boxes, valid_texts
        except Exception as e:
            print(f"[OCRProvider] RapidOCR warning: {e}")

        return [], []
