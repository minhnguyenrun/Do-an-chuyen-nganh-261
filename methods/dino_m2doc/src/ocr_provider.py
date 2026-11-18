"""OCR text provider for DINO + M2Doc multi-modal layout analysis.

Supports:
1. Fast lookup from DocLayNet raw parquet annotations (pdf_cells)
2. RapidOCR / PaddleOCR dynamic on-the-fly text extraction for arbitrary images
3. Fallback dummy text representations when OCR is unavailable.
"""

import pickle
import sys
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import pyarrow.dataset  # noqa: F401 — must be imported before cv2/torch on Windows
    import pyarrow.parquet as pq
except ImportError:
    pq = None

import cv2
import numpy as np


class OCRProvider:
    """Provides OCR text bounding boxes and text content for document images."""

    DATASET_TO_SPLIT = {
        "doclaynet": "train",
        "doclaynet_val": "validation",
        "doclaynet_test": "test",
    }

    def __init__(self, raw_data_dir: Optional[str] = None, dataset: str = "doclaynet_test"):
        self.raw_data_dir = (
            Path(raw_data_dir)
            if raw_data_dir
            else Path(__file__).resolve().parents[3] / "datasets" / "doclaynet" / "raw" / "data"
        )
        self.dataset = dataset.lower()
        self.split_prefix = self.DATASET_TO_SPLIT.get(self.dataset)
        self._parquet_cache = {}
        self._index_built = False
        self._rapid_ocr = None
        self._rapid_ocr_failed = False

    def _ensure_split_parquet_shards(self):
        """Ensures raw parquet shards for the target DocLayNet split are present so pdf_cells can be loaded."""
        if not self.split_prefix:
            return
        existing = list(self.raw_data_dir.glob(f"{self.split_prefix}-*.parquet"))
        if self.split_prefix == "test" and len(existing) < 6:
            try:
                from huggingface_hub import hf_hub_download

                raw_root = self.raw_data_dir.parent
                raw_root.mkdir(parents=True, exist_ok=True)
                for i in range(6):
                    rfilename = f"data/test-{i:05d}-of-00006.parquet"
                    if not (raw_root / rfilename).exists():
                        print(f"[OCRProvider] Downloading {rfilename} for ground-truth pdf_cells...", flush=True)
                        hf_hub_download(
                            repo_id="docling-project/DocLayNet-v1.2",
                            repo_type="dataset",
                            filename=rfilename,
                            local_dir=str(raw_root),
                        )
            except Exception as e:
                print(f"[OCRProvider] Notice: could not auto-download test parquet shards: {e}", flush=True)

    def _build_parquet_index(self):
        """Indexes available parquet files matching the current dataset split to map image_id to pdf_cells."""
        if self._index_built:
            return

        cache_tag = self.split_prefix or "all"
        cache_file = self.raw_data_dir / f".pdf_cells_cache_{cache_tag}.pkl"

        # 1. Neu file cache pickle da ton tai va chua du lieu -> dung ngay lap tuc
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                if cached_data.get("cache") and len(cached_data["cache"]) > 0:
                    self._parquet_cache = cached_data["cache"]
                    self._index_built = True
                    return
            except Exception:
                pass

        if not self.raw_data_dir.exists():
            self._index_built = True
            return

        self._ensure_split_parquet_shards()

        if self.split_prefix:
            parquet_files = sorted(self.raw_data_dir.glob(f"{self.split_prefix}-*.parquet"))
        else:
            parquet_files = sorted(self.raw_data_dir.glob("*.parquet"))

        try:
            if pq is None:
                self._index_built = True
                return

            for pf in parquet_files:
                table = pq.read_table(str(pf), columns=["id", "pdf_cells"])
                df = table.to_pandas()
                for _, row in df.iterrows():
                    img_id = row["id"]
                    cells = row["pdf_cells"]
                    if cells is not None:
                        boxes = []
                        texts = []
                        if isinstance(cells, dict):
                            raw_boxes = cells.get("bbox", [])
                            raw_texts = cells.get("text", [])
                            for b, t in zip(raw_boxes, raw_texts):
                                if b and len(b) == 4 and t:
                                    boxes.append(b)
                                    texts.append(str(t))
                        elif isinstance(cells, (list, np.ndarray)):
                            for cell in cells:
                                if isinstance(cell, dict):
                                    b = cell.get("bbox")
                                    t = cell.get("text")
                                    if b and len(b) == 4 and t:
                                        boxes.append(b)
                                        texts.append(str(t))
                        if boxes:
                            self._parquet_cache[img_id] = (boxes, texts)

            if parquet_files:
                try:
                    with open(cache_file, "wb") as f:
                        pickle.dump({"num_files": len(parquet_files), "cache": self._parquet_cache}, f)
                except Exception:
                    pass

        except Exception as e:
            print(f"[OCRProvider] Error indexing parquet files: {e}", flush=True)

        self._index_built = True

    def _get_rapid_ocr(self):
        """Initializes RapidOCR for dynamic on-the-fly text extraction."""
        if self._rapid_ocr is not None:
            return self._rapid_ocr
        if self._rapid_ocr_failed:
            return None

        try:
            from rapidocr_onnxruntime import RapidOCR

            self._rapid_ocr = RapidOCR()
            return self._rapid_ocr
        except Exception as e:
            print(f"[OCRProvider] RapidOCR unavailable ({e}), using fallback.", flush=True)
            self._rapid_ocr_failed = True
            return None

    def get_ocr_for_image(
        self,
        image_path: str,
        image_id: Optional[int] = None,
        target_size: Tuple[int, int] = (1025, 1025),
        orig_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[List[List[float]], List[str]]:
        """Retrieves OCR text boxes and tokens for a document image."""
        self._build_parquet_index()

        if image_id is not None and image_id in self._parquet_cache:
            raw_boxes, raw_texts = self._parquet_cache[image_id]
            if orig_size:
                orig_w, orig_h = orig_size
            else:
                try:
                    data = np.fromfile(image_path, dtype=np.uint8)
                    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    orig_h, orig_w = img.shape[:2]
                except Exception:
                    orig_w, orig_h = target_size

            scale_w = target_size[0] / orig_w
            scale_h = target_size[1] / orig_h

            scaled_boxes = []
            for b in raw_boxes:
                # b: [x1, y1, x2, y2]
                scaled_boxes.append(
                    [
                        float(b[0]) * scale_w,
                        float(b[1]) * scale_h,
                        float(b[2]) * scale_w,
                        float(b[3]) * scale_h,
                    ]
                )
            return scaled_boxes, raw_texts

        rapid_ocr = self._get_rapid_ocr()
        if rapid_ocr is not None:
            try:
                data = np.fromfile(image_path, dtype=np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if img is not None:
                    h, w = img.shape[:2]
                    results, _ = rapid_ocr(img)
                    if results:
                        boxes = []
                        texts = []
                        scale_w = target_size[0] / w
                        scale_h = target_size[1] / h
                        for item in results:
                            poly = item[0]
                            text = item[1]
                            xs = [pt[0] for pt in poly]
                            ys = [pt[1] for pt in poly]
                            boxes.append(
                                [
                                    float(min(xs)) * scale_w,
                                    float(min(ys)) * scale_h,
                                    float(max(xs)) * scale_w,
                                    float(max(ys)) * scale_h,
                                ]
                            )
                            texts.append(str(text))
                        return boxes, texts
            except Exception as e:
                pass

        # Fallback dummy grid
        t_w, t_h = target_size
        dummy_boxes = [
            [0.0, 0.0, float(t_w), 40.0],
            [0.0, 50.0, float(t_w), float(t_h) - 50.0],
            [0.0, float(t_h) - 40.0, float(t_w), float(t_h)],
        ]
        dummy_texts = ["Header text", "Document body paragraph content", "Footer text"]
        return dummy_boxes, dummy_texts
