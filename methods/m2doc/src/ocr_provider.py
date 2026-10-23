"""OCR text provider for M2Doc multi-modal layout analysis.

Supports:
1. Fast lookup from DocLayNet raw parquet annotations (pdf_cells)
2. RapidOCR / PaddleOCR dynamic on-the-fly text extraction for arbitrary images
3. Fallback dummy text representations when OCR is unavailable.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import pickle

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

    def __init__(self, raw_data_dir: Optional[str] = None, dataset: str = "doclaynet"):
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

        self._ensure_split_parquet_shards()

        if not self.raw_data_dir.exists():
            self._index_built = True
            return

        if self.split_prefix:
            parquet_files = sorted(self.raw_data_dir.glob(f"{self.split_prefix}-*.parquet"))
        else:
            parquet_files = sorted(self.raw_data_dir.glob("*.parquet"))

        if cache_file.exists() and parquet_files:
            try:
                with open(cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                if cached_data.get("num_files") == len(parquet_files):
                    self._parquet_cache = cached_data["cache"]
                    self._index_built = True
                    return
            except Exception:
                pass

        try:
            if pq is None:
                import pyarrow.parquet as _pq
            else:
                _pq = pq

            for pq_file in parquet_files:
                table = _pq.read_table(pq_file, columns=["metadata", "pdf_cells"])
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
                                    x1 = max(0.0, float(b[0]))
                                    y1 = max(0.0, float(b[1]))
                                    x2 = max(x1, float(b[0] + b[2]))
                                    y2 = max(y1, float(b[1] + b[3]))
                                    if x2 > x1 and y2 > y1:
                                        flat_boxes.append([x1, y1, x2, y2])
                                        flat_texts.append(t)

                    entry = (flat_boxes, flat_texts)
                    if img_id is not None:
                        self._parquet_cache[img_id] = entry
                    if orig_fname:
                        self._parquet_cache[orig_fname] = entry

            if self._parquet_cache and parquet_files:
                try:
                    with open(cache_file, "wb") as f:
                        pickle.dump({"num_files": len(parquet_files), "cache": self._parquet_cache}, f, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception:
                    pass

            self._index_built = True
        except Exception as e:
            print(f"[OCRProvider] Notice: could not index parquet files: {e}", flush=True)
            self._index_built = True

    def _init_rapid_ocr(self):
        """Initializes RapidOCR with automatic fallback to PyTorch engine when onnxruntime is unavailable."""
        from rapidocr import RapidOCR

        try:
            import onnxruntime  # noqa: F401

            return RapidOCR()
        except ImportError:
            from rapidocr import EngineType
            import torch

            use_cuda = torch.cuda.is_available()
            return RapidOCR(
                params={
                    "Det.engine_type": EngineType.TORCH,
                    "Cls.engine_type": EngineType.TORCH,
                    "Rec.engine_type": EngineType.TORCH,
                    "EngineConfig.torch.use_cuda": use_cuda,
                }
            )

    def get_ocr_for_image(
        self,
        image_path: str,
        image_id: Optional[int] = None,
        target_size: Tuple[int, int] = (1025, 1025),
        orig_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[List[List[float]], List[str]]:
        """Retrieves scaled OCR text bounding boxes [x1, y1, x2, y2] and text strings.

        Args:
            image_path: Path to document image
            image_id: Optional integer ID matching DocLayNet
            target_size: (target_width, target_height) for model input scaling
            orig_size: Optional (orig_width, orig_height) to avoid redundant image read

        Returns:
            Tuple of (scaled_bboxes, texts)
        """
        self._build_parquet_index()
        target_w, target_h = target_size

        # 1. Try parquet lookup first
        filename = Path(image_path).name
        raw_entry = None
        if image_id is not None and image_id in self._parquet_cache:
            raw_entry = self._parquet_cache[image_id]
        elif filename in self._parquet_cache:
            raw_entry = self._parquet_cache[filename]

        if orig_size is not None:
            orig_w, orig_h = orig_size
            img = None
        else:
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

        if raw_entry is not None:
            boxes, texts = raw_entry
            if len(boxes) > 0:
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
            return [], []

        # 2. Try RapidOCR fallback for images not in parquet cache
        if self._rapid_ocr_failed:
            return [], []

        if img is None:
            try:
                data = np.fromfile(str(image_path), dtype=np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            except Exception:
                img = None
            if img is None:
                return [], []

        try:
            if self._rapid_ocr is None:
                self._rapid_ocr = self._init_rapid_ocr()

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
            print(f"[OCRProvider] RapidOCR warning: {e}", flush=True)
            self._rapid_ocr_failed = True

        return [], []

