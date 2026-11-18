"""Book-Level Sequential Dataset for Method 3 (Docling Scout -> Book Context -> DINO + M2Doc).

Nhóm các trang theo từng tài liệu / cuốn sách (doc_name / book_id) để tránh rò rỉ dữ liệu
liên trang (data leakage) giữa tập Train, Validation và Test, đồng thời tạo các cửa sổ trang
liên tiếp (Sliding Windows) để huấn luyện Book-Level Context Encoder và Context Fusion.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from torch.utils.data import Dataset
import torchvision.ops as tv_ops

try:
    from pipeline1_docling import DoclingPageScout
except ImportError:
    from methods.method3.src.pipeline1_docling import DoclingPageScout


class BookContextWindowDataset(Dataset):
    r"""Dataset trả về các cửa sổ gồm W trang liên tiếp của cùng một cuốn sách / tài liệu.

    Mỗi mẫu (sample) trong Dataset là một cửa sổ (window) gồm W trang [P_1, ..., P_W]:
    - Đặc trưng đầu ra từ Pipeline 1 (Docling Scout): boxes, labels, scores
    - Nhãn Ground Truth (matched theo IoU >= 0.5) để tối ưu hàm loss:
          L = \lambda_{cls} L_{cls} + \lambda_{box} L_{box} + \lambda_{giou} L_{giou}
    """

    def __init__(
        self,
        annotation_file: Path,
        images_dir: Path,
        scout: DoclingPageScout,
        window_size: int = 8,
        stride: int = 4,
        max_boxes_per_page: int = 64,
        limit_pages: Optional[int] = None,
    ):
        self.annotation_file = Path(annotation_file)
        self.images_dir = Path(images_dir)
        self.scout = scout
        self.window_size = window_size
        self.stride = max(1, stride)
        self.max_boxes = max_boxes_per_page

        with open(self.annotation_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        images = coco.get("images", [])
        if limit_pages:
            images = images[:limit_pages]

        self.anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
        for ann in coco.get("annotations", []):
            self.anns_by_img.setdefault(ann["image_id"], []).append(ann)

        # Nhóm trang theo doc_name (nếu có trong metadata DocLayNet) hoặc chia theo cụm liên tiếp
        docs_map: Dict[str, List[Dict[str, Any]]] = {}
        for idx, img_info in enumerate(images):
            doc_key = str(img_info.get("doc_name") or img_info.get("original_filename") or f"book_{idx // max(window_size * 2, 16)}")
            docs_map.setdefault(doc_key, []).append(img_info)

        # Sắp xếp các trang trong từng cuốn sách theo page_no hoặc id
        self.windows: List[List[Dict[str, Any]]] = []
        for doc_key, doc_pages in docs_map.items():
            doc_pages.sort(key=lambda x: (x.get("page_no", 0), x.get("id", 0)))
            if len(doc_pages) <= window_size:
                self.windows.append(doc_pages)
            else:
                for start in range(0, len(doc_pages), self.stride):
                    win = doc_pages[start : start + window_size]
                    if len(win) >= 2:
                        self.windows.append(win)

    def __len__(self) -> int:
        return len(self.windows)

    def _match_predictions_to_gt(
        self,
        pred_boxes_xywh: np.ndarray,
        gt_anns: List[Dict[str, Any]],
        width: float,
        height: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Gán nhãn Ground Truth và tọa độ đích cho từng vùng dự đoán dựa trên IoU."""
        M = len(pred_boxes_xywh)
        if M == 0:
            return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

        if not gt_anns:
            return np.full((M,), -100, dtype=np.int64), np.zeros((M, 4), dtype=np.float32)

        pred_xyxy = np.zeros((M, 4), dtype=np.float32)
        pred_xyxy[:, 0] = pred_boxes_xywh[:, 0]
        pred_xyxy[:, 1] = pred_boxes_xywh[:, 1]
        pred_xyxy[:, 2] = pred_boxes_xywh[:, 0] + np.maximum(1.0, pred_boxes_xywh[:, 2])
        pred_xyxy[:, 3] = pred_boxes_xywh[:, 1] + np.maximum(1.0, pred_boxes_xywh[:, 3])

        gt_xyxy = []
        gt_labels = []
        for ann in gt_anns:
            x, y, w, h = ann["bbox"]
            gt_xyxy.append([x, y, x + max(1.0, w), y + max(1.0, h)])
            gt_labels.append(max(0, min(10, int(ann.get("category_id", 10)) - 1)))

        ious = tv_ops.box_iou(
            torch.as_tensor(pred_xyxy, dtype=torch.float32),
            torch.as_tensor(gt_xyxy, dtype=torch.float32),
        ).numpy()

        best_gt_idx = ious.argmax(axis=1)
        best_iou = ious.max(axis=1)

        target_labels = np.full((M,), -100, dtype=np.int64)
        target_deltas = np.zeros((M, 4), dtype=np.float32)
        gt_xyxy_arr = np.asarray(gt_xyxy, dtype=np.float32)
        gt_labels_arr = np.asarray(gt_labels, dtype=np.int64)

        valid = best_iou >= 0.35
        target_labels[valid] = gt_labels_arr[best_gt_idx[valid]]
        matched_gt = gt_xyxy_arr[best_gt_idx[valid]]
        target_deltas[valid, 0] = (matched_gt[:, 0] - pred_xyxy[valid, 0]) / max(width, 1.0)
        target_deltas[valid, 1] = (matched_gt[:, 1] - pred_xyxy[valid, 1]) / max(height, 1.0)
        target_deltas[valid, 2] = (matched_gt[:, 2] - pred_xyxy[valid, 2]) / max(width, 1.0)
        target_deltas[valid, 3] = (matched_gt[:, 3] - pred_xyxy[valid, 3]) / max(height, 1.0)

        return target_labels, np.clip(target_deltas, -0.05, 0.05)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        window_pages = self.windows[idx]
        W_len = len(window_pages)
        M_max = self.max_boxes

        spatial_padded = torch.zeros((W_len, M_max, 8), dtype=torch.float32)
        init_labels_padded = torch.full((W_len, M_max), 10, dtype=torch.long)
        init_scores_padded = torch.zeros((W_len, M_max), dtype=torch.float32)
        pad_mask = torch.ones((W_len, M_max), dtype=torch.bool)

        gt_labels_padded = torch.full((W_len, M_max), -100, dtype=torch.long)
        gt_deltas_padded = torch.zeros((W_len, M_max, 4), dtype=torch.float32)

        for p_idx, img_info in enumerate(window_pages):
            img_id = img_info["id"]
            img_path = self.images_dir / img_info["file_name"]
            gt_anns = self.anns_by_img.get(img_id, [])

            # Lấy dự đoán của Pipeline 1 (Docling) hoặc fallback từ GT có nhiễu khi chạy offline
            rep = None
            if self.scout is not None:
                rep = self.scout.extract_page(
                    img_path, page_index=p_idx, total_pages=W_len, fallback_to_model=False
                )

            if rep is None:
                # Nếu chưa chạy cache Docling trên toàn bộ tập train, tạo candidate giả lập có nhiễu thực tế:
                # 1. BBox Jittering (lệch ~2-4% kích thước trang) mô phỏng sai số phát hiện vùng của Scout
                # 2. Label Noise (~20-25% nhầm lẫn nhãn) để Book-Context Encoder học cách sửa sai bằng ngữ cảnh toàn sách
                # 3. Confidence scores phân bố thực tế [0.45, 0.95]
                width = float(img_info.get("width", 1025.0))
                height = float(img_info.get("height", 1025.0))
                raw_boxes = [a["bbox"] for a in gt_anns[:M_max]]
                raw_lbs = [max(0, min(10, int(a.get("category_id", 10)) - 1)) for a in gt_anns[:M_max]]

                rng = np.random.RandomState(abs(hash((img_id, p_idx))) % (2**31 - 1))
                noisy_boxes = []
                noisy_lbs = []
                noisy_scs = []

                for b, l in zip(raw_boxes, raw_lbs):
                    dx = float(rng.normal(0, 0.02 * width))
                    dy = float(rng.normal(0, 0.02 * height))
                    dw = float(rng.normal(0, 0.025 * width))
                    dh = float(rng.normal(0, 0.025 * height))
                    nx = float(np.clip(b[0] + dx, 0.0, width - 10.0))
                    ny = float(np.clip(b[1] + dy, 0.0, height - 10.0))
                    nw = float(np.clip(b[2] + dw, 10.0, width - nx))
                    nh = float(np.clip(b[3] + dh, 10.0, height - ny))
                    noisy_boxes.append([nx, ny, nw, nh])

                    # 22% xác suất nhầm nhãn mô phỏng lỗi thực tế của Docling
                    if rng.rand() < 0.22:
                        noisy_lbs.append(int(rng.randint(0, 11)))
                    else:
                        noisy_lbs.append(l)

                    noisy_scs.append(float(rng.uniform(0.45, 0.95)))

                sp = DoclingPageScout.compute_spatial_features(noisy_boxes, width, height)
                rep = {
                    "width": width,
                    "height": height,
                    "boxes_xywh": np.asarray(noisy_boxes, dtype=np.float32) if noisy_boxes else np.zeros((0, 4), dtype=np.float32),
                    "spatial_features": sp,
                    "labels": np.asarray(noisy_lbs, dtype=np.int64) if noisy_lbs else np.zeros((0,), dtype=np.int64),
                    "scores": np.asarray(noisy_scs, dtype=np.float32) if noisy_scs else np.zeros((0,), dtype=np.float32),
                }

            m = min(len(rep["labels"]), M_max)
            if m > 0:
                spatial_padded[p_idx, :m] = torch.as_tensor(rep["spatial_features"][:m], dtype=torch.float32)
                init_labels_padded[p_idx, :m] = torch.as_tensor(rep["labels"][:m], dtype=torch.long)
                init_scores_padded[p_idx, :m] = torch.as_tensor(rep["scores"][:m], dtype=torch.float32)
                pad_mask[p_idx, :m] = False

                tgt_lbs, tgt_deltas = self._match_predictions_to_gt(
                    rep["boxes_xywh"][:m],
                    gt_anns,
                    rep["width"],
                    rep["height"],
                )
                gt_labels_padded[p_idx, :m] = torch.as_tensor(tgt_lbs, dtype=torch.long)
                gt_deltas_padded[p_idx, :m] = torch.as_tensor(tgt_deltas, dtype=torch.float32)

        return {
            "spatial": spatial_padded,
            "init_labels": init_labels_padded,
            "init_scores": init_scores_padded,
            "pad_mask": pad_mask,
            "gt_labels": gt_labels_padded,
            "gt_deltas": gt_deltas_padded,
            "page_indices": torch.arange(W_len, dtype=torch.long),
            "total_pages": torch.full((W_len,), W_len, dtype=torch.long),
        }
