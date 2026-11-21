"""OCR Grid (M2Doc-style) and Book-Anchor Heatmap / Memory builders for Method 12."""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from methods.method12.common import HEATMAP_SIZE, IMAGE_SIZE, NUM_DLN_CLASSES

_MATH_RE = re.compile(r"[=+\-*/^_{}\\∑∫√≤≥≠≈α-ωΑ-Ω]")
_LIST_PREFIX_RE = re.compile(r"^(\d+[\.\)]|[\-\*\•\◦\▪]|[a-zA-Z][\.\)])\s*")


def render_ocr_grid(
    ocr_boxes: List[List[float]],
    ocr_texts: List[str],
    src_size: Tuple[float, float] = (1025.0, 1025.0),
    grid: int = HEATMAP_SIZE,
) -> np.ndarray:
    """Render an 8-channel M2Doc-style OCR feature grid [8, grid, grid].

    Channels:
      0: Text cell presence / density
      1: Log-normalized character count
      2: Digit ratio
      3: Uppercase ratio
      4: Normalized cell height (proxy for font size)
      5: Normalized cell width
      6: List/bullet prefix indicator
      7: Math/symbol ratio
    """
    out = np.zeros((8, grid, grid), dtype=np.float32)
    sw, sh = max(float(src_size[0]), 1.0), max(float(src_size[1]), 1.0)

    for box, text in zip(ocr_boxes, ocr_texts):
        if len(box) < 4:
            continue
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        if x2 <= x1 or y2 <= y1:
            continue

        gx1 = max(0, min(grid - 1, int((x1 / sw) * grid)))
        gy1 = max(0, min(grid - 1, int((y1 / sh) * grid)))
        gx2 = max(gx1 + 1, min(grid, int(math.ceil((x2 / sw) * grid))))
        gy2 = max(gy1 + 1, min(grid, int(math.ceil((y2 / sh) * grid))))

        t = str(text or "").strip()
        n_chars = len(t)
        if n_chars == 0:
            continue

        n_digits = sum(ch.isdigit() for ch in t)
        n_upper = sum(ch.isupper() for ch in t)
        n_math = len(_MATH_RE.findall(t))
        is_list = 1.0 if _LIST_PREFIX_RE.match(t) else 0.0

        feats = np.array(
            [
                1.0,
                min(1.0, math.log1p(n_chars) / 5.0),
                n_digits / n_chars,
                n_upper / n_chars,
                min(1.0, ((y2 - y1) / sh) * 10.0),
                min(1.0, (x2 - x1) / sw),
                is_list,
                min(1.0, (n_math / n_chars) * 3.0),
            ],
            dtype=np.float32,
        )
        out[:, gy1:gy2, gx1:gx2] = np.maximum(out[:, gy1:gy2, gx1:gx2], feats[:, None, None])

    return out


def render_page_layout_heatmap(
    boxes_norm: List[List[float]],
    classes_idx: List[int],
    scores: Optional[List[float]] = None,
    grid: int = HEATMAP_SIZE,
) -> np.ndarray:
    """Render an 11-channel layout heatmap [11, grid, grid] from normalized [cx, cy, w, h] boxes."""
    hm = np.zeros((NUM_DLN_CLASSES, grid, grid), dtype=np.float32)
    for idx, (b, c) in enumerate(zip(boxes_norm, classes_idx)):
        if c < 0 or c >= NUM_DLN_CLASSES:
            continue
        cx, cy, bw, bh = b
        sc = float(scores[idx]) if scores is not None else 1.0
        gx1 = max(0, min(grid - 1, int((cx - bw / 2.0) * grid)))
        gy1 = max(0, min(grid - 1, int((cy - bh / 2.0) * grid)))
        gx2 = max(gx1 + 1, min(grid, int(math.ceil((cx + bw / 2.0) * grid))))
        gy2 = max(gy1 + 1, min(grid, int(math.ceil((cy + bh / 2.0) * grid))))
        hm[c, gy1:gy2, gx1:gx2] = np.maximum(hm[c, gy1:gy2, gx1:gx2], sc)
    return hm


def build_book_anchor_heatmap(
    anchor_pages_boxes: List[Tuple[List[List[float]], List[int], List[float]]],
    grid: int = HEATMAP_SIZE,
) -> np.ndarray:
    """Average 11-channel heatmaps across selected anchor pages of a book."""
    if not anchor_pages_boxes:
        return np.zeros((NUM_DLN_CLASSES, grid, grid), dtype=np.float32)
    hms = [render_page_layout_heatmap(b, c, s, grid=grid) for (b, c, s) in anchor_pages_boxes]
    return np.mean(np.stack(hms, axis=0), axis=0).astype(np.float32)


def pack_anchor_memory(
    anchor_pages_boxes: List[Tuple[List[List[float]], List[int], List[float]]],
    max_anchors: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pack anchor boxes from a book into [max_anchors, 16] and valid mask [max_anchors].

    Feature layout per anchor: [cx, cy, w, h, onehot_11 (11), score (1)] = 16 dims.
    """
    items: List[Tuple[float, List[float], int]] = []
    for boxes, classes, scores in anchor_pages_boxes:
        for b, c, s in zip(boxes, classes, scores):
            if 0 <= c < NUM_DLN_CLASSES:
                items.append((float(s), b, int(c)))

    # Prioritize higher-confidence anchors (and diverse classes if tied)
    items.sort(key=lambda x: -x[0])
    items = items[:max_anchors]

    mem = np.zeros((max_anchors, 4 + NUM_DLN_CLASSES + 1), dtype=np.float32)
    mask = np.zeros((max_anchors,), dtype=bool)
    for i, (s, b, c) in enumerate(items):
        mem[i, 0:4] = b
        mem[i, 4 + c] = 1.0
        mem[i, 4 + NUM_DLN_CLASSES] = s
        mask[i] = True
    return mem, mask
