"""Deterministic request geometry and generation budgets (opt in only)."""
import math
from statistics import median

from .core import context_box
from .types import Config, Line


def _layout_size(width, height, edge):
    """Fit the crop aspect to an area budget, then round down to the 28px grid."""
    scale = edge / math.sqrt(width * height)
    columns = max(1, math.floor(width * scale / 28))
    rows = max(1, math.floor(height * scale / 28))
    # Minimum one grid cell on the thin axis can otherwise exceed the budget.
    cells = (edge // 28) ** 2
    columns = min(columns, cells // rows) if rows <= cells else 1
    rows = min(rows, cells // columns)
    return columns * 28, rows * 28


def build_inference_request(target: dict, lines: list[Line], width: int,
                            height: int, cfg: Config) -> dict:
    """Return page XYXY crop, helper image size, cap and a traceable policy name.

    Invalid geometry raises ValueError before slicing a crop. Legacy geometry is
    delegated unchanged to context_box; callers retain the old resize path.
    """
    if (not isinstance(width, int) or not isinstance(height, int)
            or width <= 0 or height <= 0):
        raise ValueError('inference page dimensions must be positive integers')
    bbox = target.get('bbox')
    if (not isinstance(bbox, (list, tuple)) or len(bbox) != 4
            or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in bbox)
            or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
        raise ValueError('inference target bbox must be finite and nonempty')
    if not cfg.enable_adaptive_inference:
        crop = context_box(target, lines, width, height)
        size, cap, reason = (cfg.crop_edge, cfg.crop_edge), cfg.max_tokens, 'legacy_context_square'
    else:
        selected_ids = set(target.get('line_ids', []))
        selected = [line for line in lines if line.id in selected_ids]
        heights = []
        for line in selected:
            points = line.polygon
            if (not isinstance(points, (list, tuple)) or len(points) < 3
                    or any(len(p) != 2 or not all(isinstance(v, (int, float))
                           and math.isfinite(v) for v in p) for p in points)):
                raise ValueError('inference target line polygon must be finite')
            line_height = max(p[1] for p in points) - min(p[1] for p in points)
            if line_height <= 0:
                raise ValueError('inference target line polygon must have positive height')
            heights.append(line_height)
        a, b, c, d = bbox
        line_height = median(heights) if heights else 10
        dx = max(1, min((c-a)*.10, 1.5*line_height))
        dy = max(1, min((d-b)*.10, 1.5*line_height))
        crop = [max(0, math.floor(a-dx)), max(0, math.floor(b-dy)),
                min(width, math.ceil(c+dx)), min(height, math.ceil(d+dy))]
        edge = 448 if len(selected_ids) <= 3 and len(target.get('fallback_baseline', [])) <= 1 else 560
        size = _layout_size(crop[2]-crop[0], crop[3]-crop[1], edge) if crop[2] > crop[0] and crop[3] > crop[1] else None
        sources = max(1, len(set(target.get('source_ids', []))))
        cap = min(768, max(96, 32+28*max(len(selected_ids), sources)))
        reason = 'adaptive_tight_crop_grid28_dynamic_tokens'
    if crop[2] <= crop[0] or crop[3] <= crop[1]:
        raise ValueError('inference crop is empty after page clipping')
    return {'crop_bbox': crop, 'layout_size': size, 'max_new_tokens': cap, 'reason': reason}
