"""Parser utility to extract and standardize layout bounding boxes from dots.ocr VLM outputs."""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

DOCLAYNET_CATEGORIES = {
    "Caption": 1,
    "Footnote": 2,
    "Formula": 3,
    "List-item": 4,
    "Page-footer": 5,
    "Page-header": 6,
    "Picture": 7,
    "Section-header": 8,
    "Table": 9,
    "Text": 10,
    "Title": 11,
}

PUBLAYNET_CATEGORIES = {
    "Text": 1,
    "Title": 2,
    "List": 3,
    "Table": 4,
    "Figure": 5,
}

DOCLAYNET_TO_PUBLAYNET = {
    "Text": (1, "Text"),
    "Footnote": (1, "Text"),
    "Page-header": (1, "Text"),
    "Page-footer": (1, "Text"),
    "Formula": (1, "Text"),
    "Title": (2, "Title"),
    "Section-header": (2, "Title"),
    "List-item": (3, "List"),
    "Table": (4, "Table"),
    "Picture": (5, "Figure"),
    "Caption": (5, "Figure"),
}

NORM_MAP = {
    "caption": "Caption",
    "footnote": "Footnote",
    "formula": "Formula",
    "equation": "Formula",
    "list-item": "List-item",
    "list_item": "List-item",
    "list": "List-item",
    "page-footer": "Page-footer",
    "page_footer": "Page-footer",
    "footer": "Page-footer",
    "page-header": "Page-header",
    "page_header": "Page-header",
    "header": "Page-header",
    "picture": "Picture",
    "image": "Picture",
    "figure": "Picture",
    "diagram": "Picture",
    "section-header": "Section-header",
    "section_header": "Section-header",
    "paragraph_title": "Section-header",
    "subtitle": "Section-header",
    "table": "Table",
    "text": "Text",
    "paragraph": "Text",
    "title": "Title",
    "doc_title": "Title",
}


def clean_json_string(text: str) -> str:
    """Strips markdown code blocks, comments, and extra text to isolate pure JSON."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        text = match.group(1).strip()

    # Try finding outer [ ... ]
    start_bracket = text.find("[")
    end_bracket = text.rfind("]")
    if start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
        return text[start_bracket : end_bracket + 1]

    # Try finding outer { ... }
    start_brace = text.find("{")
    end_brace = text.rfind("}")
    if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
        return text[start_brace : end_brace + 1]

    return text


def _smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 3136,
    max_pixels: int = 11289600,
) -> Tuple[int, int]:
    """Computes the Qwen2-VL smart_resize dimensions (resized_height, resized_width)."""
    import math

    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return int(h_bar), int(w_bar)


def parse_dots_layout_response(
    raw_response: str,
    img_width: int,
    img_height: int,
    default_score: float = 0.95,
    dataset: str = "doclaynet",
    input_width: Optional[int] = None,
    input_height: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Parses raw text output from dots.ocr model into standard detection schema."""
    cleaned = clean_json_string(raw_response)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: regex search for bbox and category objects
        items = re.findall(
            r'\{[^{}]*?"bbox"\s*:\s*\[([0-9.,\s]+)\][^{}]*?"(?:category|label)"\s*:\s*"([^"]+)"[^{}]*?\}',
            cleaned,
            re.IGNORECASE,
        )
        if not items:
            items = re.findall(
                r'\{[^{}]*?"(?:category|label)"\s*:\s*"([^"]+)"[^{}]*?"bbox"\s*:\s*\[([0-9.,\s]+)\][^{}]*?\}',
                cleaned,
                re.IGNORECASE,
            )
            items = [(coords, cat) for cat, coords in items]

        if not items:
            return []

        data = []
        for coords_str, cat_name in items:
            try:
                coords = [float(v.strip()) for v in coords_str.split(",") if v.strip()]
                data.append({"bbox": coords, "category": cat_name})
            except Exception:
                continue

    if isinstance(data, dict):
        if "layout" in data and isinstance(data["layout"], list):
            data = data["layout"]
        elif "elements" in data and isinstance(data["elements"], list):
            data = data["elements"]
        else:
            data = [data]

    if not isinstance(data, list):
        return []

    if input_width is None or input_height is None or input_width <= 0 or input_height <= 0:
        input_height, input_width = _smart_resize(img_height, img_width)

    is_publaynet = dataset.lower() == "publaynet"

    detections = []
    for item in data:
        if not isinstance(item, dict):
            continue

        raw_cat = item.get("category") or item.get("label") or item.get("type") or ""
        norm_cat = NORM_MAP.get(str(raw_cat).strip().lower(), str(raw_cat).strip())
        if norm_cat not in DOCLAYNET_CATEGORIES:
            continue

        if is_publaynet:
            cat_info = DOCLAYNET_TO_PUBLAYNET.get(norm_cat)
            if not cat_info:
                continue
            cat_id, cat_name = cat_info
        else:
            cat_id = DOCLAYNET_CATEGORIES[norm_cat]
            cat_name = norm_cat

        raw_bbox = item.get("bbox") or item.get("box")
        if not raw_bbox or len(raw_bbox) != 4:
            continue

        try:
            x1, y1, x2, y2 = [float(c) for c in raw_bbox]
        except (ValueError, TypeError):
            continue

        # If coords are normalized 0..1
        if max(x1, x2) <= 1.05 and max(y1, y2) <= 1.05 and img_width > 1:
            x1 = x1 * img_width
            x2 = x2 * img_width
            y1 = y1 * img_height
            y2 = y2 * img_height
        else:
            # dots.ocr outputs coordinates in the resized input image space (input_width x input_height)
            scale_x = float(img_width) / float(input_width)
            scale_y = float(img_height) / float(input_height)
            x1 = x1 * scale_x
            x2 = x2 * scale_x
            y1 = y1 * scale_y
            y2 = y2 * scale_y

        # Clip to image boundary
        x1 = max(0.0, min(float(x1), float(img_width)))
        y1 = max(0.0, min(float(y1), float(img_height)))
        x2 = max(0.0, min(float(x2), float(img_width)))
        y2 = max(0.0, min(float(y2), float(img_height)))

        bx = min(x1, x2)
        by = min(y1, y2)
        bw = abs(x2 - x1)
        bh = abs(y2 - y1)

        if bw < 2 or bh < 2:
            continue

        score = float(item.get("score", default_score))
        detections.append(
            {
                "category_id": cat_id,
                "category_name": cat_name,
                "bbox": [round(bx, 2), round(by, 2), round(bw, 2), round(bh, 2)],
                "score": round(score, 4),
            }
        )

    return detections

