"""Parser utility to extract and standardize layout bounding boxes from Chandra OCR outputs.

Supports structured HTML tags with data-bbox & data-label attributes,
structured JSON blocks, and converts coordinates to DocLayNet/PubLayNet standard [x, y, w, h] format.
"""

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

# Mapping Chandra block types to DocLayNet 11 categories
CHANDRA_NORM_MAP = {
    # Text
    "text": "Text",
    "paragraph": "Text",
    "body": "Text",
    "plain_text": "Text",
    "plain-text": "Text",
    "code-block": "Text",
    "code_block": "Text",
    "bibliography": "Text",
    "complex-block": "Text",
    "complex_block": "Text",
    "blank-page": "Text",
    "blank_page": "Text",
    # Title & Headers
    "title": "Title",
    "document-title": "Title",
    "doc_title": "Title",
    "section-header": "Section-header",
    "section_header": "Section-header",
    "sub-header": "Section-header",
    "sub_header": "Section-header",
    "heading": "Section-header",
    "table-of-contents": "Section-header",
    "toc": "Section-header",
    # Running header & footer
    "page-header": "Page-header",
    "page_header": "Page-header",
    "header": "Page-header",
    "page-footer": "Page-footer",
    "page_footer": "Page-footer",
    "footer": "Page-footer",
    # Captions & Footnotes
    "caption": "Caption",
    "footnote": "Footnote",
    # Tables
    "table": "Table",
    "tabular": "Table",
    "form": "Table",
    # Images & Diagrams
    "image": "Picture",
    "figure": "Picture",
    "diagram": "Picture",
    "chart": "Picture",
    "picture": "Picture",
    "illustration": "Picture",
    "chemical-block": "Picture",
    "chemical_block": "Picture",
    # Math & Formulas
    "equation-block": "Formula",
    "equation_block": "Formula",
    "equation": "Formula",
    "formula": "Formula",
    "math": "Formula",
    # Lists
    "list-group": "List-item",
    "list_group": "List-item",
    "list-item": "List-item",
    "list_item": "List-item",
    "list": "List-item",
}


def parse_bbox_values(val: Any) -> Optional[List[float]]:
    """Converts diverse bbox representations (list, tuple, string) into [x1, y1, x2, y2]."""
    if isinstance(val, (list, tuple)) and len(val) >= 4:
        try:
            return [float(v) for v in val[:4]]
        except (ValueError, TypeError):
            return None
    elif isinstance(val, str):
        nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", val)
        if len(nums) >= 4:
            try:
                return [float(nums[0]), float(nums[1]), float(nums[2]), float(nums[3])]
            except (ValueError, TypeError):
                return None
    return None


def parse_html_bboxes(raw_text: str) -> List[Dict[str, Any]]:
    """Extracts bounding boxes and types from Chandra HTML markup with data-bbox and data-label attributes."""
    results = []

    # 1. Attempt parsing with BeautifulSoup if available
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(raw_text, "html.parser")
        for el in soup.find_all(lambda tag: tag.has_attr("data-bbox") or tag.has_attr("bbox")):
            bbox_str = el.get("data-bbox") or el.get("bbox")
            cls = el.get("class")
            cls_str = cls[0] if isinstance(cls, list) and cls else cls
            label = el.get("data-label") or el.get("label") or cls_str or el.get("type") or el.name
            coords = parse_bbox_values(bbox_str)
            if coords:
                results.append({"type": str(label), "bbox": coords})
    except Exception:
        results = []

    if results:
        return results

    # 2. Fallback: Regex parsing matching any HTML tag with data-bbox or bbox attribute
    tag_pattern = re.compile(r"<([a-zA-Z0-9_\-]+)([^>]*?)>", re.DOTALL)
    bbox_pattern = re.compile(r'(?:data-bbox|bbox)=["\']([^"\']+)["\']', re.IGNORECASE)
    label_pattern = re.compile(r'(?:data-label|label|type|class)=["\']([^"\']+)["\']', re.IGNORECASE)

    for match in tag_pattern.finditer(raw_text):
        tag_name, attrs_str = match.groups()
        b_match = bbox_pattern.search(attrs_str)
        if b_match:
            bbox_str = b_match.group(1)
            l_match = label_pattern.search(attrs_str)
            label = l_match.group(1) if l_match else tag_name
            coords = parse_bbox_values(bbox_str)
            if coords:
                results.append({"type": str(label), "bbox": coords})

    # 3. Match self-closing elements <box type="..." bbox="..."/>
    box_matches = re.findall(
        r'<box\s+[^>]*?(?:type|class)=["\']([^"\']+)["\'][^>]*?bbox=["\']([^"\']+)["\'][^>]*?/?>',
        raw_text,
    )
    for b_type, bbox_str in box_matches:
        coords = parse_bbox_values(bbox_str)
        if coords:
            results.append({"type": b_type, "bbox": coords})

    return results


def clean_json_markdown(text: str) -> str:
    """Strips markdown code blocks to expose pure JSON content."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    return text


def parse_chandra_layout_response(
    raw_response: str,
    img_width: int,
    img_height: int,
    default_score: float = 0.95,
    dataset: str = "doclaynet",
) -> List[Dict[str, Any]]:
    """Parses raw text output from Chandra model into standard detection schema."""
    if not raw_response or not raw_response.strip():
        return []

    text = clean_json_markdown(raw_response)
    raw_items = []

    # 1. Attempt JSON parsing
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ["layout", "blocks", "elements", "predictions", "regions"]:
                if key in data and isinstance(data[key], list):
                    raw_items = data[key]
                    break
            if not raw_items and "bbox" in data:
                raw_items = [data]
        elif isinstance(data, list):
            raw_items = data
    except json.JSONDecodeError:
        pass

    # 2. If JSON parsing failed, try extracting HTML style bboxes with data-bbox and data-label
    if not raw_items:
        raw_items = parse_html_bboxes(raw_response)

    # 3. If still empty, regex search for JSON-like bbox and type patterns
    if not raw_items:
        regex_matches = re.findall(
            r'\{[^{}]*?"(?:bbox|box)"\s*:\s*\[([0-9.,\s]+)\][^{}]*?"(?:type|category|label)"\s*:\s*"([^"]+)"[^{}]*?\}',
            text,
            re.IGNORECASE,
        )
        if not regex_matches:
            regex_matches = re.findall(
                r'\{[^{}]*?"(?:type|category|label)"\s*:\s*"([^"]+)"[^{}]*?"(?:bbox|box)"\s*:\s*\[([0-9.,\s]+)\][^{}]*?\}',
                text,
                re.IGNORECASE,
            )
            regex_matches = [(b, t) for t, b in regex_matches]

        for bbox_str, type_str in regex_matches:
            coords = parse_bbox_values(bbox_str)
            if coords:
                raw_items.append({"bbox": coords, "type": type_str})

    is_publaynet = dataset.lower() == "publaynet"

    # Standardize and filter detections
    detections = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue

        raw_type = item.get("type") or item.get("category") or item.get("label") or "Text"
        type_key = str(raw_type).strip().lower().replace("_", "-").replace(" ", "-")
        norm_cat = CHANDRA_NORM_MAP.get(type_key, "Text")

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

        raw_bbox = item.get("bbox") or item.get("box") or item.get("polygon")
        coords = parse_bbox_values(raw_bbox)
        if not coords:
            continue

        x1, y1, x2, y2 = coords

        # Scale detection:
        # Chandra coordinates are normalized 0..1000 (or 0..1)
        if max(x1, x2) <= 1.05 and img_width > 1:
            x1 = x1 * img_width
            x2 = x2 * img_width
            y1 = y1 * img_height
            y2 = y2 * img_height
        elif max(x1, x2) <= 1000.0 and max(y1, y2) <= 1000.0:
            x1 = x1 * img_width / 1000.0
            x2 = x2 * img_width / 1000.0
            y1 = y1 * img_height / 1000.0
            y2 = y2 * img_height / 1000.0

        # Clip within image boundaries
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

        score = float(item.get("score", item.get("confidence", default_score)))
        detections.append(
            {
                "category_id": cat_id,
                "category_name": cat_name,
                "bbox": [round(bx, 2), round(by, 2), round(bw, 2), round(bh, 2)],
                "score": round(score, 4),
            }
        )

    return detections
