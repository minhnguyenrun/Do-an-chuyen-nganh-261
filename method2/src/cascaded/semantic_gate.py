"""Inspectible OCR/geometry evidence, not calibrated label probabilities."""
import re
from statistics import median
from .core import line_box

CAPTION = re.compile(r'^\s*(?:fig(?:ure)?\.?|table|chart|hình|bảng|biểu đồ)\s*(?:\d|[IVX]+)', re.I)
LIST = re.compile(r'^\s*(?:[•●▪*-]|\(?\d+[.)]|[A-Za-z][.)])\s+')


def semantic_score(label, lines, page_lines, bbox, width, height):
    present = [l for l in lines if l.text.strip() and l.rec_score is not None and l.rec_score >= .75]
    if not lines or len(present) != len(lines):
        return 0., ['missing_or_low_recognition_confidence']
    text = ' '.join(l.text.strip() for l in lines)
    boxes = [line_box(l) for l in lines]
    sizes = [b[3]-b[1] for b in boxes]
    body = median(line_box(l)[3]-line_box(l)[1] for l in page_lines) if page_lines else median(sizes)
    h = median(sizes)
    cy = (bbox[1]+bbox[3])/2/height
    short = len(lines) <= 3 and len(text.split()) <= 35
    letters = [c for c in text if c.isalpha()]
    uppercase = len(letters) >= 3 and sum(c.isupper() for c in letters)/len(letters) >= .8
    heading = short and not text.rstrip().endswith(('.', ';', ',')) and (h >= 1.15*body or uppercase)
    caption = bool(CAPTION.search(text))
    if label == 'text':
        supported = len(text.split()) >= 3 and not caption and not LIST.search(text) and not heading
        return (.8 if supported else .2), ['body_transcript' if supported else 'body_role_ambiguous']
    if label in ('title','doc_title','paragraph_title'):
        return (1. if heading else .2), ['short_heading_size_or_uppercase' if heading else 'heading_evidence_missing']
    if label == 'caption':
        # Lack of a prefix is uncertainty, not proof that it cannot be a caption.
        return (1. if caption else .3), ['caption_prefix' if caption else 'caption_prefix_unavailable']
    if label == 'list_item':
        return (1. if LIST.search(text) else .2), ['list_marker' if LIST.search(text) else 'list_marker_missing']
    if label == 'table':
        rows = []
        for b in sorted(boxes, key=lambda b:(b[1],b[0])):
            center = (b[1]+b[3])/2
            row = next((r for r in rows if abs(r[0]-center) <= max(3,h*.5)), None)
            if row is None:
                rows.append([center,[b]])
            else:
                row[1].append(b)
        grid_rows = sum(any(a[2] < b[0] for a,b in zip(sorted(r[1]),sorted(r[1])[1:])) for r in rows)
        return (1. if grid_rows >= 2 else .1), ['multiple_geometric_rows_and_columns' if grid_rows >= 2 else 'table_grid_missing']
    if label == 'formula':
        operators = len(re.findall(r'[=+^_∑∫√≤≥]|\\(?:frac|sum|int|sqrt)',text))
        density = len(re.findall(r'[=+^_∑∫√≤≥*/-]',text))/max(1,len(re.sub(r'\s','',text)))
        supported = len(lines) <= 3 and operators >= 2 and (density > .25 or len(text.split()) <= 15)
        return (1. if supported else .1), ['math_transcript_and_short_block' if supported else 'math_evidence_missing']
    if label in ('header','footer','footnote'):
        position = cy < .08 if label == 'header' else cy > .90
        supported = position and h <= body*.95
        return (1. if supported else .2), ['margin_and_small_font' if supported else 'margin_role_unverified']
    # OCR lines cannot establish that a crop is a picture.
    return 0., ['unsupported_semantic_role:'+label]


def expansion_supported(region_lines, extra, page_lines):
    if not region_lines or extra.rec_score is None or extra.rec_score < .75:
        return False
    height = median(line_box(l)[3]-line_box(l)[1] for l in region_lines)
    size = line_box(extra)[3]-line_box(extra)[1]
    text = extra.text.strip()
    return (abs(size-height)/max(1,height) < .20 and not CAPTION.search(text)
            and not LIST.search(text) and not (text.isupper() and len(text) >= 3))
