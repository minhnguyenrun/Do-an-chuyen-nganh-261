import re
from statistics import median

import cv2
import numpy as np

from .residual_prompt import UNLABELED_LABELS, build_classification_prompt, parse_classification_with_note
from .evidence_features import spatial_grid
from ..spatial.coordinates import polygon_bbox
from ..spatial.crop_builder import build_crop


def build_target_crop(image, bbox):
    """Only target pixels, without the full page, neighbour margin or annotations."""
    target = build_crop(image, bbox, mode='raw')
    height, width = target.shape[:2]
    scale = min(3., 1280 / max(height, width))
    if scale != 1:
        target = cv2.resize(target, (max(1, round(width * scale)), max(1, round(height * scale))),
                            interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    return target


# Backwards-compatible name; the returned image now contains only target pixels.
build_context_crop = build_target_crop


async def classify_unlabeled_regions(image, graph, contents, client, scheduler):
    regions = [r for r in graph.nodes.values() if r.label in UNLABELED_LABELS]
    requests = [(f'classify:{r.id}', build_target_crop(image, r.bbox),
                 build_classification_prompt(r, (graph.page_w, graph.page_h), contents.get(r.id, '')))
                for r in regions]
    raw = await scheduler.run(client, requests)
    evidence_map, evidence = {}, []
    def validate(result):
        if result.status == 'ok':
            try:
                classification, note = parse_classification_with_note(result.content)
                result = result.model_copy(update={'classification': classification,
                    'error': '; '.join(filter(None, (result.error, note)))})
            except ValueError as exc:
                result = result.model_copy(update={'status': 'invalid_classification', 'error': str(exc)})
        return result
    retry_regions = []
    for region, result in zip(regions, raw):
        result = validate(result)
        evidence_map[region.id] = result
        evidence.append(result)
        if result.status in ('invalid_classification', 'incomplete'):
            retry_regions.append(region)
    if retry_regions:
        retries = [(f'classify:{r.id}:retry', build_target_crop(image, r.bbox),
                    build_classification_prompt(r, (graph.page_w, graph.page_h),
                                                contents.get(r.id, ''), retry=True)) for r in retry_regions]
        retried = await scheduler.run(client, retries)
        for region, result in zip(retry_regions, retried):
            result = validate(result)
            evidence.append(result)
            if result.status != 'budget_exhausted':
                evidence_map[region.id] = result
    return evidence_map, evidence


def residual_role_evidence(hyp, graph, lines):
    """Independent layout checks; a valid JSON label alone is never sufficient."""
    if len(hyp.source_region_ids) != 1 or graph.nodes[hyp.source_region_ids[0]].label not in UNLABELED_LABELS:
        return False, ['classification_requires_unlabeled_source']
    if not lines:
        return False, ['classification_requires_text_lines']
    a, b, c, d = hyp.candidate_bbox
    cy = (b + d) / 2 / graph.page_h
    boxes = [polygon_bbox(line.polygon) for line in lines]
    heights = [box[3] - box[1] for box in boxes]
    all_heights = [polygon_bbox(line.polygon)[3] - polygon_bbox(line.polygon)[1] for line in graph.lines]
    body_height = median(all_heights) if all_heights else median(heights)
    transcript = '\n'.join(line.transcript for line in lines)
    label = hyp.candidate_label
    supported = False
    reason = ''
    if label == 'text':
        supported = .12 < cy < .88 and (len(lines) >= 2 or len(transcript.split()) >= 6)
        reason = 'body_text_position_and_lines'
    elif label == 'header':
        supported = d <= graph.page_h * .16 and len(lines) <= 5
        reason = 'top_page_margin'
    elif label == 'footer':
        supported = b >= graph.page_h * .84 and len(lines) <= 5
        reason = 'bottom_page_margin'
    elif label == 'title':
        supported = len(lines) <= 4 and len(transcript.split()) <= 35 and (
            max(heights) >= body_height * 1.2 or (cy < .25 and max(heights) >= body_height * 1.05))
        reason = 'heading_size_and_short_block'
    elif label == 'figure_title':
        visuals = [r for r in graph.nodes.values() if r.label in ('image', 'figure', 'chart', 'table')]
        for region in visuals:
            x1, y1, x2, y2 = region.bbox
            overlap = max(0, min(c, x2) - max(a, x1)) / max(1, min(c - a, x2 - x1))
            gap = max(b - y2, y1 - d, 0)
            vertical_overlap = max(0, min(d, y2) - max(b, y1)) / max(1, min(d - b, y2 - y1))
            horizontal_gap = max(a - x2, x1 - c, 0)
            adjacent = (overlap >= .25 and gap <= max(35, graph.page_h * .06)) or (
                vertical_overlap >= .5 and horizontal_gap <= max(35, graph.page_w * .06))
            if adjacent and len(lines) <= 6:
                # Text deep inside a picture is not independent evidence of a caption.
                inside = (a >= x1 and c <= x2 and b >= y1 and d <= y2)
                if not inside:
                    supported = True
                    break
        reason = 'caption_adjacent_to_visual_region'
    elif label == 'footnote':
        supported = b >= graph.page_h * .65 and median(heights) <= body_height * .9
        reason = 'small_text_in_lower_page'
    elif label == 'list_item':
        supported = bool(re.search(r'(?m)^\s*(?:[•●▪▫◦‣*–-]|\(?\d+[.)]|[A-Za-z][.)])\s+', transcript))
        reason = 'independent_ocr_list_marker'
    elif label == 'table':
        supported = bool(spatial_grid(lines)[2])
        reason = 'aligned_line_grid'
    elif label == 'formula':
        supported = len(lines) <= 3 and len(re.findall(r'[=+^_∑∫√]|\\(?:frac|sum|int)', transcript)) >= 2
        reason = 'independent_ocr_math'
    return supported, [reason] if supported else [f'classification_spatial_mismatch:{label}']
