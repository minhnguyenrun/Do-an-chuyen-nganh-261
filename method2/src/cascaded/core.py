"""Pure geometry cascade. Models are injected; no inference at import time."""
from copy import deepcopy
from statistics import median
import math
from shapely.geometry import box
from .types import Proposal
from .v3_geometry import line_shape, proposal_shape, set_proposal_bbox


VISUAL = {'image', 'chart', 'table', 'formula'}
TEXT = {'text', 'doc_title', 'paragraph_title', 'title', 'caption', 'footnote',
        'header', 'footer', 'list_item'}


def area(b):
    return max(0, b[2]-b[0]) * max(0, b[3]-b[1])


def intersect(a, b):
    return [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]


def iou(a, b):
    overlap = area(intersect(a, b))
    return overlap / max(1e-9, area(a)+area(b)-overlap)


def envelope(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def line_box(line):
    return [min(p[0] for p in line.polygon), min(p[1] for p in line.polygon),
            max(p[0] for p in line.polygon), max(p[1] for p in line.polygon)]


def coverage(line, bbox):
    polygon = line_shape(line)
    return polygon.intersection(box(*bbox)).area / polygon.area


def deduplicate(proposals, cfg):
    kept = []
    for p in sorted(deepcopy(proposals), key=lambda p: (-(p.score or 0), p.source != 'paddle', p.id)):
        match = next((q for q in kept if q.label == p.label and iou(q.bbox, p.bbox) >= cfg.nms_iou), None)
        if match:
            match.source_ids = list(dict.fromkeys(match.source_ids+p.source_ids))
        else:
            kept.append(p)
    return kept


def assign(regions, lines, threshold, *, use_v3_polygon=False, preserved_owners=None):
    for r in regions:
        r.line_ids = []
    if not regions:
        return
    shapes = [(r, proposal_shape(r, use_v3_polygon=use_v3_polygon)) for r in regions]
    for line in lines:
        polygon = line_shape(line)
        choices = [(polygon.intersection(shape).area / polygon.area, -area(r.bbox), r.id, r)
                   for r, shape in shapes]
        choices = [c for c in choices if c[0] >= threshold]
        if choices:
            owner = preserved_owners.get(line.id) if preserved_owners else None
            preserved = next((c for c in choices if c[2] == owner), None) if owner is not None else None
            (preserved or max(choices, key=lambda c: c[:3]))[3].line_ids.append(line.id)


def unresolved(regions, lines):
    assigned = {lid for r in regions for lid in r.line_ids}
    return [l for l in lines if l.id not in assigned]


def cluster(lines):
    """Require horizontal overlap: never connect separated document columns."""
    groups = []
    for line in sorted(lines, key=lambda l: (line_box(l)[1], line_box(l)[0], l.id)):
        b = line_box(line)
        matches = []
        for group in groups:
            a = envelope([line_box(l) for l in group])
            overlap = max(0, min(a[2], b[2])-max(a[0], b[0])) / max(1, min(a[2]-a[0], b[2]-b[0]))
            h = median(line_box(l)[3]-line_box(l)[1] for l in group+[line])
            gap = max(a[1]-b[3], b[1]-a[3], 0)
            if overlap >= .5 and gap <= h * 1.5:
                matches.append(group)
        if len(matches) == 1:
            matches[0].append(line)
        else:
            groups.append([line])
    return groups


def prepare(proposals, lines, cfg):
    proposals = deduplicate([p for p in proposals if p.score is not None and p.score >= cfg.low], cfg)
    conflicts = {p.id for p in proposals for q in proposals
                 if p.id != q.id and p.label != q.label and iou(p.bbox, q.bbox) >= cfg.nms_iou}
    strong = [deepcopy(p) for p in proposals if p.score >= cfg.high and p.id not in conflicts]
    for r in strong:
        r.stage = 'strong'
    assign(strong, lines, cfg.line_coverage)
    tier1 = deepcopy(strong)
    weak = [deepcopy(p) for p in proposals if p.score < cfg.high and p.id not in conflicts]
    remaining = unresolved(strong, lines)
    # All weak proposals compete before any assignment, avoiding order-dependent recovery.
    recovered = []
    for p in weak:
        supported = [l for l in remaining if coverage(l, p.bbox) >= cfg.line_coverage]
        if not supported or any(iou(p.bbox, r.bbox) >= cfg.nms_iou for r in strong):
            continue
        if any(q.id != p.id and q.label != p.label and
               any(coverage(l, q.bbox) >= cfg.line_coverage for l in supported) for q in weak):
            continue
        p.stage, p.reasons = 'recovery', ['weak_box_supported_by_ocr']
        recovered.append(p)
    strong += recovered
    assign(strong, lines, cfg.line_coverage)
    heights = [line_box(l)[3]-line_box(l)[1] for l in lines]
    h = median(heights) if heights else 1
    # Decisions use a snapshot; multiple plausible owners leave a line orphaned.
    additions = []
    for line in unresolved(strong, lines):
        b = line_box(line)
        owners = []
        for r in strong:
            a = r.bbox
            gap = max(a[1]-b[3], b[1]-a[3], 0)
            horizontal = max(0, min(a[2], b[2])-max(a[0], b[0])) / max(1, b[2]-b[0])
            expanded = envelope([a, b])
            blocked = any(q.id != r.id and q.label in VISUAL and
                          area(intersect(expanded, q.bbox)) > area(intersect(a, q.bbox))
                          for q in proposals)
            # Do not absorb a line covered by an unresolved conflicting detector proposal.
            disputed = any(p.id in conflicts and coverage(line, p.bbox) >= cfg.line_coverage for p in proposals)
            if r.label in TEXT and horizontal >= .8 and gap <= h and abs(a[0]-b[0]) <= h and not blocked and not disputed:
                owners.append(r)
        if len(owners) == 1:
            additions.append((owners[0], line))
    for r, line in additions:
        set_proposal_bbox(r, envelope([r.bbox, line_box(line)]))
        r.line_ids.append(line.id)
        r.stage = 'recovery'
        r.reasons.append('aligned_adjacent_line')
    targets = []
    remaining = unresolved(strong, lines)
    for i, group in enumerate(cluster(remaining)):
        targets.append({'id': f'target:{i}', 'bbox': envelope([line_box(l) for l in group]),
                        'line_ids': [l.id for l in group], 'source_ids': [], 'reason': 'orphan'})
    # Include conflicts without OCR as review evidence, even though VLM acceptance needs lines.
    for p in proposals:
        if p.id in conflicts:
            related = [t for t in targets if area(intersect(t['bbox'], p.bbox)) > 0]
            if related:
                for t in related:
                    t['source_ids'] = list(dict.fromkeys(t['source_ids']+p.source_ids))
                    t['reason'] = 'detector_conflict'
            elif not any(iou(p.bbox, t['bbox']) >= cfg.nms_iou for t in targets):
                targets.append({'id': f'target:{len(targets)}', 'bbox': p.bbox,
                                'line_ids': [], 'source_ids': p.source_ids, 'reason': 'conflict_without_ocr'})
    return tier1, strong, targets


def context_box(target, lines, width, height):
    a, b, c, d = target['bbox']
    selected = [l for l in lines if l.id in target['line_ids']]
    h = median(line_box(l)[3]-line_box(l)[1] for l in selected) if selected else 10
    dx, dy = max((c-a)*.2, 2*h), max((d-b)*.2, 2*h)
    return [max(0, math.floor(a-dx)), max(0, math.floor(b-dy)),
            min(width, math.ceil(c+dx)), min(height, math.ceil(d+dy))]


def to_page_bbox(normalized, crop_bbox):
    if (len(normalized) != 4 or not all(isinstance(v, (float, int)) and math.isfinite(v) and 0 <= v <= 1 for v in normalized)
            or normalized[0] >= normalized[2] or normalized[1] >= normalized[3]):
        raise ValueError('MinerU bbox must be normalized, finite and nonempty')
    x, y, right, bottom = crop_bbox
    return [x+normalized[0]*(right-x), y+normalized[1]*(bottom-y),
            x+normalized[2]*(right-x), y+normalized[3]*(bottom-y)]


def finish(regions, targets, lines, predictions, cfg):
    final = deepcopy(regions)
    by_id = {l.id: l for l in lines}
    audit = []
    for t in targets:
        response = predictions.get(t['id'], {'status': 'budget_exhausted', 'blocks': []})
        audit.append(dict(target=t, **response))
        accepted = []
        if response['status'] == 'ok':
            try:
                candidates = []
                for i, block in enumerate(response['blocks']):
                    bbox = intersect(to_page_bbox(block['bbox'], response['crop_bbox']), t['bbox'])
                    if area(bbox) <= 0 or block['label'] == 'unknown':
                        continue
                    own = [by_id[id] for id in t['line_ids'] if coverage(by_id[id], bbox) >= cfg.line_coverage]
                    if not own or len(cluster(own)) != 1:
                        continue
                    # Expand only to the measured target lines, preserving complete polygons.
                    bbox = envelope([bbox]+[line_box(l) for l in own])
                    if any(iou(bbox, r.bbox) >= cfg.nms_iou or
                           (r.label in VISUAL and area(intersect(bbox, r.bbox)) > 0) for r in final):
                        continue
                    candidates.append(Proposal(f"mineru:{t['id']}:{i}", bbox, block['label'], 'mineru',
                                               block.get('raw_label', block['label']), None,
                                               [l.id for l in own], t['source_ids'] or [t['id']],
                                               'mineru', ['layout_only_target_line_support']))
                # Reject competing labels instead of taking the first model block.
                candidates = [p for p in candidates if not any(q.id != p.id and p.label != q.label and
                               set(p.line_ids) & set(q.line_ids) for q in candidates)]
                used = set()
                for p in deduplicate(candidates, cfg):
                    if not used.intersection(p.line_ids):
                        accepted.append(p)
                        used.update(p.line_ids)
            except (ValueError, KeyError, TypeError) as exc:
                audit[-1]['status'], audit[-1]['error'] = 'invalid_output', str(exc)
                accepted = []
        final += accepted
        used = {lid for r in final for lid in r.line_ids}
        missing = [by_id[id] for id in t['line_ids'] if id not in used]
        for i, group in enumerate(cluster(missing)):
            final.append(Proposal(f"unknown:{t['id']}:{i}", envelope([line_box(l) for l in group]),
                                  'unknown', 'orphan', 'unknown', None, [l.id for l in group],
                                  t['source_ids'] or [t['id']], 'unresolved', [audit[-1]['status']], True))
        if not t['line_ids']:
            final.append(Proposal(f"unknown:{t['id']}", t['bbox'], 'unknown', 'conflict', 'unknown',
                                  None, [], t['source_ids'], 'unresolved', ['no_target_lines'], True))
    assigned = [lid for r in final for lid in r.line_ids]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(by_id):
        raise ValueError('cascade lost or duplicated OCR lines')
    return sorted(final, key=lambda r: (r.bbox[1], r.bbox[0], r.id)), audit
