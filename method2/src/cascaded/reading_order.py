"""Page-local paragraph boundaries evidenced by measured Paddle V3 order."""
from collections import Counter
from numbers import Integral
from statistics import median

from .core import VISUAL, cluster, envelope, line_box
from .semantic_gate import expansion_supported
from .v3_geometry import proposal_intersection_area, v3_polygons_enabled


BODY = {'text', 'list_item'}


def _same_column_corridor(before, after):
    a, b = before.bbox, after.bbox
    left, right = max(a[0], b[0]), min(a[2], b[2])
    if a[3] < b[1] and right-left >= .70*max(a[2]-a[0], b[2]-b[0]):
        return [left, a[3], right, b[1]]
    return None


def build_reading_graph(proposals, lines, cfg):
    """Only consecutive, globally unambiguous model ranks can form an edge."""
    enabled = cfg.variant == 'v3' and cfg.enable_v3_reading_order
    graph = {'edges': [], 'rank_count': 0, 'reasons': [],
             'use_v3_polygon': v3_polygons_enabled(cfg)}
    if not enabled or not lines:
        graph['reasons'].append('reading_order_disabled' if not enabled else 'no_ocr_lines')
        return graph

    ranked = [p for p in proposals if p.source == 'paddle'
              and isinstance(p.reading_order_rank, Integral)
              and not isinstance(p.reading_order_rank, bool) and p.reading_order_rank >= 0]
    graph['rank_count'] = len(ranked)
    ranks = Counter(p.reading_order_rank for p in ranked)
    ids = Counter(p.id for p in proposals)
    anchors = {int(p.reading_order_rank): p for p in ranked if ranks[p.reading_order_rank] == 1
               and ids[p.id] == 1 and p.label in BODY}
    for rank, before in sorted(anchors.items()):
        after = anchors.get(rank+1)
        if after is None:
            continue
        corridor = _same_column_corridor(before, after)
        if corridor is None:
            continue
        graph['edges'].append({'before_id': before.id, 'after_id': after.id,
                               'before_rank': rank, 'after_rank': rank+1,
                               'corridor': corridor})
    if any(count > 1 for count in ranks.values()):
        graph['reasons'].append('ambiguous_model_ranks')
    if not graph['edges']:
        graph['reasons'].append('no_directional_same_column_edges' if ranked else 'no_measured_ranks')
    return graph


def _horizontal_coverage(bbox, corridor):
    overlap = max(0, min(bbox[2], corridor[2])-max(bbox[0], corridor[0]))
    return overlap / max(1e-9, bbox[2]-bbox[0])


def _inside_corridor(bounds, boxes, corridor):
    return (corridor is not None and corridor[1] <= bounds[1] and bounds[3] <= corridor[3]
            and all(_horizontal_coverage(bbox, corridor) >= .80 for bbox in boxes))


def graph_neighbor(region, orphan_lines, graph, lines, confirmed):
    """Find a sole nearby confirmed endpoint for lines inside an order corridor."""
    def abstain(*reasons, evidence=None):
        return {'supported': False, 'neighbor_id': None, 'before_id': None, 'after_id': None,
                'reasons': list(reasons), 'evidence': evidence or {}}

    if not graph['edges']:
        return abstain(*graph['reasons'])
    if not orphan_lines:
        return abstain('no_orphan_lines')
    orphan_ids = {line.id for line in orphan_lines}
    if len(orphan_ids) != len(orphan_lines) or any(orphan_ids.intersection(p.line_ids) for p in confirmed):
        return abstain('lines_already_owned_or_repeated')
    boxes = [line_box(line) for line in orphan_lines]
    bounds = envelope(boxes)
    by_id = {p.id: p for p in confirmed}
    recipients = []
    current_geometry_rejected = False
    for edge in graph['edges']:
        before_id, after_id = edge['before_id'], edge['after_id']
        if before_id not in by_id or after_id not in by_id:
            continue
        if by_id[before_id].label not in BODY or by_id[after_id].label not in BODY:
            continue
        measured_corridor = edge['corridor']
        if not _inside_corridor(bounds, boxes, measured_corridor):
            continue
        current_corridor = _same_column_corridor(by_id[before_id], by_id[after_id])
        if not _inside_corridor(bounds, boxes, current_corridor):
            current_geometry_rejected = True
            continue
        corridors = [measured_corridor] if current_corridor == measured_corridor else [measured_corridor, current_corridor]
        if any(p.id not in (before_id, after_id) and any(proposal_intersection_area(
                corridor, p, use_v3_polygon=graph['use_v3_polygon'] and p.label in VISUAL) > 0
                for corridor in corridors)
               for p in confirmed):
            continue
        for candidate_id, boundary in ((before_id, 'last'), (after_id, 'first')):
            candidate = by_id[candidate_id]
            gap = bounds[1]-candidate.bbox[3] if boundary == 'last' else candidate.bbox[1]-bounds[3]
            owned_ids = set(candidate.line_ids)
            owned = [line for line in lines if line.id in owned_ids]
            if not owned or candidate.label not in BODY:
                continue
            heights = [bbox[3]-bbox[1] for bbox in map(line_box, owned)]
            if (0 <= gap <= median(heights)
                    and all(_horizontal_coverage(bbox, candidate.bbox) >= .80 for bbox in boxes)
                    and all(expansion_supported(owned, line, lines) for line in orphan_lines)):
                recipients.append(dict(edge, neighbor_id=candidate_id, boundary=boundary,
                                       current_corridor=current_corridor, gap=gap,
                                       line_ids=[line.id for line in orphan_lines]))
    if not recipients:
        return abstain('current_anchor_geometry_ineligible' if current_geometry_rejected else 'no_eligible_boundary')
    if len(recipients) != 1:
        return abstain('competing_boundary_recipients', evidence={'recipients': recipients})
    evidence = recipients[0]
    if region is not None and region.id != evidence['neighbor_id']:
        return abstain('different_boundary_recipient', evidence=evidence)
    return {'supported': True, 'neighbor_id': evidence['neighbor_id'],
            'before_id': evidence['before_id'], 'after_id': evidence['after_id'],
            'reasons': ['unique_directional_same_column_boundary'], 'evidence': evidence}


def cluster_orphans(orphan_lines, graph, lines, confirmed):
    """Split legacy clusters only when distinct boundaries have positive evidence."""
    for group in cluster(orphan_lines):
        if not graph['edges'] or len(group) == 1:
            yield group
            continue
        by_boundary = {}
        for line in group:
            association = graph_neighbor(None, [line], graph, lines, confirmed)
            by_boundary.setdefault(association['neighbor_id'], []).append(line)
        if len(by_boundary.keys()-{None}) <= 1:
            yield group
        else:
            groups = [part for batch in by_boundary.values() for part in cluster(batch)]
            yield from sorted(groups, key=lambda part: (line_box(part[0])[1], line_box(part[0])[0], part[0].id))
