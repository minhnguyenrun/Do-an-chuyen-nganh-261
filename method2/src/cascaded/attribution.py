"""V2.1 MinerU adjudication by frozen OCR ownership, never by target-box area.

Geometry and OCR ownership are hard evidence gates. The semantic score can
come from either the legacy transcript checks or the optional learned head.
"""
from copy import deepcopy
from math import atan2, degrees, isfinite
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from .types import Proposal, label_for, serialize_proposal
from .core import VISUAL, intersect, iou, envelope, coverage, line_box, cluster, to_page_bbox
from .fusion import quality, decision, scored_semantics
from .semantic_gate import expansion_supported
from .geometry_refine import refine_candidate_bbox
from .reading_order import build_reading_graph, graph_neighbor
from .v3_geometry import protected_overlap, proposal_intersection_area, set_proposal_bbox, v3_polygons_enabled


def horizontal(line):
    points = list(Polygon(line.polygon).minimum_rotated_rectangle.exterior.coords)[:4]
    edges = [(a, b) for a, b in zip(points, points[1:]+points[:1])]
    a, b = max(edges, key=lambda e: (e[1][0]-e[0][0])**2+(e[1][1]-e[0][1])**2)
    angle = abs(degrees(atan2(b[1]-a[1], b[0]-a[0]))) % 180
    return min(angle, 180-angle) <= 10


def line_link(line, bbox, cfg):
    b = line_box(line)
    dx, dy = max(1e-9, b[2]-b[0]), max(1e-9, b[3]-b[1])
    cross = intersect(b, bbox)
    cx = max(0, cross[2]-cross[0])/dx
    cy = max(0, cross[3]-cross[1])/dy
    poly = coverage(line, bbox)
    center_margin = min((b[1]+b[3])/2-bbox[1], bbox[3]-(b[1]+b[3])/2)/dy
    snap = (poly < cfg.line_coverage and horizontal(line) and cx >= cfg.snap_cov_x
            and cy >= cfg.snap_cov_y and center_margin >= cfg.snap_center_margin)
    return {'line_id': line.id, 'polygon_coverage': poly, 'cov_x': cx, 'cov_y': cy,
            'center_margin': center_margin, 'normal': poly >= cfg.line_coverage,
            'snap': snap, 'horizontal': horizontal(line)}


def ink_evidence(bbox, target_ids, lines, owners, confirmed, *, use_v3_polygon=False):
    """Use OCR polygon area, counting overlapping ink only once per category."""
    window = box(*bbox)
    touched = [(l, Polygon(l.polygon).intersection(window)) for l in lines]
    touched = [(l, shape) for l, shape in touched if shape.area > 1e-8]
    total = unary_union([shape for _, shape in touched]).area
    foreign = [(l, shape) for l, shape in touched if l.id not in target_ids]
    foreign_area = unary_union([shape for _, shape in foreign]).area
    neighbor_ids = sorted({owners[l.id] for l, _ in foreign if l.id in owners})
    other_orphans = [l.id for l, _ in foreign if l.id not in owners]
    overlaps = {r.id: protected_overlap(bbox, r, use_v3_polygon=use_v3_polygon)
                for r in confirmed if r.label in VISUAL}
    return {'non_target_ink_ratio': foreign_area/max(1e-9, total),
            'target_ink_area': unary_union([s for l, s in touched if l.id in target_ids]).area,
            'foreign_line_ids': [l.id for l, _ in foreign], 'neighbor_ids': neighbor_ids,
            'other_orphan_line_ids': other_orphans,
            'nontext_overlap': max(overlaps.values(), default=0.), 'nontext_overlaps': overlaps}


def absorb_region(neighbor, own, lines, owners, confirmed, cfg, width, height, candidate_id,
                  *, reading_association=None, expanded_during_preparation=False, semantic=None):
    old = [l for l in lines if l.id in neighbor.line_ids]
    if neighbor.label != 'text' or not old or not all(expansion_supported(old, l, lines) for l in own):
        return None, 'ABSORB_FONT_OR_ROLE_MISMATCH'
    # Alignment and connected lines establish the column; the model box is not used.
    a = neighbor.bbox
    graph_first_line = (reading_association is not None and reading_association['supported']
                        and reading_association['neighbor_id'] == reading_association['after_id'] == neighbor.id)
    first = min(own, key=lambda line: (line_box(line)[1], line_box(line)[0], line.id)) if graph_first_line else None
    for l in own:
        b = line_box(l)
        h = b[3]-b[1]
        overlap = max(0, min(a[2], b[2])-max(a[0], b[0]))/max(1, min(a[2]-a[0], b[2]-b[0]))
        if overlap < cfg.snap_cov_x:
            return None, 'ABSORB_COLUMN_MISMATCH'
        if abs(a[0]-b[0]) > h:
            if l is not first or b[0] <= a[0]:
                return None, 'ABSORB_COLUMN_MISMATCH'
            if expanded_during_preparation:
                return None, 'NEIGHBOR_ALREADY_EDITED'
    if len(cluster(old+own)) != 1:
        return None, 'ABSORB_DISCONNECTED_LINES'
    r = deepcopy(neighbor)
    set_proposal_bbox(r, envelope([neighbor.bbox]+[line_box(l) for l in own]))
    r.polygon = None  # ownership now includes line evidence beyond the original detection
    r.line_ids = list(dict.fromkeys(neighbor.line_ids+[l.id for l in own]))
    r.source_ids = list(dict.fromkeys(r.source_ids+[candidate_id]))
    r.stage, r.score = 'mineru_absorb', None  # old detector score does not score new geometry
    r.reasons = r.reasons+['ACCEPTED_NEIGHBOR_MERGE']
    others = [p for p in confirmed if p.id != neighbor.id]
    # The expanded envelope must not sweep in a second owner or another orphan.
    use_v3_polygon = v3_polygons_enabled(cfg)
    ev = ink_evidence(r.bbox, set(r.line_ids), lines, owners, others, use_v3_polygon=use_v3_polygon)
    if ev['foreign_line_ids'] or ev['nontext_overlap'] > cfg.max_nontext_overlap:
        return None, 'ABSORB_ENVELOPE_CONTAMINATION'
    if any(proposal_intersection_area(r.bbox, p, use_v3_polygon=use_v3_polygon and p.label in VISUAL)
           > proposal_intersection_area(a, p, use_v3_polygon=use_v3_polygon and p.label in VISUAL)+1e-8
           for p in others):
        return None, 'ABSORB_INTERSECTS_OTHER_REGION'
    if not all(coverage(l, r.bbox) >= cfg.line_coverage for l in old+own):
        return None, 'ABSORB_LOST_OLD_LINES'
    sem, _, _ = scored_semantics('text', old+own, lines, r.bbox, width, height,
                                 enabled=cfg.enable_semantic_quality, semantic=semantic)
    if sem < cfg.min_semantic_score:
        return None, 'semantic_gate_failed'
    return r, None


def candidate_boxes(t, response, snapshot, lines, cfg, width, height, edited, *, reading_graph=None, semantic=None):
    target_ids = set(t['line_ids'])
    fallback = {p['id'] for p in t['fallback_baseline']}
    confirmed = [p for p in snapshot if p.id not in fallback]
    owners = {lid: p.id for p in confirmed for lid in p.line_ids}
    byregion = {p.id: p for p in confirmed}
    target_lines = [l for l in lines if l.id in target_ids]
    use_v3_polygon = v3_polygons_enabled(cfg)
    if reading_graph is None:
        reading_graph = build_reading_graph(snapshot, lines, cfg)
    pool, history = [], []
    for i, b in enumerate(response['blocks']):
        h = {'box_index': i, 'raw_bbox': b.get('bbox') if isinstance(b, dict) else b,
             'codes': [], 'selected_partition': False}
        history.append(h)
        try:
            if not isinstance(b, dict):
                raise ValueError('block must be an object')
            bbox = to_page_bbox(b['bbox'], response['crop_bbox'])
            h['page_bbox'] = list(bbox)
            if cfg.enable_geometry_refinement:
                label = label_for(b.get('raw_label', b['label']), 'mineru')
                h['pre_refinement_line_links'] = [line_link(l, bbox, cfg) for l in target_lines
                                                  if coverage(l, bbox) > 1e-8]
                refinement = refine_candidate_bbox(bbox, label, target_lines, lines,
                                                   owners, confirmed, cfg)
                h.update(label=label, geometry_refinement=refinement,
                         refined_bbox=list(refinement['bbox']))
                h['codes'] += refinement['codes']
                if not refinement['valid']:
                    continue
                bbox = refinement['bbox']
            touched = [l for l in target_lines if coverage(l, bbox) > 1e-8]
            if not touched:
                h['codes'].append('PURE_CONTEXT_BOX')
                continue
            links = [line_link(l, bbox, cfg) for l in touched]
            h['line_links'] = links
            own = [l for l, link in zip(touched, links) if link['normal'] or link['snap']]
            bad = [v for v in links if not (v['normal'] or v['snap'])]
            if bad:
                h['codes'] += sorted({'HORIZONTAL_SPLIT_UNVERIFIED' if v['cov_x'] < cfg.snap_cov_x
                                     and v['cov_y'] >= cfg.snap_cov_y else 'LINE_BISECTION' for v in bad})
                continue
            before = list(bbox)
            if any(v['snap'] for v in links):
                bbox = [bbox[0], min([bbox[1]]+[line_box(l)[1] for l in own]),
                        bbox[2], max([bbox[3]]+[line_box(l)[3] for l in own])]
                h['codes'].append('VERTICAL_SNAP_APPLIED')
            # Snapping can touch additional short/overlapping target lines.
            # Recompute ownership as well as contamination on the final geometry.
            post_touched = [l for l in target_lines if coverage(l, bbox) > 1e-8]
            post_links = [line_link(l, bbox, cfg) for l in post_touched]
            h.update(pre_snap_bbox=before, post_snap_bbox=list(bbox), post_snap_links=post_links)
            ev = ink_evidence(bbox, target_ids, lines, owners, confirmed, use_v3_polygon=use_v3_polygon)
            h['attribution'] = ev
            if any(not v['normal'] for v in post_links):
                h['codes'] += sorted({'HORIZONTAL_SPLIT_UNVERIFIED' if v['cov_x'] < cfg.snap_cov_x
                                     and v['cov_y'] >= cfg.snap_cov_y else 'LINE_BISECTION'
                                     for v in post_links if not v['normal']})
                continue
            own = post_touched
            if len(cluster(own)) != 1:
                h['codes'].append('cross_column_or_disconnected_lines')
                continue
            if ev['nontext_overlap'] > cfg.max_nontext_overlap:
                h['codes'].append('PROTECTED_NONTEXT_OVERLAP')
                continue
            label = label_for(b.get('raw_label', b['label']), 'mineru')
            score, why, prediction = scored_semantics(
                label, own, lines, bbox, width, height, enabled=cfg.enable_semantic_quality, semantic=semantic)
            h.update(label=label, semantic_score=score, semantic_reasons=why)
            if prediction is not None:
                h['semantic_prediction'] = prediction
            if score < cfg.min_semantic_score:
                h['codes'].append('semantic_gate_failed')
                continue
            id = f"mineru:{t['id']}:{i}"
            absorb = None
            # A box touching an owned line belongs to the ABSORB path, even if
            # that small line contributes <10% of the total OCR polygon area.
            if ev['neighbor_ids']:
                if len(ev['neighbor_ids']) == 1 and ev['neighbor_ids'][0] in edited:
                    h['codes'].append('NEIGHBOR_ALREADY_EDITED')
                    continue
                if len(ev['neighbor_ids']) != 1 or ev['other_orphan_line_ids'] or fallback or label != 'text':
                    h['codes'].append('FOREIGN_CONTAMINATION')
                    continue
                absorb = ev['neighbor_ids'][0]
                neighbor = byregion[absorb]
                association = None
                if cfg.variant == 'v3' and cfg.enable_v3_reading_order and reading_graph['rank_count']:
                    association = graph_neighbor(neighbor, own, reading_graph, lines, confirmed)
                    h['reading_order'] = association
                r, failure = absorb_region(neighbor, own, lines, owners, confirmed,
                                           cfg, width, height, id, reading_association=association, semantic=semantic,
                                           expanded_during_preparation=absorb in reading_graph.get('expanded_region_ids', ()))
                if failure:
                    h['codes'].append(failure)
                    continue
            else:
                if ev['non_target_ink_ratio'] > cfg.max_foreign_ink:
                    h['codes'].append('FOREIGN_CONTAMINATION')
                    continue
                r = Proposal(id, list(bbox), label, 'mineru', b.get('raw_label', label), None,
                             [l.id for l in own], t['source_ids'] or [t['id']], 'mineru', why)
                if any(iou(r.bbox, p.bbox) >= cfg.nms_iou for p in confirmed):
                    h['codes'].append('DUPLICATE_CONFIRMED_REGION')
                    continue
            h.update(candidate_id=id, target_line_ids=[l.id for l in own], absorb_neighbor=absorb,
                     candidate_bbox=list(r.bbox))
            h['codes'].append('STEP2_VALID_CANDIDATE')
            pool.append({'id': id, 'region': r, 'target_ids': {l.id for l in own},
                         'absorb': absorb, 'history': h})
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            h.update(error=str(exc))
            h['codes'].append('INVALID_BOX')
    return pool, history


def best_partition(pool, keep, target_lines, snapshot, lines, cfg, width, height, semantic=None):
    target_ids = {l.id for l in target_lines}
    required = {lid for p in keep for lid in p.line_ids}
    byregion = {p.id: p for p in snapshot}
    scored = []
    for c in pool:
        score, _ = quality([c['region']], target_lines, lines, width, height, cfg, semantic=semantic)
        scored.append((c, score))
    ranked = sorted(scored, key=lambda v: (-len(v[0]['target_ids']), -v[1], v[0]['id']))
    for c, _ in ranked[cfg.partition_candidates:]:
        c['history']['codes'].append('PARTITION_CANDIDATE_LIMIT')
    pool = [c for c, _ in ranked[:cfg.partition_candidates]]
    cache = {}

    def evaluate(selection):
        key = tuple(c['id'] for c in selection)
        if key not in cache:
            absorb = next((c['absorb'] for c in selection if c['absorb']), None)
            base = keep+([byregion[absorb]] if absorb else [])
            ids = target_ids | {lid for p in base for lid in p.line_ids}
            evaluation_lines = [l for l in lines if l.id in ids]
            regs = [c['region'] for c in selection]
            q, features = quality(regs, evaluation_lines, lines, width, height, cfg, semantic=semantic)
            k, kfeatures = quality(base, evaluation_lines, lines, width, height, cfg, semantic=semantic)
            assigned = {lid for c in selection for lid in c['target_ids']}
            count = len(assigned & target_ids)/max(1, len(target_ids))
            complete = (required <= assigned and count >= (1. if keep else cfg.orphan_line_fraction))
            cache[key] = {'selection': selection, 'q': q, 'keep_q': k, 'features': features,
                          'keep_features': kfeatures, 'count_coverage': count, 'valid': complete,
                          'assigned': assigned, 'absorb': absorb, 'base': base}
        return cache[key]

    beam = [()]
    for c in pool:
        extended = []
        for s in beam:
            if any(c['target_ids'] & x['target_ids'] or iou(c['region'].bbox, x['region'].bbox) > cfg.partition_iou
                   or (c['absorb'] and x['absorb']) for x in s):
                continue
            # Only one edited neighbor may participate in an atomic partition.
            extended.append(s+(c,))
        beam += extended
        beam = sorted(beam, key=lambda s: (-evaluate(s)['valid'], -evaluate(s)['q'],
                                          -evaluate(s)['count_coverage'], tuple(c['id'] for c in s)))[:cfg.partition_beam]
    valid = [evaluate(s) for s in beam if s and evaluate(s)['valid']]
    # Rank feasible partitions by unchanged Q, not by acceptance count.
    return (max(valid, key=lambda v: (v['q'], v['count_coverage'], tuple(c['id'] for c in v['selection'])))
            if valid else None), {'candidate_count': len(pool), 'beam_size': len(beam),
                                 'best_count_coverage': max((evaluate(s)['count_coverage'] for s in beam), default=0.)}


def finish_layout(state, lines, predictions, cfg, width, height, semantic=None):
    snapshot = deepcopy(state['recovery'])  # ownership never changes to justify a box
    final = deepcopy(snapshot)
    reviews, decisions = deepcopy(state['review']), deepcopy(state['decisions'])
    audit, edited = [], set()
    if cfg.variant == 'v3' and cfg.enable_v3_reading_order:
        edited.update(d['reading_order']['neighbor_id'] for d in decisions if 'reading_order' in d)
    byline = {l.id: l for l in lines}
    for t in state['targets']:
        fallback = {p['id'] for p in t['fallback_baseline']}
        keep = [p for p in snapshot if p.id in fallback]
        response = predictions.get(t['id'], {'status': 'budget_exhausted', 'blocks': []})
        entry = dict(target=deepcopy(t), **deepcopy(response))
        entry['boxes'], entry['checkpoints'] = [], []
        audit.append(entry)
        target_lines = [byline[lid] for lid in t['line_ids']]
        confirmed = [p for p in snapshot if p.id not in fallback]
        owners = {lid: p.id for p in confirmed for lid in p.line_ids}
        entry['frozen_ownership'] = {'target_line_ids': list(t['line_ids']),
            'confirmed_line_owners': owners, 'confirmed_region_ids': [p.id for p in confirmed],
            'other_orphan_line_ids': [l.id for l in lines if l.id not in owners and l.id not in t['line_ids']]}
        if response.get('status') == 'ok':
            crop = response.get('crop_bbox')
            if (not isinstance(response.get('blocks'), list) or not isinstance(crop, (list, tuple))
                    or len(crop) != 4 or not all(isinstance(v, (float, int)) and isfinite(v) for v in crop)
                    or not (0 <= crop[0] < crop[2] <= width and 0 <= crop[1] < crop[3] <= height)):
                response = dict(response, status='invalid_output')
                entry.update(status='invalid_output', error='Malformed blocks or crop coordinate frame')
        record = decision('KEEP' if keep else 'REVIEW', t['source_ids'], 'keep_or_abstain',
                          target_id=t['id'], baseline=[serialize_proposal(p) for p in keep], candidates=[],
                          rejection_reasons=[])
        best = None
        if response.get('status') != 'ok':
            entry['checkpoints'].append('MODEL_'+response.get('status', 'invalid_output').upper())
        elif not target_lines:
            entry['checkpoints'].append('NO_TARGET_LINES')
        else:
            pool, history = candidate_boxes(t, response, snapshot, lines, cfg, width, height, edited,
                                            reading_graph=state.get('reading_graph'), semantic=semantic)
            entry['boxes'] = history
            if cfg.enable_geometry_refinement:
                entry['geometry_diagnostics'] = {
                    'raw_box_count': len(history), 'valid_candidate_count': len(pool),
                    'changed_box_count': sum(h.get('geometry_refinement', {}).get('changed', False)
                                             for h in history),
                    'refinement_rejected_count': sum('GEOMETRY_REFINEMENT_REJECTED' in h['codes']
                                                    for h in history),
                    'pre_refinement_bisection_count': sum(any(not (v['normal'] or v['snap'])
                        for v in h.get('pre_refinement_line_links', [])) for h in history)}
            touched = any('PURE_CONTEXT_BOX' not in h['codes'] for h in history)
            if not touched:
                entry['checkpoints'].append('STEP1_NO_BOX_TOUCHED_TARGET')
            elif not pool:
                entry['checkpoints'].append('NO_VALID_PARTITION')
            else:
                best, partition = best_partition(pool, keep, target_lines, snapshot, lines, cfg, width, height,
                                                 semantic=semantic)
                entry['partition_search'] = partition
                if best is None:
                    entry['checkpoints'] += ['STEP3_INCOMPLETE_TARGET_SET_COVERAGE', 'NO_VALID_PARTITION']
        if best:
            regs = [c['region'] for c in best['selection']]
            selected = {c['id'] for c in best['selection']}
            for h in entry['boxes']:
                h['selected_partition'] = h.get('candidate_id') in selected
            record.update(keep_score=best['keep_q'], replace_score=best['q'],
                          keep_features=best['keep_features'], replace_features=best['features'],
                          count_coverage=best['count_coverage'], candidates=[serialize_proposal(p) for p in regs])
            accepted = best['q'] > best['keep_q']+cfg.min_replace_gain
            if cfg.enable_geometry_refinement:
                entry['quality_diagnostics'] = {
                    'q_keep': best['keep_q'], 'q_new': best['q'],
                    'required_gain': cfg.min_replace_gain, 'partition_feasible': True,
                    'blocked_by_legacy_q': not accepted}
            changed = fallback | ({best['absorb']} if best['absorb'] else set())
            proposed = [p for p in final if p.id not in changed]+deepcopy(regs)
            # No global reassignment: a new box may never steal another owner's line.
            ids = [lid for p in proposed for lid in p.line_ids]
            old_ids = {lid for p in best['base'] for lid in p.line_ids}
            valid = (len(ids) == len(set(ids)) and old_ids <= set(ids)
                     and all(coverage(byline[lid], p.bbox) >= cfg.line_coverage for p in regs for lid in p.line_ids)
                     and not any(iou(p.bbox, q.bbox) >= cfg.nms_iou for p in regs
                                 for q in final if q.id not in changed))
            if accepted and valid:
                final = proposed
                if best['absorb']:
                    edited.add(best['absorb'])
                code = 'ACCEPTED_NEIGHBOR_MERGE' if best['absorb'] else 'STEP4_ACCEPTED_MINERU'
                record.update(action='ABSORB' if best['absorb'] else 'REPLACE' if keep else 'ADD', reason=code)
                entry['checkpoints'].append(code)
            else:
                entry['checkpoints'].append('ATOMIC_ROLLBACK' if not valid else 'STEP4_LOST_TO_KEEP_BASELINE')
        record['rejection_reasons'] = [c for c in entry['checkpoints'] if c not in ('STEP4_ACCEPTED_MINERU', 'ACCEPTED_NEIGHBOR_MERGE')]
        decisions.append(record)
        if record['action'] in ('KEEP', 'REVIEW'):
            reviews.append({'id': 'review:'+t['id'], 'bbox': t['bbox'], 'label': 'unknown',
                            'line_ids': t['line_ids'], 'source_ids': t['source_ids'],
                            'reason': 'replacement_rejected' if keep else 'orphan_abstained',
                            'codes': list(entry['checkpoints']), 'target_id': t['id']})
    owned = [lid for p in final for lid in p.line_ids]
    # Only unowned lines are unresolved. Detector proposal review is separate evidence.
    for i, group in enumerate(cluster([l for l in lines if l.id not in set(owned)])):
        reviews.append({'id': f'review:unresolved:{i}', 'bbox': envelope([line_box(l) for l in group]),
                        'label': 'unknown', 'source_ids': [], 'line_ids': [l.id for l in group],
                        'reason': 'unresolved_ocr_lines'})
    reviewed = {lid for r in reviews for lid in r.get('line_ids', [])}
    if len(owned) != len(set(owned)) or not set(byline) <= set(owned) | reviewed:
        raise ValueError('line accounting failed: neither detected nor reviewed')
    return sorted(final, key=lambda p: (p.bbox[1], p.bbox[0], p.id)), audit, reviews, decisions
