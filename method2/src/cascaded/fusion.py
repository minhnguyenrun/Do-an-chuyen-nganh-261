"""V2: asymmetric PP baseline + small YOLO supplements + audited KEEP/REPLACE."""
from copy import deepcopy
from math import isfinite
from statistics import median
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from .types import Proposal, label_for, serialize_proposal
from .core import (TEXT, VISUAL, area, intersect, iou, envelope, coverage, line_box, assign,
                   cluster, unresolved, context_box, to_page_bbox)
from .reading_order import build_reading_graph, cluster_orphans, graph_neighbor
from .semantic_gate import semantic_score, expansion_supported
from .v3_geometry import (measured_shape, proposal_intersection_area, proposal_line_coverage,
                          protected_overlap, set_proposal_bbox, v3_polygons_enabled)


def compatible(a, b):
    return a == b or ('title' in (a,b) and a in ('title','doc_title','paragraph_title')
                     and b in ('title','doc_title','paragraph_title'))


def review(region, reason, **extra):
    return {'id': 'review:'+region.id, 'bbox': list(region.bbox), 'label': region.label,
            'source_ids': list(region.source_ids), 'line_ids': list(region.line_ids),
            'reason': reason, **extra}


def decision(action, sources, reason, **extra):
    return {'action': action, 'source_ids': list(sources), 'reason': reason, **extra}


def small(p, cfg, width, height):
    return (area(p.bbox)/(width*height) <= cfg.yolo_max_area
            and (p.bbox[3]-p.bbox[1])/height <= cfg.yolo_max_height)


def overlaps_locked(p, locked, threshold):
    return any(iou(p.bbox,r.bbox) >= threshold or
               area(intersect(p.bbox,r.bbox))/max(1,area(p.bbox)) >= .8 for r in locked)


def add_review_lines(regions, lines, reviews, threshold):
    for i, group in enumerate(cluster(unresolved(regions, lines))):
        ids = [l.id for l in group]
        reviews.append({'id': f'review:orphan:{i}', 'bbox': envelope([line_box(l) for l in group]),
                        'label': 'unknown', 'source_ids': [], 'line_ids': ids,
                        'reason': 'unresolved_ocr_lines'})


def learned_evidence(semantic, label, owned, lines, bbox, width, height):
    if semantic is None:
        return None, {'available': False, 'abstain_reason': 'resolver_unavailable',
                      'calibration_status': 'unavailable', 'probabilities': {}, 'supported_classes': [],
                      'model_fingerprint': None, 'feature_fingerprint': None, 'cpu_seconds': 0.}
    evidence = semantic.predict(label, owned, lines, bbox, width, height)
    score = evidence['probabilities'].get(label)
    if (not evidence['available'] or evidence['abstain_reason'] is not None
            or evidence['calibration_status'] != 'calibrated' or label not in evidence['supported_classes']
            or score is None or not isfinite(score) or not 0 <= score <= 1):
        return None, evidence
    return score, evidence


def scored_semantics(label, owned, lines, bbox, width, height, *, enabled, semantic,
                     abstain_blocks=False):
    evidence = None
    if enabled:
        probability, evidence = learned_evidence(semantic, label, owned, lines, bbox, width, height)
        if probability is not None:
            return probability, ['calibrated_semantic_probability'], evidence
        if abstain_blocks and evidence['available']:
            return None, [evidence['abstain_reason'] or 'semantic_probability_unavailable'], evidence
    score, reasons = semantic_score(label, owned, lines, bbox, width, height)
    return score, reasons, evidence


def passes_recovery_semantics(score, evidence, cfg):
    required = cfg.min_semantic_score
    if evidence is not None and evidence['available']:
        required = max(.60, required)
    return score is not None and score >= required


def arbitrate_conflict(pp, yolo, regions, lines, cfg, width, height, semantic, detector_coverage):
    """Evaluate one frozen PP/YOLO conflict. Any unsafe geometry keeps MinerU routing."""
    if pp.stage != 'strong' or pp.label not in TEXT or yolo.label not in TEXT or not pp.line_ids:
        return None
    if any(not (0 <= p.bbox[0] < p.bbox[2] <= width and 0 <= p.bbox[1] < p.bbox[3] <= height)
           for p in (pp, yolo)):
        return None
    owned_ids = set(pp.line_ids)
    owned = [line for line in lines if line.id in owned_ids]
    if (len(cluster(owned)) != 1
            or any(not line.text.strip() or line.rec_score is None or line.rec_score < .75 for line in owned)):
        return None
    for source in (pp, yolo):
        if any(detector_coverage(line, source) < cfg.line_coverage if line.id in owned_ids
               else detector_coverage(line, source) > 1e-8 for line in lines):
            return None
    use_v3_polygon = v3_polygons_enabled(cfg)
    if any(other.id != pp.id and (iou(pp.bbox, other.bbox) >= cfg.nms_iou
            or (other.label in VISUAL and any(protected_overlap(
                source.bbox, other, use_v3_polygon=use_v3_polygon) > cfg.max_nontext_overlap
                for source in (pp, yolo)))) for other in regions):
        return None
    pp_p, pp_evidence = learned_evidence(semantic, pp.label, owned, lines, pp.bbox, width, height)
    # Score both proposed labels on the PP geometry that would actually be committed.
    yolo_p, yolo_evidence = learned_evidence(semantic, yolo.label, owned, lines, pp.bbox, width, height)
    details = {'pp': pp_evidence, 'yolo': yolo_evidence, 'committed': False}
    if pp_p is None or yolo_p is None:
        return details
    pp_joint = .55 * pp.score + .45 * pp_p
    yolo_joint = .55 * yolo.score + .45 * yolo_p
    margin = abs(pp_joint - yolo_joint)
    details.update(pp_joint=pp_joint, yolo_joint=yolo_joint, margin=margin)
    if max(pp_joint, yolo_joint) >= .65 and margin >= .15:
        details.update(committed=True, winner=pp.label if pp_joint > yolo_joint else yolo.label)
    return details


def prepare_layout(proposals, lines, cfg, width, height, semantic=None):
    proposals = deepcopy(proposals)
    for p in proposals:
        p.label = label_for(p.raw_label, p.source)
    graph = build_reading_graph(proposals, lines, cfg)
    use_v3_polygon = v3_polygons_enabled(cfg)

    def detector_coverage(line, proposal):
        return proposal_line_coverage(line, proposal, use_v3_polygon=use_v3_polygon)

    proposals = [p for p in proposals if p.score is not None and p.score >= cfg.low]
    regions = [deepcopy(p) for p in proposals if p.source == 'paddle' and p.score >= cfg.high and p.label!='unknown']
    for p in regions:
        p.stage = 'strong'
    assign(regions, lines, cfg.line_coverage, use_v3_polygon=use_v3_polygon)
    reviews, decisions, conflicts, candidates = [], [], [], []
    reviews += [review(p,'unclassified_pp_proposal') for p in proposals if p.source=='paddle' and p.label=='unknown']
    for q in sorted((p for p in proposals if p.source == 'yolo'), key=lambda p:(-p.score,p.id)):
        if not cfg.enable_yolo or not small(q,cfg,width,height):
            reviews.append(review(q,'yolo_disabled' if not cfg.enable_yolo else 'outside_small_box_role'))
            continue
        matches = [r for r in regions if iou(q.bbox,r.bbox) >= cfg.nms_iou]
        nested = [r for r in regions if r not in matches and area(intersect(q.bbox,r.bbox))/max(1,area(q.bbox)) >= .8]
        if matches and all(compatible(q.label,r.label) for r in matches):
            for r in matches:
                r.source_ids = list(dict.fromkeys(r.source_ids+q.source_ids))
            action = 'KEEP'
            reason = 'compatible_labels_keep_pp_geometry'
            # A generic title supports the PP subtype but cannot determine it.
            if cfg.enable_wbf and len(matches) == 1 and matches[0].source == 'paddle' and matches[0].label == q.label and q.score >= .25:
                r = matches[0]
                bbox = [(r.score*a+q.score*b)/(r.score+q.score) for a,b in zip(r.bbox,q.bbox)]
                others = [p for p in regions if p.id != r.id]
                clears_mask = (tuple(bbox) != tuple(r.bbox)
                               and measured_shape(r, use_v3_polygon=use_v3_polygon) is not None)
                if (all(coverage(l,bbox) >= cfg.line_coverage for l in lines if l.id in r.line_ids)
                        and not (clears_mask and any(coverage(l,bbox) >= cfg.line_coverage
                                                    for l in lines if l.id not in r.line_ids))
                        and not any(iou(bbox,p.bbox) >= cfg.protected_iou for p in others)):
                    set_proposal_bbox(r, bbox)
                    r.stage = 'wbf'
                    action, reason = 'WBF', 'same_label_one_to_one_line_preserving'
                else:
                    reason = 'wbf_rejected_geometry_keep_pp'
            decisions.append(decision(action,[q.id]+[r.id for r in matches],reason))
            continue
        if nested:
            reviews.append(review(q,'nested_in_pp_block',parent_ids=[r.id for r in nested]))
            decisions.append(decision('KEEP',[q.id]+[r.id for r in nested],'parent_child_not_duplicate_detection'))
            continue
        if matches:
            fallback = [r for r in matches if r.source == 'paddle']
            if not fallback:
                reviews.append(review(q,'overlap_with_accepted_supplement'))
                continue
            pp_score = max(r.score for r in fallback)
            if q.score < .25 or (pp_score-q.score >= .35):
                reviews.append(review(q,'weak_yolo_cannot_veto_pp'))
                decisions.append(decision('KEEP',[r.id for r in fallback]+[q.id],'pp_dominance'))
            elif q.score >= .25 and pp_score >= .30 and abs(pp_score-q.score) < .35:
                conflicts.append((fallback,q))
            else:
                # A higher score from a different detector alone cannot replace PP.
                reviews.append(review(q,'uncalibrated_score_not_replacement_evidence'))
                decisions.append(decision('KEEP',[r.id for r in fallback]+[q.id],'retain_pp_without_semantic_proof'))
            continue
        candidates.append(q)
    for q in candidates:
        supported = [l for l in unresolved(regions,lines) if coverage(l,q.bbox) >= cfg.line_coverage]
        score, why, evidence = scored_semantics(
            q.label, supported, lines, q.bbox, width, height, enabled=cfg.enable_semantic_arbitration,
            semantic=semantic, abstain_blocks=True)
        details = {'semantic_prediction': evidence} if evidence is not None else {}
        # Proposals competing for the same lines are sent for review, not added by order.
        contested = any(p.id != q.id and not compatible(p.label,q.label) and
                        any(coverage(l,p.bbox) >= cfg.line_coverage for l in supported) for p in candidates)
        if (q.score >= cfg.high and supported and passes_recovery_semantics(score, evidence, cfg)
                and not contested and not overlaps_locked(q,regions,cfg.protected_iou)):
            q = deepcopy(q)
            q.line_ids, q.stage, q.reasons = [l.id for l in supported], 'yolo_small', why
            regions.append(q)
            decisions.append(decision('ADD',q.source_ids,'small_yolo_ocr_verified',semantic_score=score,
                                      features=why,**details))
        elif q.score >= cfg.high:
            reviews.append(review(q,'small_yolo_not_verified',semantic_score=score,features=why,**details))
    assign(regions, lines, cfg.line_coverage, use_v3_polygon=use_v3_polygon)
    ensemble = deepcopy(regions)
    conflicted_ids = {r.id for rs,_ in conflicts for r in rs}
    preserved_owners = {}
    if cfg.enable_recovery:
        weak_images = [p for p in proposals if p.source == 'paddle' and p.label == 'image'
                       and p.score < cfg.high]
        weak = [p for p in proposals if p.source == 'paddle' and p.label != 'image'
                and p.score < cfg.high]
        weak += [p for p in candidates if p.score < cfg.high]
        for q in sorted(weak,key=lambda p:(-p.score,p.source!='paddle',p.id)):
            owned = [l for l in unresolved(regions,lines) if detector_coverage(l,q) >= cfg.line_coverage]
            score, why, evidence = scored_semantics(
                q.label, owned, lines, q.bbox, width, height, enabled=cfg.enable_semantic_arbitration,
                semantic=semantic, abstain_blocks=True)
            details = {'semantic_prediction': evidence} if evidence is not None else {}
            contested = any(p.id != q.id and not compatible(p.label,q.label) and
                            any(detector_coverage(l,p) >= cfg.line_coverage for l in owned) for p in weak)
            if (not owned or not passes_recovery_semantics(score, evidence, cfg) or contested
                    or overlaps_locked(q,regions,cfg.protected_iou)):
                reviews.append(review(q,'weak_recovery_rejected',semantic_score=score,features=why,**details))
                continue
            q = deepcopy(q)
            q.line_ids, q.stage, q.reasons = [l.id for l in owned], 'recovery', why
            regions.append(q)
            decisions.append(decision('ADD',q.source_ids,'weak_box_semantic_recovery',semantic_score=score,
                                      features=why,**details))
        # Expansion is a proposal with rollback: never modify a locked neighbour.
        confirmed = [r for r in regions if r.id not in conflicted_ids]
        expanded_ids, graph_edited_ids = set(), set()
        boundary_attempts = []
        for l in list(unresolved(regions,lines)):
            b = line_box(l)
            owners = []
            association = None
            for r in regions:
                owned = [x for x in lines if x.id in r.line_ids]
                if (r.id in conflicted_ids or r.id in graph_edited_ids or r.label not in ('text','list_item')
                        or not expansion_supported(owned,l,lines)):
                    continue
                h = median(line_box(x)[3]-line_box(x)[1] for x in owned)
                overlap = max(0,min(b[2],r.bbox[2])-max(b[0],r.bbox[0]))/max(1,b[2]-b[0])
                gap = max(b[1]-r.bbox[3],r.bbox[1]-b[3],0)
                if overlap < .8 or gap > h:
                    continue
                expanded = envelope([r.bbox,b])
                blocked = any(p.id != r.id and proposal_intersection_area(
                    expanded,p,use_v3_polygon=use_v3_polygon and p.label in VISUAL) > proposal_intersection_area(
                    r.bbox,p,use_v3_polygon=use_v3_polygon and p.label in VISUAL) for p in regions)
                if blocked:
                    continue
                allowed_ids = set(r.line_ids) | {l.id}
                if measured_shape(r, use_v3_polygon=use_v3_polygon) is not None:
                    if (tuple(expanded) == tuple(r.bbox)
                            or any(coverage(x,expanded) < cfg.line_coverage for x in owned)
                            or any(coverage(x,expanded) >= cfg.line_coverage for x in lines if x.id not in allowed_ids)):
                        continue
                evidence = None
                if abs(b[0]-r.bbox[0]) > h:
                    if not graph['rank_count'] or r.id in expanded_ids or b[0] <= r.bbox[0]:
                        continue
                    if association is None:
                        association = graph_neighbor(None, [l], graph, lines, confirmed)
                        boundary_attempts.append({'line_id': l.id, 'association': association})
                    if not (association['supported'] and association['after_id'] == r.id == association['neighbor_id']):
                        continue
                    if (any(coverage(x,expanded) < cfg.line_coverage for x in owned)
                            or any(coverage(x,expanded) > 1e-8 for x in lines if x.id not in allowed_ids)
                            or semantic_score(r.label,owned+[l],lines,expanded,width,height)[0] < cfg.min_semantic_score):
                        continue
                    evidence = association
                owners.append((r, expanded, evidence))
            if len(owners)==1:
                r, expanded, evidence = owners[0]
                before = list(r.bbox)
                set_proposal_bbox(r, expanded)
                r.stage = 'recovery'
                r.line_ids.append(l.id)
                expanded_ids.add(r.id)
                details = {}
                if evidence is not None:
                    graph_edited_ids.add(r.id)
                    preserved_owners.update((line_id, r.id) for line_id in r.line_ids)
                    details['reading_order'] = evidence
                decisions.append(decision('ADD',r.source_ids,'line_preserving_boundary_expansion',
                                          before=before,after=r.bbox,line_ids=[l.id],**details))
        if boundary_attempts:
            graph['boundary_attempts'] = boundary_attempts
        if graph['edges'] and expanded_ids:
            graph['expanded_region_ids'] = sorted(expanded_ids)
        for q in sorted(weak_images, key=lambda p: (-p.score, p.id)):
            if any(area(intersect(q.bbox, r.bbox)) > 0 for r in regions):
                reviews.append(review(q, 'weak_image_overlaps_region'))
                continue
            # Keep existing OCR ownership when a line spans disjoint regions.
            at_risk = {line.id for line in lines if detector_coverage(line, q) >= cfg.line_coverage}
            for r in regions:
                for line_id in r.line_ids:
                    if line_id in at_risk:
                        preserved_owners.setdefault(line_id, r.id)
            q = deepcopy(q)
            q.line_ids, q.stage, q.reasons = [], 'recovery', ['weak_image_no_region_overlap']
            regions.append(q)
            decisions.append(decision('ADD', q.source_ids, 'weak_image_no_region_overlap'))
    assign(regions, lines, cfg.line_coverage, use_v3_polygon=use_v3_polygon,
           preserved_owners=preserved_owners)
    targets = []
    by_id = {p.id: p for p in regions}
    if cfg.enable_semantic_arbitration:
        conflict_counts = {}
        for fallback, _ in conflicts:
            for r in fallback:
                conflict_counts[r.id] = conflict_counts.get(r.id, 0) + 1
        pending = []
        for fallback, q in conflicts:
            if len(fallback) != 1 or conflict_counts[fallback[0].id] != 1:
                pending.append((fallback, q))
                continue
            r = by_id[fallback[0].id]
            evidence = arbitrate_conflict(r, q, regions, lines, cfg, width, height, semantic, detector_coverage)
            if evidence is None:
                pending.append((fallback, q))
                continue
            committed = evidence['committed']
            action = 'RELABEL' if committed and evidence['winner'] != r.label else 'KEEP' if committed else 'REVIEW'
            decisions.append(decision(action, list(dict.fromkeys(r.source_ids + q.source_ids)),
                                      'semantic_conflict_committed' if committed else 'semantic_conflict_abstained',
                                      previous_label=r.label, line_ids=list(r.line_ids), semantic_arbitration=evidence))
            if committed:
                r.label = evidence['winner']
                r.source_ids = list(dict.fromkeys(r.source_ids + q.source_ids))
                conflicted_ids.discard(r.id)
            else:
                pending.append((fallback, q))
        conflicts = pending
    # Merge overlapping conflict components, keeping every PP fallback (not just b_high).
    for fallback,q in conflicts:
        ids = {r.id for r in fallback}
        joined = [t for t in targets if ids.intersection(p['id'] for p in t['fallback_baseline'])]
        baseline = {r.id:serialize_proposal(by_id[r.id]) for r in fallback}
        source_ids = list(q.source_ids)
        for t in joined:
            baseline.update({p['id']:p for p in t['fallback_baseline']})
            source_ids += t['source_ids']
            targets.remove(t)
        bbox = envelope([p['bbox'] for p in baseline.values()]+[q.bbox])
        line_ids = [l.id for l in lines if any(detector_coverage(l,by_id[id])>=cfg.line_coverage for id in baseline)]
        targets.append({'bbox':bbox,'line_ids':line_ids,'source_ids':list(dict.fromkeys(list(baseline)+source_ids)),
                        'fallback_baseline':list(baseline.values()),'reason':'true_conflict'})
    assigned = {id for t in targets for id in t['line_ids']}
    orphans = [l for l in unresolved(regions,lines) if l.id not in assigned]
    confirmed = [r for r in regions if r.id not in conflicted_ids]
    for group in cluster_orphans(orphans, graph, lines, confirmed):
        targets.append({'bbox':envelope([line_box(l) for l in group]),'line_ids':[l.id for l in group],
                        'source_ids':[],'fallback_baseline':[],'reason':'orphan'})
    for i,t in enumerate(targets):
        t['id'] = f'target:{i}'
    state = {'ensemble':ensemble,'recovery':deepcopy(regions),'targets':targets,'review':reviews,'decisions':decisions}
    if graph['rank_count']:
        state['reading_graph'] = graph
    return state


def quality(regions, target_lines, all_lines, width, height, cfg, semantic=None):
    union = unary_union([box(*r.bbox) for r in regions])
    total = sum(Polygon(l.polygon).area for l in target_lines)
    covered = cut = 0.
    for l in target_lines:
        shape = Polygon(l.polygon)
        ratio = shape.intersection(union).area/max(1,shape.area)
        if ratio >= cfg.line_coverage:
            covered += shape.area
        if any(.20 < coverage(l,r.bbox) < .70 for r in regions):
            cut += shape.area
    lexical, count, features = 0.,0,[]
    for r in regions:
        owned = [l for l in target_lines if coverage(l,r.bbox)>=cfg.line_coverage]
        score, reasons, evidence = scored_semantics(
            r.label, owned, all_lines, r.bbox, width, height, enabled=cfg.enable_semantic_quality,
            semantic=semantic)
        lexical += score*len(owned)
        count += len(owned)
        diagnostic = {'region_id':r.id,'semantic_score':score,'reasons':reasons}
        if evidence is not None:
            diagnostic['semantic_prediction'] = evidence
        features.append(diagnostic)
    terms = {'coverage':covered/max(1,total),'split_penalty':cut/max(1,total),'semantic':lexical/max(1,count)}
    return .45*terms['coverage']-.30*terms['split_penalty']+.25*terms['semantic'], dict(terms,features=features)


def finish_layout(state, lines, predictions, cfg, width, height, semantic=None):
    final = deepcopy(state['recovery'])
    reviews, decisions = deepcopy(state['review']),deepcopy(state['decisions'])
    audit = []
    byline = {l.id:l for l in lines}
    for t in state['targets']:
        fallback_ids = {r['id'] for r in t['fallback_baseline']}
        keep = [r for r in final if r.id in fallback_ids]
        locked = [r for r in final if r.id not in fallback_ids]
        target_lines = [byline[id] for id in t['line_ids']]
        response = predictions.get(t['id'], {'status':'budget_exhausted','blocks':[]})
        entry = dict(target=t,**response)
        audit.append(entry)
        keep_score,keep_features = quality(keep,target_lines,lines,width,height,cfg,semantic=semantic)
        replacements, rejection = [],[]
        if response['status']=='ok':
            try:
                raw_boxes = [to_page_bbox(b['bbox'],response['crop_bbox']) for b in response['blocks']]
                for i,(b,bbox) in enumerate(zip(response['blocks'],raw_boxes)):
                    if area(intersect(bbox,t['bbox']))/max(1,area(bbox)) < .8:
                        continue  # context-only block; don't clip a whole block into a fragment
                    own = [l for l in target_lines if coverage(l,bbox)>=cfg.line_coverage]
                    if not own:
                        continue
                    if len(cluster(own)) != 1:
                        rejection.append('cross_column_or_disconnected_lines')
                    label = label_for(b.get('raw_label',b['label']),'mineru')
                    score,why,_ = scored_semantics(
                        label, own, lines, bbox, width, height, enabled=cfg.enable_semantic_quality, semantic=semantic)
                    if score < cfg.min_semantic_score:
                        rejection.append('semantic_gate_failed')
                    r = Proposal(f"mineru:{t['id']}:{i}",bbox,label,'mineru',b.get('raw_label',label),None,
                                 [l.id for l in own],t['source_ids'] or [t['id']],'mineru',why)
                    if overlaps_locked(r,locked,cfg.protected_iou):
                        rejection.append('overlaps_locked_region')
                    replacements.append(r)
            except (ValueError,TypeError,KeyError) as exc:
                entry['status'],entry['error'] = 'invalid_output',str(exc)
                rejection.append('invalid_output')
        elif keep:
            rejection.append('model_unavailable_keep_baseline')
        else:
            rejection.append('model_unavailable_review_orphan')
        assigned = [id for r in replacements for id in r.line_ids]
        if len(assigned)!=len(set(assigned)):
            rejection.append('duplicate_lines')
        if set(assigned)!=set(t['line_ids']):
            rejection.append('missing_target_lines')
        if any(iou(a.bbox,b.bbox)>=cfg.protected_iou for i,a in enumerate(replacements) for b in replacements[i+1:]):
            rejection.append('overlapping_replacements')
        replace_score,replace_features = quality(replacements,target_lines,lines,width,height,cfg,semantic=semantic)
        if replace_score <= keep_score+cfg.min_replace_gain:
            rejection.append('insufficient_quality_gain')
        accepted = (bool(replacements) and not rejection and replace_score > keep_score+cfg.min_replace_gain)
        record = decision('REPLACE' if accepted and keep else 'ADD' if accepted else 'KEEP' if keep else 'REVIEW',
                          t['source_ids'],'quality_gain_accepted' if accepted else 'keep_or_abstain',
                          target_id=t['id'],keep_score=keep_score,replace_score=replace_score,
                          keep_features=keep_features,replace_features=replace_features,
                          rejection_reasons=list(dict.fromkeys(rejection)),
                          baseline=[serialize_proposal(r) for r in keep],
                          candidates=[serialize_proposal(r) for r in replacements])
        decisions.append(record)
        if accepted:
            final = locked+replacements
        else:
            reviews.append({'id':'review:'+t['id'],'bbox':t['bbox'],'label':'unknown',
                            'line_ids':t['line_ids'],'source_ids':t['source_ids'],
                            'reason':'replacement_rejected' if keep else 'orphan_abstained',
                            'target_id':t['id']})
    weak_images = [r for r in final if r.source == 'paddle' and r.label == 'image'
                   and 'weak_image_no_region_overlap' in r.reasons]
    at_risk = {line.id for line in lines if any(coverage(line, r.bbox) >= cfg.line_coverage
                                                for r in weak_images)}
    preserved_owners = {line_id: r.id for r in final if r not in weak_images
                        for line_id in r.line_ids if line_id in at_risk}
    assign(final,lines,cfg.line_coverage,preserved_owners=preserved_owners)
    add_review_lines(final,lines,reviews,cfg.line_coverage)
    owned = [id for r in final for id in r.line_ids]
    reviewed = {id for r in reviews for id in r.get('line_ids',[])}
    if len(owned)!=len(set(owned)) or not set(byline)<=set(owned)|reviewed:
        raise ValueError('line accounting failed: neither detected nor reviewed')
    return sorted(final,key=lambda r:(r.bbox[1],r.bbox[0],r.id)),audit,reviews,decisions
