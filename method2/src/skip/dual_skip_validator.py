import re
from itertools import combinations
from shapely.geometry import Polygon, box
from ..semantic.evidence_features import (extract_formula_features, extract_table_features,
    table_alignment, calculate_content_consistency, parse_table,formula_content_consistency)
from ..semantic.residual_labeling import residual_role_evidence


def owned_lines(hyp,skip_h,skip_l):
    graph = skip_h.graph
    ids = list(dict.fromkeys(lid for rid in hyp.source_region_ids for lid in graph.ownership.get(rid,())))
    by_id = {line.line_id: line for line in skip_l.lines}
    return [by_id[lid] for lid in ids if lid in by_id]


def validate_hard_constraints(hyp,skip_h,skip_l,evidence,*,use_skip_h=True,use_skip_l=True):
    flags = []
    graph = skip_h.graph
    if not set(hyp.source_region_ids) <= set(graph.nodes):
        return False,['unknown_source_region']
    if hyp.action == 'KEEP':
        return True,[]
    if evidence is None or evidence.status != 'ok' or not evidence.content.strip():
        return False,['missing_or_failed_probe']
    if evidence.hypothesis_id != hyp.hypothesis_id:
        flags.append('evidence_id_mismatch')
    if hyp.action == 'MERGE':
        if len(hyp.source_region_ids) < 2:
            flags.append('merge_requires_multiple_sources')
        if any(graph.is_cross_column(a,b) for a,b in combinations(hyp.source_region_ids,2)):
            flags.append('cross_column_violation')
        if any(graph.nodes[id].label in ('figure','image','chart') for id in hyp.source_region_ids):
            flags.append('protected_visual_region')
        remaining = set(hyp.source_region_ids) if use_skip_h else set()
        if not remaining:
            remaining = {hyp.source_region_ids[0]}
        reached = {remaining.pop()}
        while remaining:
            adjacent = {id for id in remaining if any(tuple(sorted((id,r))) in graph.neighbors for r in reached)}
            if not adjacent:
                flags.append('disconnected_merge')
                break
            reached.update(adjacent)
            remaining -= adjacent
    x1,y1,x2,y2 = hyp.candidate_bbox
    if x1 < 0 or y1 < 0 or x2 > graph.page_w or y2 > graph.page_h:
        flags.append('out_of_page')
    lines = owned_lines(hyp,skip_h,skip_l)
    for line in lines if use_skip_l else ():
        poly = Polygon(line.polygon)
        if not poly.is_valid or poly.area <= 0 or poly.intersection(box(*hyp.candidate_bbox)).area/poly.area < .98:
            flags.append('missing_line_coverage')
            break
    if not lines:
        flags.append('missing_spatial_evidence')
    transcripts = [l.transcript for l in lines if l.transcript.strip()]
    if len(transcripts) != len(lines):
        flags.append('independent_transcripts_unavailable')
    if 'glm_residual_classification' in hyp.geometry_reasons:
        if evidence.classification is None or evidence.classification.label != hyp.candidate_label:
            flags.append('missing_glm_classification')
        else:
            supported, reasons = residual_role_evidence(hyp, graph, lines)
            if not supported:
                flags.extend(reasons)
            if calculate_content_consistency(evidence.classification.text, transcripts) < .8:
                flags.append('classification_transcript_mismatch')
    if hyp.action == 'SPLIT':
        parts = hyp.split_parts
        if len(parts) < 2 or len(evidence.part_contents) != len(parts):
            flags.append('ungrounded_split')
        assigned = [id for part in parts for id in part.line_ids]
        if len(assigned) != len(set(assigned)) or set(assigned) != {l.line_id for l in lines}:
            flags.append('split_missing_or_duplicate_lines')
        for a,b in combinations(parts,2):
            if box(*a.bbox).intersection(box(*b.bbox)).area > 0:
                flags.append('overlapping_split_children')
        for i,part in enumerate(parts):
            relevant = [l for l in lines if l.line_id in part.line_ids]
            if not box(*hyp.candidate_bbox).covers(box(*part.bbox)) or any(not box(*part.bbox).covers(Polygon(l.polygon)) for l in relevant):
                flags.append('split_geometry_mismatch')
            if i < len(evidence.part_contents) and relevant:
                if calculate_content_consistency(evidence.part_contents[i],[l.transcript for l in relevant]) < .8:
                    flags.append('split_content_loss')
        # Validate the gap from measured boxes, not a self-declared reason string.
        sorted_parts = sorted(parts,key=lambda p:p.bbox[1])
        if len(sorted_parts) >= 2 and any(b.bbox[1]-a.bbox[3] < 10 for a,b in zip(sorted_parts,sorted_parts[1:])):
            flags.append('split_no_geometric_gap')
    if hyp.candidate_label == 'table':
        features = extract_table_features(evidence.content,len(lines))
        if not features['is_valid_html']:
            flags.append('invalid_table_html')
        if use_skip_l:
            alignment = table_alignment(evidence.content,lines)
        else:
            # Skip-H-only ablation derives its grid from macro detections.
            from ..schemas.region import TextLineEvidence
            macro = []
            for id in hyp.source_region_ids:
                a,b,c,d = graph.nodes[id].bbox
                macro.append(TextLineEvidence(line_id=id,polygon=((a,b),(c,b),(c,d),(a,d))))
            alignment = table_alignment(evidence.content,macro)
        if not alignment:
            flags.append('table_geometry_mismatch')
    if hyp.candidate_label == 'formula':
        features = extract_formula_features(evidence.content)
        if not features['is_balanced'] or not features['has_math_tokens']:
            flags.append('invalid_formula')
        independent = ' '.join(transcripts)
        if len(re.findall(r'[=+^_∑∫√]|\\(?:frac|sum|int)',independent)) < 2 or len(lines) > 3:
            flags.append('formula_independent_evidence_missing')
        if formula_content_consistency(evidence.content,transcripts) < .8:
            flags.append('formula_content_mismatch')
    elif hyp.action == 'RELABEL' and hyp.candidate_label != 'table':
        if 'glm_residual_classification' not in hyp.geometry_reasons or evidence.classification is None:
            flags.append('unsupported_relabel_evidence')
    # Tables additionally match cells to their corresponding geometric line, not just counts.
    if hyp.candidate_label == 'table' and not flags:
        ordered = sorted(lines,key=lambda l:(min(p[1] for p in l.polygon),min(p[0] for p in l.polygon)))
        cells = [cell for row in parse_table(evidence.content).rows for cell in row]
        if len(cells) != len(ordered) or any(calculate_content_consistency(cell,[l.transcript]) < .8 for cell,l in zip(cells,ordered)):
            flags.append('table_cell_content_mismatch')
    elif hyp.candidate_label != 'formula' and transcripts and calculate_content_consistency(evidence.content,transcripts) < .8:
        flags.append('content_loss_or_hallucination')
    if use_skip_h and hyp.action in ('MERGE','REFINE_BBOX'):
        for id in hyp.source_region_ids:
            if not box(*hyp.candidate_bbox).covers(box(*graph.nodes[id].bbox)):
                flags.append('source_geometry_loss')
    return not flags,list(dict.fromkeys(flags))
