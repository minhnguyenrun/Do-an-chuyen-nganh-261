from shapely.geometry import Polygon,box
from .evidence_features import calculate_content_consistency, table_alignment, extract_formula_features, spatial_grid, formula_content_consistency
from ..skip.dual_skip_validator import owned_lines
from ..spatial.coordinates import area,union_bboxes
from .residual_labeling import residual_role_evidence


def hypothesis_features(hyp,evidence,skip_h,skip_l):
    lines = owned_lines(hyp,skip_h,skip_l)
    coverage = sum(box(*hyp.candidate_bbox).intersection(Polygon(l.polygon)).area/max(1,Polygon(l.polygon).area) for l in lines)/max(1,len(lines))
    content = evidence.content if evidence else ''
    consistency = calculate_content_consistency(content,[l.transcript for l in lines])
    source_boxes = [r.bbox for r in skip_h.regions if r.id in hyp.source_region_ids]
    macro = min(1.,sum(area(b) for b in source_boxes)/max(1,area(union_bboxes(source_boxes))))
    geometry = (coverage+macro)/2
    structure = .5
    recognition = .6 if content.strip() else 0.
    if evidence and evidence.classification and 'glm_residual_classification' in hyp.geometry_reasons:
        supported, _ = residual_role_evidence(hyp, skip_h.graph, lines)
        structure = float(supported)
        recognition = calculate_content_consistency(evidence.classification.text, [l.transcript for l in lines])
    if hyp.candidate_label == 'table':
        structure = table_alignment(content,lines)
        recognition = structure
    elif hyp.candidate_label == 'formula':
        f = extract_formula_features(content)
        structure = f['is_balanced']*f['has_math_tokens']
        recognition = structure
        # Different LaTeX spelling is not measured as plain text CER.
        consistency = formula_content_consistency(content,[l.transcript for l in lines])
    elif hyp.action == 'SPLIT':
        structure = 1.0 if len(hyp.split_parts) >= 2 else 0.
        recognition = float(bool(evidence and len(evidence.part_contents) == len(hyp.split_parts)))
    elif hyp.action == 'KEEP' and spatial_grid(lines)[2]:
        structure = .2 # Plain text hypothesis does not explain a measured table grid.
    return dict(geometry=geometry,recognition=recognition,consistency=consistency,structure=structure,
                loss=1-coverage,complexity=0 if hyp.action == 'KEEP' else .5,macro=macro,micro=coverage)


def calculate_hypothesis_score(hyp,evidence,feats,skip_h,skip_l,cfg):
    if evidence is None or evidence.status != 'ok' or not evidence.content.strip():
        return 0.
    defaults = dict(geometry=.25,recognition=.30,consistency=.20,structure=.25,loss=.25,complexity=.06)
    weights = defaults | cfg.get('weights',{})
    features = hypothesis_features(hyp,evidence,skip_h,skip_l) | feats
    if not cfg.get('use_skip_l',True):
        features['geometry'] = features['macro']
        features['loss'] = 0.
    if not cfg.get('use_skip_h',True):
        features['geometry'] = features['micro']
    value = sum(weights[k]*features[k] for k in ('geometry','recognition','consistency','structure'))
    value -= weights['loss']*features['loss']+weights['complexity']*features['complexity']
    return max(0.,min(1.,value))
