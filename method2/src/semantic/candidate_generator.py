import re
from statistics import median
from ..schemas.hypothesis import RegionHypothesis, SplitPart
from ..spatial.coordinates import union_bboxes, polygon_bbox
from .evidence_features import spatial_grid
from .residual_prompt import UNLABELED_LABELS


def generate_hypotheses(graph, max_hypotheses_per_group=3, max_candidate_group_size=8, enabled_actions=None,
                        residual_classifications=None):
    limit = min(3,max_hypotheses_per_group)
    if limit < 1 or max_candidate_group_size < 1:
        raise ValueError('hypothesis limits must be positive')
    max_group = min(8,max_candidate_group_size)
    enabled = set(['KEEP','MERGE','SPLIT','RELABEL','REFINE_BBOX'] if enabled_actions is None else enabled_actions)
    result = []
    residual_classifications = residual_classifications or {}
    text_like_labels = {'text'} | UNLABELED_LABELS
    def add_group(ids, alternatives, force_keep=False):
        alternatives = [a for a in alternatives if a['action'] in enabled]
        unique = []
        for alternative in alternatives:
            key = (alternative['action'], alternative.get('candidate_label'), alternative.get('candidate_bbox'),
                   repr(alternative.get('split_parts')))
            if not any(key == existing_key for existing_key, _ in unique):
                unique.append((key, alternative))
        alternatives = [value for _, value in unique]
        if not alternatives and not force_keep:
            return
        group_id = 'group:' + ':'.join(ids)
        rs = [graph.nodes[id] for id in ids]
        lines = tuple(lid for id in ids for lid in graph.ownership.get(id,()))
        base = dict(group_id=group_id,source_region_ids=ids,source_line_ids=lines,
                    candidate_bbox=union_bboxes(r.bbox for r in rs),candidate_label=rs[0].label)
        result.append(RegionHypothesis(hypothesis_id=group_id+':KEEP',action='KEEP',**base))
        for i,alt in enumerate(alternatives[:limit-1]):
            result.append(RegionHypothesis(hypothesis_id=f'{group_id}:{i}',**(base|alt)))
    # Connected local groups, never an entire column truncated to fit the limit.
    pending = {r.id for r in graph.nodes.values() if r.label in text_like_labels}
    while pending:
        seed = min(pending,key=lambda id:(graph.nodes[id].bbox[1],graph.nodes[id].bbox[0]))
        pending.remove(seed)
        component = {seed}
        changed = True
        while changed:
            changed = False
            for a,b in graph.neighbors:
                other = b if a in component else a if b in component else None
                if other in pending:
                    component.add(other)
                    pending.remove(other)
                    changed = True
        ids = sorted(component,key=lambda id:(graph.nodes[id].bbox[1],graph.nodes[id].bbox[0]))
        if 2 <= len(ids) <= max_group:
            owned = {lid for id in ids for lid in graph.ownership.get(id,())}
            lines = [l for l in graph.lines if l.line_id in owned]
            rows,cols,alignment = spatial_grid(lines)
            if alignment:
                add_group(ids,[dict(action='MERGE',candidate_label='table',geometry_reasons=('aligned_line_grid',))])
    for region in sorted(graph.nodes.values(),key=lambda r:r.id):
        own = set(graph.ownership.get(region.id,()))
        lines = sorted((l for l in graph.lines if l.line_id in own),key=lambda l:polygon_bbox(l.polygon)[1])
        alternatives = []
        classification = residual_classifications.get(region.id)
        if (classification is not None and classification.status == 'ok' and classification.classification is not None
                and classification.classification.label != 'unknown'):
            alternatives.append(dict(action='RELABEL', candidate_label=classification.classification.label,
                                     geometry_reasons=('glm_residual_classification',)))
        if len(lines) >= 2:
            boxes = [polygon_bbox(l.polygon) for l in lines]
            heights = [b[3]-b[1] for b in boxes]
            gaps = [boxes[i+1][1]-boxes[i][3] for i in range(len(boxes)-1)]
            cut = max(range(len(gaps)),key=gaps.__getitem__)
            if gaps[cut] >= max(10,median(heights)*1.5):
                parts = (lines[:cut+1],lines[cut+1:])
                # A larger first line and a shorter first cluster support title/body.
                heading = len(parts[0]) == 1 and heights[0] >= median(heights)
                children = tuple(SplitPart(id=f'{region.id}:part:{i}',bbox=union_bboxes(polygon_bbox(l.polygon) for l in part),
                                          label='title' if i == 0 and heading else 'text',
                                          line_ids=tuple(l.line_id for l in part)) for i,part in enumerate(parts))
                alternatives.append(dict(action='SPLIT',split_parts=children,geometry_reasons=('vertical_line_gap',)))
        text = ' '.join(l.transcript for l in lines)
        math_count = len(re.findall(r'[=+^_∑∫√]|\\(?:frac|sum|int)',text))
        if region.label in text_like_labels and math_count >= 2:
            alternatives.append(dict(action='RELABEL',candidate_label='formula',geometry_reasons=('independent_ocr_math',)))
        if lines and region.label in text_like_labels:
            b = union_bboxes(polygon_bbox(l.polygon) for l in lines)
            if b[0] < region.bbox[0] or b[1] < region.bbox[1] or b[2] > region.bbox[2] or b[3] > region.bbox[3]:
                alternatives.append(dict(action='REFINE_BBOX',candidate_bbox=union_bboxes([b,region.bbox]),geometry_reasons=('clipped_line_recovery',)))
            if spatial_grid(lines)[2]:
                alternatives.append(dict(action='RELABEL',candidate_label='table',geometry_reasons=('aligned_line_grid',)))
        add_group([region.id],alternatives,force_keep=region.id in residual_classifications)
    return result
