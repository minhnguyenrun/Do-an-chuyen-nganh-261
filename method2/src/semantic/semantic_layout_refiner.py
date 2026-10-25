from ..skip.dual_skip_validator import validate_hard_constraints
from ..schemas.output import RegionRevision
from .hypothesis_scorer import calculate_hypothesis_score
from .residual_labeling import residual_role_evidence
from ..skip.dual_skip_validator import owned_lines


def refine_hypotheses(group_id,choices,evidence_map,skip_h,skip_l,cfg):
    if not choices or len(choices)>3:
        raise ValueError('a group requires 1..3 hypotheses')
    baselines = [h for h in choices if h.action == 'KEEP']
    if len(baselines) != 1:
        raise ValueError('a group requires exactly one KEEP baseline')
    baseline = baselines[0]
    if any(set(h.source_region_ids) != set(baseline.source_region_ids) for h in choices):
        raise ValueError('hypotheses must share the same sources')
    baseline_evidence = evidence_map.get(baseline.hypothesis_id)
    base_score = calculate_hypothesis_score(baseline,baseline_evidence,{},skip_h,skip_l,cfg)
    best,best_score = baseline,base_score
    reasons = []
    alternative_scores = []
    if baseline_evidence is None or baseline_evidence.status != 'ok':
        reasons.append('baseline_evidence_unavailable')
    else:
        for hyp in choices:
            if hyp.action == 'KEEP':
                continue
            evidence = evidence_map.get(hyp.hypothesis_id)
            passed,flags = validate_hard_constraints(hyp,skip_h,skip_l,evidence,
                use_skip_h=cfg.get('use_skip_h',True),use_skip_l=cfg.get('use_skip_l',True))
            reasons.extend(flags)
            if passed:
                score = calculate_hypothesis_score(hyp,evidence,{},skip_h,skip_l,cfg)
                alternative_scores.append(score)
                if score > best_score:
                    best,best_score = hyp,score
    accept = best.action != 'KEEP' and best_score >= cfg.get('min_absolute_score',.60) and best_score-base_score+1e-9 >= cfg.get('min_score_gain',.10)
    alternative_scores.sort(reverse=True)
    if len(alternative_scores)>1 and alternative_scores[0]-alternative_scores[1] < cfg.get('ambiguity_margin',.05):
        accept = False
        reasons.append('ambiguous_alternatives')
    chosen = best if accept else baseline
    accepted_reasons = chosen.geometry_reasons
    if accept and 'glm_residual_classification' in chosen.geometry_reasons:
        _, spatial_reasons = residual_role_evidence(chosen, skip_h.graph, owned_lines(chosen, skip_h, skip_l))
        accepted_reasons = (*accepted_reasons, *spatial_reasons, 'independent_transcript_verified')
    return RegionRevision(revision_id=f'rev:{group_id}',action=chosen.action,source_region_ids=chosen.source_region_ids,
        target_region_id=(chosen.source_region_ids[0] if chosen.action in ('KEEP','RELABEL','REFINE_BBOX') else f'final:{group_id}'),
        new_label=chosen.candidate_label,new_bbox=chosen.candidate_bbox,status='accepted' if accept else 'abstained',
        reasons=accepted_reasons if accept else tuple(dict.fromkeys(reasons+['insufficient_score_gain_or_confidence'])),
        hypothesis_id=chosen.hypothesis_id,baseline_score=base_score,score=best_score,
        split_parts=chosen.split_parts,needs_review=not accept)
