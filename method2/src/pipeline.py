"""Four-stage pipeline. All coordinates refer to the unscaled BGR page image."""
import asyncio
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import numpy as np
import yaml
from .schemas.evidence import ParseEvidence
from .spatial.coordinates import polygon_bbox,expand_bbox
from .spatial.crop_builder import build_crop
from .spatial.line_assignment import assign_lines_to_regions
from .spatial.residual_regions import detect_residual_lines,cluster_residual_lines
from .spatial.region_graph import build_region_graph
from .skip.high_spatial_skip import HighSpatialSkip
from .skip.low_spatial_skip import LowSpatialSkip
from .skip.revision_assembler import apply_revisions_to_graph,assemble_document
from .semantic.candidate_generator import generate_hypotheses
from .semantic.prompt_router import route_prompt
from .semantic.probe_scheduler import ProbeScheduler
from .semantic.semantic_layout_refiner import refine_hypotheses
from .semantic.residual_labeling import classify_unlabeled_regions


def load_config(overrides=None):
    with (Path(__file__).resolve().parents[1]/'configs/default.yaml').open(encoding='utf-8') as stream:
        cfg = yaml.safe_load(stream)
    def merge(target,source):
        for key,value in source.items():
            if isinstance(value,dict) and isinstance(target.get(key),dict):
                merge(target[key],value)
            else:
                target[key] = deepcopy(value)
    merge(cfg,overrides or {})
    if cfg['hypotheses']['max_iterations'] != 1:
        raise ValueError('v4 supports exactly one refinement iteration')
    if cfg['validation']['min_score_gain'] < .10 or cfg['validation']['min_absolute_score'] < .60:
        raise ValueError('v4 conservative thresholds cannot be lowered')
    return cfg


async def parse_page(page_image,config,layout_adapter,text_adapter,light_ocr_adapter,glm_client,*,progress_callback=None):
    if not isinstance(page_image,np.ndarray) or page_image.dtype != np.uint8 or page_image.ndim != 3 or page_image.shape[2] != 3 or not page_image.size:
        raise ValueError('expected a nonempty uint8 BGR image (H,W,3)')
    cfg = load_config(config)
    def progress(stage, message):
        if progress_callback is not None:
            progress_callback(stage, message)
    h,w = page_image.shape[:2]
    warnings,evidence = [],[]
    # Model inference is blocking: keep it off the service event loop.
    progress('spatial', 'Đang phát hiện layout và các dòng chữ…')
    originals = tuple(await asyncio.to_thread(layout_adapter.detect,page_image))
    original_lines = tuple(await asyncio.to_thread(text_adapter.detect_lines,page_image)) if cfg['spatial']['enable_text_detector'] else ()
    lines = original_lines
    if cfg['routing']['enable_light_ocr'] and lines:
        progress('light_ocr', 'Đang nhận dạng từng dòng bằng PaddleOCR…')
        try:
            crops = [build_crop(page_image,polygon_bbox(l.polygon),mode='raw') for l in lines]
            transcripts = await asyncio.to_thread(light_ocr_adapter.recognize,crops)
            if len(transcripts) != len(lines):
                raise ValueError('line OCR response length mismatch')
            lines = tuple(l.model_copy(update={'transcript':text}) for l,text in zip(lines,transcripts))
        except Exception as exc:
            warnings.append(f'light_ocr_failed:{exc}')
    ownership = assign_lines_to_regions(originals,lines,cfg['hypotheses']['coverage_threshold'])
    assigned = {id for ids in ownership.values() for id in ids}
    residuals = cluster_residual_lines(detect_residual_lines(lines,assigned)) if cfg['spatial']['recover_residuals'] else []
    reserved = {r.id for r in originals}
    recovered = []
    for i,r in enumerate(residuals):
        id = f'recovered:{i}'
        while id in reserved:
            id = '_'+id
        reserved.add(id)
        recovered.append(r.model_copy(update={'id':id,'order':len(originals)+i}))
    regions = (*originals,*recovered)
    graph = build_region_graph(regions,w,h,lines=lines,coverage_threshold=cfg['hypotheses']['coverage_threshold'])
    progress('graph', f'Đã dựng graph: {len(regions)} vùng, {len(lines)} dòng chữ.')
    high,low = HighSpatialSkip(regions,graph),LowSpatialSkip(lines,page_image)
    inference = cfg['inference']
    scheduler = ProbeScheduler(inference['max_glm_requests_per_page'],inference['max_in_flight'])
    line_map = {l.line_id:l for l in lines}
    contents = {}
    def crop(bbox,polygon=None,line_ids=()):
        expanded = expand_bbox(bbox,cfg['spatial']['crop_margin'],w,h)
        return build_crop(page_image,expanded,polygon,line_polygons=[line_map[id].polygon for id in line_ids],
                          mask_certain=bool(line_ids))
    # Parse originals once. These recognitions are the KEEP baseline and rollback content.
    requests = []
    for region in regions:
        ids = graph.ownership[region.id]
        transcript = '\n'.join(line_map[id].transcript for id in ids)
        if cfg['routing']['enable_light_ocr'] and transcript.strip() and region.label not in cfg['routing']['complex_region_labels']:
            contents[region.id] = transcript
        else:
            requests.append((f'parse:{region.id}',crop(region.bbox,region.polygon,ids),route_prompt(region.label)))
    progress('recognition', f'Đang nhận dạng {len(requests)} vùng cần GLM…')
    initial = await scheduler.run(glm_client,requests)
    evidence.extend(initial)
    for ev in initial:
        id = ev.hypothesis_id.removeprefix('parse:')
        contents[id] = ev.content if ev.status == 'ok' else '\n'.join(line_map[lid].transcript for lid in graph.ownership[id])
        if ev.status != 'ok':
            warnings.append(f'{ev.status}:{id}')
    residual_classifications = {}
    if (cfg['routing']['classify_unlabeled_regions'] and cfg['routing']['semantic_probe_on_ambiguous']
            and 'RELABEL' in cfg['hypotheses']['enabled_actions']):
        progress('residual_classification', 'Đang đưa vùng chưa có nhãn qua GLM phân loại với ngữ cảnh trang…')
        residual_classifications, classified_evidence = await classify_unlabeled_regions(
            page_image, graph, contents, glm_client, scheduler)
        evidence.extend(classified_evidence)
        for region_id, classified in residual_classifications.items():
            if classified.classification is not None and classified.classification.label == 'unknown':
                warnings.append(f'classification_unknown:{region_id}')
    hyp_cfg = cfg['hypotheses']
    hyps = generate_hypotheses(graph,hyp_cfg['max_hypotheses_per_group'],hyp_cfg['max_candidate_group_size'],
        hyp_cfg['enabled_actions'],residual_classifications=residual_classifications) if cfg['routing']['semantic_probe_on_ambiguous'] else []
    groups = defaultdict(list)
    ev_map = {}
    pending = []
    for hyp in hyps:
        groups[hyp.group_id].append(hyp)
        if hyp.action == 'KEEP':
            content = '\n'.join(contents.get(id,'') for id in hyp.source_region_ids)
            ev = ParseEvidence(hypothesis_id=hyp.hypothesis_id,parser='reuse',prompt='',content=content,status='ok' if content.strip() else 'error')
            ev_map[hyp.hypothesis_id] = ev
            evidence.append(ev)
        elif 'glm_residual_classification' in hyp.geometry_reasons and hyp.candidate_label not in ('table', 'formula'):
            classification = residual_classifications[hyp.source_region_ids[0]]
            ev = classification.model_copy(update={'hypothesis_id': hyp.hypothesis_id,
                'content': contents.get(hyp.source_region_ids[0], ''), 'parser': 'glm_residual_label'})
            ev_map[hyp.hypothesis_id] = ev
            evidence.append(ev)
        elif hyp.action == 'SPLIT':
            for i,part in enumerate(hyp.split_parts):
                pending.append((f'{hyp.hypothesis_id}:part:{i}',crop(part.bbox,line_ids=part.line_ids),route_prompt(part.label)))
        else:
            pending.append((hyp.hypothesis_id,crop(hyp.candidate_bbox,hyp.candidate_polygon,hyp.source_line_ids),route_prompt(hyp.candidate_label)))
    progress('semantic', f'Đang kiểm chứng giả thuyết: {len(pending)} crop.')
    probed = await scheduler.run(glm_client,pending)
    by_hypothesis = {hyp.hypothesis_id: hyp for hyp in hyps}
    probed = [ev.model_copy(update={'classification': residual_classifications[by_hypothesis[ev.hypothesis_id].source_region_ids[0]].classification})
              if ev.hypothesis_id in by_hypothesis and 'glm_residual_classification' in by_hypothesis[ev.hypothesis_id].geometry_reasons
              else ev for ev in probed]
    evidence.extend(probed)
    ev_map.update({ev.hypothesis_id:ev for ev in probed})
    for hyp in hyps:
        if hyp.action == 'SPLIT':
            parts = [ev_map[f'{hyp.hypothesis_id}:part:{i}'] for i in range(len(hyp.split_parts))]
            ev = ParseEvidence(hypothesis_id=hyp.hypothesis_id,parser='glm',prompt=route_prompt(hyp.candidate_label),
                content='\n'.join(p.content for p in parts),status='ok' if all(p.status == 'ok' for p in parts) else 'error',
                part_contents=tuple(p.content for p in parts))
            ev_map[hyp.hypothesis_id] = ev
            evidence.append(ev)
    revisions = [refine_hypotheses(id,choices,ev_map,high,low,cfg['validation']) for id,choices in groups.items()]
    progress('assembly', 'Đang kiểm tra ràng buộc và lắp ráp kết quả…')
    final_graph = apply_revisions_to_graph(graph,revisions)
    for rev in revisions:
        if rev.revision_id not in final_graph.applied_revision_ids:
            continue
        ev = ev_map[rev.hypothesis_id]
        if rev.action == 'SPLIT':
            for part,text in zip(rev.split_parts,ev.part_contents):
                contents[part.id] = text
        else:
            contents[rev.target_region_id] = ev.content
    for ev in evidence:
        if ev.status != 'ok':
            warnings.append(f'{ev.status}:{ev.hypothesis_id}' + (f':{ev.error}' if ev.error else ''))
    if not regions:
        warnings.append('no_regions_detected')
    result = assemble_document(originals,final_graph,contents,revisions,warnings,original_lines=original_lines,
        recovered_regions=recovered,document_id=config.get('document_id','document'),page_index=config.get('page_index',0),
        evidence=evidence,glm_requests=scheduler.used)
    progress('done', f'Hoàn tất: {len(result.final_regions)} vùng; {scheduler.used} lượt GLM.')
    return result
