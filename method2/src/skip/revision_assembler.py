from copy import deepcopy
from shapely.geometry import Polygon,box
from ..schemas.region import LayoutRegion,FinalRegion
from ..schemas.output import PageParseResult,Provenance
from ..spatial.reading_order import reconcile_order_as_dag
from ..spatial.line_assignment import assign_lines_to_regions


def apply_revisions_to_graph(graph,revisions):
    updated = deepcopy(graph)
    consumed = set()
    for revision in revisions:
        if revision.status != 'accepted' or revision.action == 'KEEP':
            continue
        sources = set(revision.source_region_ids)
        try:
            if consumed & sources or not sources <= updated.nodes.keys():
                raise ValueError('conflicting_or_unknown_source')
            if revision.action != 'SPLIT' and any(not box(*revision.new_bbox).covers(box(*updated.nodes[id].bbox)) for id in sources):
                raise ValueError('source_geometry_loss')
            if revision.action == 'MERGE' and len({updated.columns[id] for id in sources})>1:
                raise ValueError('cross_column_violation')
            lineage = tuple(dict.fromkeys(s for id in revision.source_region_ids for s in updated.lineage[id]))
            order = min(updated.nodes[id].order for id in sources)
            column = updated.columns[revision.source_region_ids[0]]
            if revision.action == 'SPLIT':
                if len(revision.split_parts)<2:
                    raise ValueError('missing_split_children')
                replacements = [LayoutRegion(id=p.id,bbox=p.bbox,label=p.label,order=order) for p in revision.split_parts]
            else:
                replacements = [LayoutRegion(id=revision.target_region_id,bbox=revision.new_bbox,label=revision.new_label,order=order)]
            ids = [r.id for r in replacements]
            if len(ids) != len(set(ids)) or set(ids) & (updated.nodes.keys()-sources):
                raise ValueError('target_id_collision')
            source_lines = {lid for id in sources for lid in updated.ownership.get(id,())}
            relevant = [l for l in updated.lines if l.line_id in source_lines]
            ownership = assign_lines_to_regions(replacements,relevant,iou_threshold=.98)
            if {lid for values in ownership.values() for lid in values} != source_lines:
                raise ValueError('line_coverage_loss')
            if revision.action == 'SPLIT':
                from itertools import combinations
                if any(box(*a.bbox).intersection(box(*b.bbox)).area for a,b in combinations(replacements,2)):
                    raise ValueError('overlapping_split_children')
            trial = deepcopy(updated)
            incoming = {a for a,b in trial.edges if b in sources and a not in sources}
            outgoing = {b for a,b in trial.edges if a in sources and b not in sources}
            trial.edges = {(a,b) for a,b in trial.edges if a not in sources and b not in sources}
            for id in sources:
                del trial.nodes[id]
                del trial.columns[id]
                trial.ownership.pop(id,None)
                trial.lineage.pop(id,None)
            for region in replacements:
                trial.nodes[region.id] = region
                trial.columns[region.id] = column
                trial.ownership[region.id] = ownership[region.id]
                trial.lineage[region.id] = lineage
            trial.edges.update((id,ids[0]) for id in incoming)
            trial.edges.update((ids[-1],id) for id in outgoing)
            trial.edges.update(zip(ids,ids[1:]))
            before = len(trial.warnings)
            reconcile_order_as_dag(trial)
            if len(trial.warnings)>before:
                raise ValueError('revision_introduced_order_cycle')
            trial.applied_revision_ids.add(revision.revision_id)
            updated = trial
            consumed.update(sources)
        except (ValueError,KeyError) as exc:
            updated.warnings.append(f'rollback:{revision.revision_id}:{exc}')
            updated.rolled_back_revision_ids.add(revision.revision_id)
    return updated


def assemble_document(original_regions,final_graph,recognitions,revisions,warnings,*,original_lines=(),
                      recovered_regions=(),document_id='document',page_index=0,evidence=(),glm_requests=0):
    ordered = reconcile_order_as_dag(final_graph)
    revisions = [r.model_copy(update={'status':'rolled_back','needs_review':True,'reasons':(*r.reasons,'assembly_rollback')})
                 if r.revision_id in final_graph.rolled_back_revision_ids else r for r in revisions]
    review = {s for r in revisions if r.needs_review for s in r.source_region_ids}
    finals = []
    markdown = []
    for order,id in enumerate(ordered):
        region = final_graph.nodes[id]
        content = recognitions.get(id,'')
        sources = final_graph.lineage.get(id,(id,))
        fmt = {'table':'html','formula':'latex'}.get(region.label,'text')
        finals.append(FinalRegion(id=id,bbox_final=region.bbox,label_final=region.label,order_final=order,content=content,
            content_format=fmt,source_region_ids=sources,source_line_ids=final_graph.ownership.get(id,()),
            needs_review=bool(review.intersection(sources)) or not content.strip()))
        if region.label == 'formula':
            markdown.append('$$\n'+content+'\n$$')
        elif region.label in ('title','doc_title','paragraph_title'):
            markdown.append('# '+content)
        else:
            markdown.append(content)
    assigned_lines = {id for region in finals for id in region.source_line_ids}
    omitted = {line.line_id for line in original_lines}-assigned_lines
    if omitted:
        warnings = [*warnings,'unassigned_original_lines:'+','.join(sorted(omitted))]
    return PageParseResult(document_id=document_id,page_index=page_index,original_regions=original_regions,
        original_lines=original_lines,recovered_regions=recovered_regions,final_regions=finals,revisions=revisions,
        evidence=evidence,provenance=[Provenance(region_id=r.id,source_region_ids=r.source_region_ids,
        source_line_ids=r.source_line_ids) for r in finals],warnings=tuple(dict.fromkeys([*warnings,*final_graph.warnings])),
        markdown='\n\n'.join(markdown),glm_requests=glm_requests)
