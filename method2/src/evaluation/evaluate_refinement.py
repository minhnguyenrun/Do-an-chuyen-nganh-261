import json
from collections import Counter
from pathlib import Path
from .evaluate_parsing import compute_reading_order_ned,compute_parsing_metrics
from .evaluate_detection import compute_detection_metrics


def _key(value):
    data = value if isinstance(value,dict) else value.model_dump(mode='json')
    parts = tuple((tuple(p['bbox']),p['label']) for p in data.get('split_parts',()))
    return data['action'],tuple(sorted(data['source_region_ids'])),data.get('new_label',''),parts


def compute_refinement_metrics(predicted_revisions,ground_truth_revisions):
    predicted = [p for p in predicted_revisions if (p.get('status') if isinstance(p,dict) else p.status) == 'accepted'
                 and (p.get('action') if isinstance(p,dict) else p.action) != 'KEEP']
    gt = [g for g in ground_truth_revisions if g['action'] != 'KEEP']
    metrics = {}
    total_fp = 0
    for action in ('MERGE','SPLIT','RELABEL','REFINE_BBOX'):
        p = Counter(_key(v) for v in predicted if _key(v)[0] == action)
        g = Counter(_key(v) for v in gt if _key(v)[0] == action)
        tp = sum((p & g).values())
        fp = sum(p.values())-tp
        total_fp += fp
        precision,recall = tp/max(1,sum(p.values())),tp/max(1,sum(g.values()))
        metrics.update({f'{action.lower()}_precision':precision,f'{action.lower()}_recall':recall,
                        f'{action.lower()}_f1':2*precision*recall/max(1e-12,precision+recall)})
    metrics['over_correction_count'] = float(total_fp)
    metrics['over_correction_rate'] = total_fp/max(1,len(predicted))
    return metrics


ABLATIONS = {
    'A0': {'official_sdk':True},
    'A1': {'routing':{'enable_light_ocr':False,'semantic_probe_on_ambiguous':False},'spatial':{'enable_text_detector':False,'recover_residuals':False}},
    'A2': {'routing':{'enable_light_ocr':False,'semantic_probe_on_ambiguous':False}},
    'A3': {'routing':{'semantic_probe_on_ambiguous':False}},
    'A4': {'hypotheses':{'enabled_actions':['KEEP','RELABEL']}},
    'A5': {'hypotheses':{'enabled_actions':['KEEP','RELABEL','MERGE']}},
    'A6': {'hypotheses':{'enabled_actions':['KEEP','RELABEL','MERGE','SPLIT']}},
    'A7': {'validation':{'use_skip_l':False}},
    'A8': {'validation':{'use_skip_h':False}},
    'A9': {},
}


def run_ablation_suite(dataset_dir,configurations,runner=None):
    """Evaluate JSON page records. runner(record, name, overrides) returns predictions.

    Without a runner, every page must contain measured `predictions[name]`.
    A0 must be supplied by the official SDK runner; never simulate baseline metrics.
    """
    records = []
    for file in sorted(Path(dataset_dir).glob('*.json')):
        data = json.loads(file.read_text(encoding='utf-8'))
        records.extend(data if isinstance(data,list) else [data])
    if not records:
        raise ValueError('dataset contains no JSON page records')
    results = {}
    for name in configurations:
        if name not in ABLATIONS:
            raise ValueError(f'unknown ablation: {name}')
        predicted,truth = [],[]
        page_metrics = []
        for record in records:
            prediction = runner(record,name,ABLATIONS[name]) if runner else record.get('predictions',{}).get(name)
            if prediction is None:
                raise ValueError(f'missing measured predictions for {name}: {record.get("image", "page")}')
            if hasattr(prediction,'model_dump'):
                prediction = prediction.model_dump(mode='json')
            expected = record['ground_truth']
            # Namespace IDs per page before aggregating to prevent accidental cross-page matches.
            page_id = str(len(page_metrics))
            for source,target in ((prediction.get('revisions',[]),predicted),(expected.get('revisions',[]),truth)):
                for item in source:
                    target.append(item | {'source_region_ids':[page_id+':'+id for id in item['source_region_ids']]})
            order = [r['id'] for r in prediction.get('final_regions',[])]
            metrics = {}
            if 'reading_order' in expected:
                metrics['reading_order_ned'] = compute_reading_order_ned(order,expected['reading_order'])
            if 'text' in expected:
                metrics.update(compute_parsing_metrics(prediction.get('markdown',''),expected['text']))
            if 'regions' in expected:
                boxes = [dict(bbox=r['bbox_final'],label=r['label_final']) for r in prediction.get('final_regions',[])]
                metrics.update({'detection_'+k:v for k,v in compute_detection_metrics(boxes,expected['regions']).items()})
            if 'latency_seconds' in prediction:
                metrics['latency_seconds'] = float(prediction['latency_seconds'])
            if 'glm_requests' in prediction:
                metrics['glm_requests'] = float(prediction['glm_requests'])
            page_metrics.append(metrics)
        values = compute_refinement_metrics(predicted,truth)
        for key in {k for p in page_metrics for k in p}:
            values[key] = sum(p[key] for p in page_metrics if key in p)/sum(key in p for p in page_metrics)
        values['pages'] = float(len(records))
        results[name] = values
    return results
