"""Same-image, same-taxonomy evaluation using the repository's shared evaluator."""
from pathlib import Path
import importlib.util
import json
import numpy as np
from .types import category
from .runner import read_json, write_json, image_read


def shared_metrics():
    path = Path(__file__).resolve().parents[3]/'src/common/metrics.py'
    spec = importlib.util.spec_from_file_location('cascaded_shared_metrics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def convert(regions, merged):
    converted = []
    for r in regions:
        a, b, c, d = r['bbox']
        item = {'category_id': category(r['label'], merged), 'bbox': [a, b, c-a, d-b]}
        if r.get('score') is not None:
            item['score'] = r['score']
        converted.append(item)
    return converted


def evaluate(results, annotation, output_dir, merged=False, run_manifest=None):
    if not results or any(r['mode'] != 'REAL' for r in results):
        raise ValueError('Quality reporting requires saved REAL predictions; MOCK is excluded')
    metrics = shared_metrics()
    gt = read_json(annotation)
    ids = [r['record']['image_id'] for r in results]
    if None in ids or len(ids) != len(set(ids)):
        raise ValueError('Expected unique DocLayNet image IDs')
    images = {im['id']: im for im in gt['images']}
    for r in results:
        im = images[r['record']['image_id']]
        if (r['width'], r['height']) != (im['width'], im['height']):
            raise ValueError('Image/annotation coordinate frame mismatch')
    def mapped(c):
        return 8 if merged and c == 11 else c
    names = {mapped(c['id']): 'Heading' if merged and c['id'] in (8, 11) else c['name'] for c in gt['categories']}
    names[0] = 'Unmapped'
    annotations = [dict(a, category_id=mapped(a['category_id'])) for a in gt['annotations'] if a['image_id'] in ids]
    by_gt = {id: [] for id in ids}
    for a in annotations:
        by_gt[a['image_id']].append(a)
    rows, detail = [], {}
    for stage in ('paddle', 'ensemble', 'recovery', 'final'):
        pred = {r['record']['image_id']: convert(r['stages'][stage], merged) for r in results}
        detail[stage] = {}
        for threshold in (.5, .75):
            strict = metrics.evaluate(by_gt, pred, names, threshold, True)
            loose = metrics.evaluate(by_gt, pred, names, threshold, False)
            correct = matched = 0
            for id in ids:
                pairs, _, _ = metrics.match_image(by_gt[id], pred[id], threshold, False)
                matched += len(pairs)
                correct += sum(by_gt[id][g]['category_id'] == pred[id][p]['category_id'] for g, p, _ in pairs)
            accuracy = correct/matched if matched else None
            detail[stage][str(threshold)] = {'strict': strict, 'loose': loose,
                                           'classification_accuracy': accuracy, 'localized_matches': matched}
            from collections import Counter
            tp_sources, fp_sources, fn_classes = Counter(), Counter(), Counter()
            error_regions = []
            for r in results:
                id = r['record']['image_id']
                regions = r['stages'][stage]
                pairs, missing_gt, false_pred = metrics.match_image(by_gt[id], pred[id], threshold, True)
                tp_sources.update(regions[pi]['source'] for _,pi,_ in pairs)
                fp_sources.update(regions[pi]['source'] for pi in false_pred)
                fn_classes.update(names.get(by_gt[id][gi]['category_id'],'Unmapped') for gi in missing_gt)
                error_regions.append({'image_id':id,
                    'fp_region_ids':[regions[pi]['id'] for pi in false_pred],
                    'fn_annotation_ids':[by_gt[id][gi]['id'] for gi in missing_gt]})
            detail[stage][str(threshold)]['source_errors'] = {
                'tp_by_source':dict(tp_sources),'fp_by_source':dict(fp_sources),
                'fn_by_gt_class':dict(fn_classes),'per_image':error_regions}
            overall = strict['overall']
            macro = [v['f1'] for v in strict['per_category'].values() if v['support']]
            rows.append({'stage': stage, 'iou': threshold, **overall,
                         'macro_f1_gt_present': float(np.mean(macro)) if macro else None,
                         'loose_f1': loose['overall']['f1'], 'classification_accuracy': accuracy})
        ap, ap_error = None, 'N/A: final/recovered scores are not calibrated detector confidence'
        if stage in ('paddle', 'ensemble'):
            coco_gt = {'images': [images[id] for id in ids], 'annotations': annotations,
                       'categories': [{'id': id, 'name': name} for id, name in names.items() if id]}
            measured = [dict(p, image_id=id) for id, boxes in pred.items() for p in boxes if p['category_id'] in names and p['category_id']]
            if any(p.get('score') is None for boxes in pred.values() for p in boxes if p['category_id']):
                ap_error = 'N/A: missing detector score'
            else:
                try:
                    ap = metrics.coco_map(coco_gt, measured)
                    ap_error = ''
                except ImportError as exc:
                    ap_error = str(exc)
        detail[stage]['ap'] = ap
        detail[stage]['ap_status'] = ap_error or 'ok'
        detail[stage]['ap_excluded_unmapped'] = sum(p['category_id'] == 0 for boxes in pred.values() for p in boxes)
        for row in rows:
            if row['stage'] == stage:
                row.update({key: ap['overall'].get(key) if ap else None for key in ('map', 'map50', 'map75')})
    timing = timing_report(results)
    report = {'image_ids': ids, 'images': len(ids), 'merged_headings': merged,
              'variant': results[0]['config']['variant'],
              'configuration': results[0]['config'],
              'schema_version': results[0].get('schema_version', 'cascaded-layout-v1'),
              'annotation_sha256': __import__('hashlib').sha256(Path(annotation).read_bytes()).hexdigest(),
              'sources': [{'image_id': r['record']['image_id'], 'image_sha256': r['image_sha256']} for r in results],
              'rows': rows, 'details': detail, 'timing': timing,
              'notes': ['Scores across detectors are not calibrated.',
                        'Generic title is Unmapped in literal mode; merged mode maps it to Heading.',
                        'Latency is measured, not a promise of 30–50 ms/page.',
                        'Stage runtimes include preparation/gating as applicable, exclude model initialization.']}
    report['review_summary'] = review_summary(results)
    if run_manifest:
        report['phase_runs'] = read_json(run_manifest).get('phase_runs', {})
    output_dir = Path(output_dir)
    write_json(output_dir/'metrics.json', report)
    # Frozen evaluated inputs make results reproducible even after later runs.
    for r in results:
        write_json(output_dir/'snapshot'/f'{r["record"]["image_id"]}.json', r)
    import pandas as pd
    pd.DataFrame(rows).to_csv(output_dir/'summary.csv', index=False, encoding='utf-8-sig')
    per_class = [{'stage': stage, 'iou': threshold, 'category_id': cid, **values}
                 for stage, data in detail.items() for threshold in ('0.5', '0.75')
                 for cid, values in data[threshold]['strict']['per_category'].items()]
    pd.DataFrame(per_class).to_csv(output_dir/'per_class.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(attribution_rows(results)).to_csv(output_dir/'mineru_boxes.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([{'document_id': r.get('document_id'), **d} for r in results
                  for d in r.get('decisions', [])]).to_csv(output_dir/'decisions.csv', index=False, encoding='utf-8-sig')
    text = ['# Cascaded layout evaluation', '', f'Images: {len(ids)}; merged headings: {merged}.', '',
            '| Stage | IoU | Precision | Recall | Strict F1 | Loose F1 |', '|---|---:|---:|---:|---:|---:|']
    text += [f"| {r['stage']} | {r['iou']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['loose_f1']:.4f} |" for r in rows]
    text += ['', 'Final AP: N/A. No final confidence was fabricated.', '',
             'Latency excludes model initialization; device-wide memory and phase wall times are in the run manifest.',
             'Generic title remains Unmapped in literal evaluation. See metrics.json for full evidence.']
    (output_dir/'report.md').write_text('\n'.join(text), encoding='utf-8')
    return report


def timing_report(results):
    times = [sum(v['inference_seconds'] for v in r['timing'].values()) for r in results]
    all_lines = sum(len(r['lines']) for r in results)
    solved = {}
    for stage in ('ensemble', 'recovery', 'final'):
        solved[stage] = sum(len(p['line_ids']) for r in results for p in r['stages'][stage]
                            if p['label'] != 'unknown')/max(1, all_lines)
    memory = [v['peak_memory']['reserved_mb'] for r in results for v in r['timing'].values() if v.get('peak_memory')]
    token_counts = [e.get('tokens') for r in results for e in r['evidence'] if e['status'] not in ('budget_exhausted', 'no_target_lines', 'disabled', 'invalid_request')]
    return {'median_seconds': float(np.median(times)), 'p95_seconds': float(np.percentile(times, 95)),
            'amortized_pages_per_second': len(times)/sum(times) if sum(times) else None,
            'inference_seconds_total': sum(times), 'solved_line_fraction': solved,
            'mineru_crop_calls': len(token_counts),
            'mineru_tokens': sum(token_counts) if all(v is not None for v in token_counts) else None,
            'torch_peak_reserved_mb': max(memory) if memory else None,
            'paddle_peak_vram_mb': None,
            'memory_note': 'PyTorch allocator only. Paddle child VRAM is not measured; no fabricated total.'}


def review_summary(results):
    from collections import Counter
    reasons = Counter(e['reason'] for r in results for e in r.get('review_evidence', []))
    actions = Counter(d['action'] for r in results for d in r.get('decisions', []))
    checkpoints = Counter(c for r in results for e in r['evidence'] for c in e.get('checkpoints', []))
    box_codes = Counter(c for r in results for e in r['evidence'] for h in e.get('boxes', []) for c in h['codes'])
    total = sum(len(r['lines']) for r in results)
    unresolved = sum(len({l['id'] for l in r['lines']} -
                         {id for p in r['stages']['final'] for id in p['line_ids']}) for r in results)
    return {'unresolved_line_count': unresolved, 'total_lines': total,
            'unresolved_line_fraction': unresolved/max(1,total),
            'review_reason_counts': dict(reasons), 'decision_counts': dict(actions),
            'mineru_checkpoints': dict(checkpoints), 'mineru_box_codes': dict(box_codes),
            'review_box_count': sum(len(r.get('review_evidence',[])) for r in results),
            'note':'GT not detected remains FN, even when its OCR lines are in review.'}


def attribution_rows(results):
    """One row per returned MinerU box; full polygon links remain in result.json."""
    return [{'document_id': r.get('document_id'), 'target_id': e['target']['id'],
             'status': e['status'], 'box_index': h['box_index'],
             'label': h.get('label'), 'codes': '|'.join(h['codes']),
             'raw_bbox': h.get('raw_bbox'), 'page_bbox': h.get('page_bbox'),
             'post_snap_bbox': h.get('post_snap_bbox'), 'candidate_bbox': h.get('candidate_bbox'),
             'target_line_ids': h.get('target_line_ids', []),
             'non_target_ink_ratio': h.get('attribution', {}).get('non_target_ink_ratio'),
             'nontext_overlap': h.get('attribution', {}).get('nontext_overlap'),
             'neighbor_ids': h.get('attribution', {}).get('neighbor_ids', []),
             'semantic_score': h.get('semantic_score'),
             'selected_partition': h.get('selected_partition', False)}
            for r in results for e in r['evidence'] for h in e.get('boxes', [])]


def compare(report_paths, output_dir):
    reports = [read_json(path) for path in report_paths]
    if len(reports) != 2:
        raise ValueError('Select exactly one S report and one V3 report')
    a, b = reports
    if [a.get('variant'), b.get('variant')] != ['s', 'v3']:
        raise ValueError('Supply S report first, V3 report second')
    for field in ('image_ids', 'sources', 'annotation_sha256', 'merged_headings', 'schema_version'):
        if a[field] != b[field]:
            raise ValueError(f'Cannot compare different datasets/taxonomies: {field}')
    configurations = [{k: v for k, v in report['configuration'].items() if k != 'variant'} for report in reports]
    if configurations[0] != configurations[1]:
        raise ValueError('Cannot compare runs with different gating/runtime configurations')
    import pandas as pd
    rows = [{'variant': variant, **row,
             **{k: report['timing'][k] for k in ('median_seconds', 'p95_seconds', 'amortized_pages_per_second', 'mineru_crop_calls', 'mineru_tokens')}}
            for variant, report in zip(('s', 'v3'), reports) for row in report['rows']]
    table = pd.DataFrame(rows)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    table.to_csv(Path(output_dir)/'comparison.csv', index=False, encoding='utf-8-sig')
    write_json(Path(output_dir)/'comparison.json', {'sources': [str(p) for p in report_paths], 'rows': rows})
    return table


def preview(result, max_crops=8, annotation=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    image = image_read(result['record']['image'])[:, :, ::-1]
    figure, axes = plt.subplots(1, 5, figsize=(27, 8))
    panels = [('Strong/weak proposals', result['proposals']), ('Tier 1', result['stages']['ensemble']),
              ('Tier 2', result['stages']['recovery']), ('Final detections', result['stages']['final']),
              ('Review only (not detections)', result.get('review_evidence',[]))]
    for ax, (title, regions) in zip(axes, panels):
        ax.imshow(image)
        for p in regions:
            x, y, c, d = p['bbox']
            color = 'gold' if title.startswith('Review') else 'red' if p['label'] == 'unknown' else 'orange' if (p.get('score') or 0) < .45 else 'lime'
            ax.add_patch(Rectangle((x, y), c-x, d-y, fill=False, edgecolor=color))
            ax.text(x, y, p['label'], color=color, fontsize=7)
        ax.set_title(title)
        ax.axis('off')
    figure.suptitle(f"{result['mode']} — {result['document_id']}")
    figure.tight_layout()
    from IPython.display import display
    display(figure)
    plt.close(figure)
    if annotation:
        gt = read_json(annotation)
        fig, ax = plt.subplots(figsize=(9, 12))
        ax.imshow(image)
        names = {c['id']: c['name'] for c in gt['categories']}
        for a in gt['annotations']:
            if a['image_id'] == result['record']['image_id']:
                x, y, w, h = a['bbox']
                ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='cyan'))
                ax.text(x, y, names[a['category_id']], color='blue', fontsize=8)
        ax.set_title('Ground truth — same image')
        ax.axis('off')
        display(fig)
        plt.close(fig)
    # Exact target windows, including orphan line polygons, for audit.
    for e in result['evidence'][:max_crops]:
        fig, ax = plt.subplots(figsize=(7, 4))
        bbox = e.get('crop_bbox', e['target']['bbox'])
        x, y, c, d = map(int, bbox)
        ax.imshow(image[y:d, x:c])
        for line in result['lines']:
            if line['id'] in e['target']['line_ids']:
                points = np.asarray(line['polygon'])-[x, y]
                ax.plot(*np.vstack([points, points[0]]).T, color='red')
        confirmed = set(e.get('frozen_ownership', {}).get('confirmed_region_ids', []))
        for p in result['stages']['recovery']:
            a, b, c, d = p['bbox']
            if p['id'] in confirmed and a < bbox[2] and c > bbox[0] and b < bbox[3] and d > bbox[1]:
                ax.add_patch(Rectangle((a-x,b-y), c-a, d-b, fill=False, edgecolor='cyan'))
        for h in e.get('boxes', []):
            if 'page_bbox' not in h:
                continue
            a,b,c,d = h.get('post_snap_bbox', h['page_bbox'])
            color = 'magenta' if h.get('selected_partition') else 'orange'
            ax.add_patch(Rectangle((a-x,b-y), c-a, d-b, fill=False, edgecolor=color, linestyle='--'))
            ax.text(a-x,b-y,f"#{h['box_index']}", color=color, fontsize=8)
        ax.set_xlim(0, bbox[2]-x)
        ax.set_ylim(bbox[3]-y, 0)
        codes = ', '.join(e.get('checkpoints', []))
        ax.set_title(f"{e['target']['id']}: {e['status']}\n{codes}", fontsize=9, wrap=True)
        fig.text(.05, .02, 'Red: target OCR; cyan: confirmed; orange: MinerU; magenta: selected partition', fontsize=8)
        ax.axis('off')
        display(fig)
        plt.close(fig)
