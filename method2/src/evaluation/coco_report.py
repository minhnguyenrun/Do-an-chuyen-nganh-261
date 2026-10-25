"""COCO AP and strict Micro-F1 through the repository's shared metrics module."""
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path

from ..label_mapping import doclaynet_category


def _score(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1:
        return float(value)
    return None


def _final_score(region, originals, line_scores):
    if 'score' in region:
        return _score(region['score']), 'explicit_final_score'
    sources = region.get('source_region_ids', [])
    values = [_score(originals[source].get('score')) for source in sources if source in originals]
    if sources and len(values) == len(sources):
        return (min(values) if all(v is not None for v in values) else None), 'inherited_layout_min'
    # RecoveredRegion.score has a synthetic schema default; do NOT use it.
    line_ids = region.get('source_line_ids', [])
    if not line_ids:
        return None, 'missing_detector_evidence'
    values.extend(_score(line_scores.get(line_id)) for line_id in line_ids)
    return (min(values) if values and all(v is not None for v in values) else None), 'inherited_detector_min'


def evaluate_coco_snapshot(report, snapshot_dir, output_dir, *, merged_headings=True, strict_iou=.5):
    import pandas as pd
    if not 0 < strict_iou <= 1:
        raise ValueError('strict_iou must be in (0, 1]')
    path = Path(__file__).resolve().parents[3] / 'src/common/metrics.py'
    spec = importlib.util.spec_from_file_location('dual_skip_shared_coco_metrics', path)
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    gt = json.loads(Path(report['annotation_file']).read_text(encoding='utf-8'))
    image_ids = {source['image_id'] for source in report['sources']}
    if not image_ids:
        raise ValueError('No evaluated real images in the snapshot.')
    def category(category_id):
        return 8 if merged_headings and category_id == 11 else category_id
    names = {category(c['id']): ('Heading' if merged_headings and c['id'] in (8, 11) else c['name'])
             for c in gt['categories']}
    annotations = [ann | {'id': index + 1, 'category_id': category(ann['category_id'])}
                   for index, ann in enumerate(a for a in gt['annotations'] if a['image_id'] in image_ids)]
    gt_coco = dict(info=gt.get('info', {}), images=[im for im in gt['images'] if im['id'] in image_ids],
                   annotations=annotations, categories=[dict(id=id, name=name) for id, name in sorted(names.items())])
    gt_by_image = {image_id: [] for image_id in image_ids}
    for ann in annotations:
        gt_by_image[ann['image_id']].append(dict(category_id=ann['category_id'], bbox=ann['bbox']))
    modes, rows, class_rows = {}, [], []
    for final, mode in ((False, 'baseline'), (True, 'final')):
        scored, predictions = [], {image_id: [] for image_id in image_ids}
        excluded, score_origins, missing = Counter(), Counter(), []
        for image_id in sorted(image_ids):
            result = json.loads((Path(snapshot_dir) / str(image_id) / 'result.json').read_text(encoding='utf-8'))
            if any('mock' in evidence.get('parser', '').lower() for evidence in result.get('evidence', [])):
                raise ValueError('Mock predictions cannot be used for real COCO quality reporting.')
            originals = {r['id']: r for r in result['original_regions']}
            line_scores = {line['line_id']: line.get('score') for line in result.get('original_lines', [])}
            for region in result['final_regions' if final else 'original_regions']:
                label = region['label_final' if final else 'label']
                category_id = doclaynet_category(label, heading_mode='merged' if merged_headings else 'literal')
                x1, y1, x2, y2 = region['bbox_final' if final else 'bbox']
                prediction = dict(category_id=category_id, bbox=[x1, y1, x2 - x1, y2 - y1])
                predictions[image_id].append(prediction)
                if category_id not in names:
                    excluded[label] += 1
                    continue
                score, origin = _final_score(region, originals, line_scores) if final else (
                    _score(region.get('score')), 'original_layout_score')
                score_origins[origin] += 1
                if score is None:
                    missing.append(dict(image_id=image_id, region_id=region['id']))
                else:
                    scored.append(prediction | {'image_id': image_id, 'score': score})
        strict = metrics.evaluate(gt_by_image, predictions, names | {0: 'Unmapped'}, strict_iou,
                                  require_same_category=True)
        ap, error = None, ''
        if missing:
            error = f'{len(missing)} mapped predictions have no measured score; AP is unavailable, no score is fabricated.'
        else:
            try:
                ap = metrics.coco_map(gt_coco, scored)
            except ImportError as exc:
                error = str(exc)
        policy = 'inherited detector scores (proxy, not calibrated final confidence)' if final else 'original PP-DocLayoutV3 scores'
        modes[mode] = dict(coco=ap, strict=strict, score_policy=policy, score_origins=dict(score_origins),
            ap_excluded_unmapped=dict(excluded), missing_scores=missing, ap_error=error,
            total_predictions=sum(map(len, predictions.values())), scored_ap_predictions=len(scored))
        overall = strict['overall']
        def percent(key):
            value = ap['overall'][key] if ap else -1
            return 100 * value if value >= 0 else None
        rows.append({'Method': 'Dual-Skip v4 final' if final else 'PP-DocLayoutV3 baseline',
            'COCO mAP@[.50:.95] (%)': percent('map'), 'AP50 (%)': percent('map50'),
            'AP75 (%)': percent('map75'), 'Strict Micro-F1 (%)': 100 * overall['f1'],
            'Strict Precision (%)': 100 * overall['precision'], 'Strict Recall (%)': 100 * overall['recall'],
            'Strict IoU': strict_iou, 'TP': overall['tp'], 'FP': overall['fp'], 'FN': overall['fn'],
            'AP excluded unmapped': sum(excluded.values()), 'Score policy': policy,
            'AP status': error or 'ok'})
        if ap:
            for category_id, value in ap['per_category'].items():
                class_rows.append({'Method': mode, 'Class': value['name'],
                    'AP@[.50:.95] (%)': 100 * value['ap'] if value['ap'] >= 0 else None,
                    'AP50 (%)': 100 * value['ap50'] if value['ap50'] >= 0 else None})
    output = dict(image_count=len(image_ids), source_snapshot=str(Path(snapshot_dir).resolve()),
        metrics_source='src/common/metrics.py::coco_map and evaluate(require_same_category=True)',
        taxonomy='10 classes, merged headings' if merged_headings else '11 original classes', modes=modes,
        notes=['AP excludes unmapped category IDs as required by coco_map; strict F1 retains those predictions as FP.',
               'Final AP inherits measured detector scores, not a GLM class probability or a calibrated final score.',
               'A missing measured score makes AP unavailable for that method; no predictions with missing scores are silently dropped.'])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'coco_metrics.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    summary, per_class = pd.DataFrame(rows), pd.DataFrame(class_rows)
    summary.to_csv(output_dir / 'coco_summary.csv', index=False, encoding='utf-8-sig')
    per_class.to_csv(output_dir / 'coco_per_class.csv', index=False, encoding='utf-8-sig')
    return output, summary, per_class
