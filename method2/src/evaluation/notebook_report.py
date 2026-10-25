"""Measured layout metrics and final-vs-GT figures from saved real predictions."""
import importlib.util
import json
from pathlib import Path

from ..label_mapping import DOCLAYNET_CLASSES, doclaynet_category


def evaluate_saved_notebook(annotation_file, predictions_dir, output_dir, *, merged_headings=True, iou=.5):
    import pandas as pd
    if iou not in (.5, .75):
        raise ValueError('IoU must be 0.5 or 0.75')
    predictions_dir, output_dir = Path(predictions_dir).resolve(), Path(output_dir).resolve()
    if output_dir.is_relative_to(predictions_dir):
        raise ValueError('Keep evaluation snapshots outside the predictions directory.')
    path = Path(__file__).resolve().parents[2] / 'scripts/evaluate_saved_doclaynet.py'
    spec = importlib.util.spec_from_file_location('notebook_saved_doclaynet_eval', path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    report = evaluator.evaluate_saved(annotation_file, predictions_dir, output_dir)
    evaluator.write_markdown_report(report, output_dir)
    rows = []
    suffix = 'headings_merged' if merged_headings else 'literal'
    for threshold, modes in report['metrics'].items():
        for kind, name in [('baseline', 'PP-DocLayoutV3 (baseline)'), ('final', 'Dual-Skip v4 (final)')]:
            values = modes[f'{kind}_{suffix}']
            overall = values['strict']['overall']
            accuracy = values['classification_on_localized_matches']
            rows.append({'Method': name, 'IoU': float(threshold),
                'Precision (%)': 100 * overall['precision'], 'Recall (%)': 100 * overall['recall'],
                'Micro-F1 (%)': 100 * overall['f1'],
                'Macro-F1 (%)': 100 * overall['macro_f1_gt_present'],
                'Classification accuracy (%)': None if accuracy['accuracy'] is None else 100 * accuracy['accuracy'],
                'BBox-only F1 (%)': 100 * values['localization_only']['overall']['f1'],
                'TP': overall['tp'], 'FP': overall['fp'], 'FN': overall['fn'],
                'Correct / localized': f"{accuracy['correct']}/{accuracy['matched']}"})
    summary = pd.DataFrame(rows)
    selected = report['metrics'][str(iou)][f'final_{suffix}']
    category_rows = []
    for category_id, values in selected['strict']['per_category'].items():
        if not (values['support'] or values['fp']):
            continue
        category_rows.append({'Class': 'Heading' if merged_headings and int(category_id) == 8 else
                               DOCLAYNET_CLASSES.get(int(category_id), 'Unmapped'),
            'GT': values['support'], 'TP': values['tp'], 'FP': values['fp'], 'FN': values['fn'],
            'Precision (%)': 100 * values['precision'], 'Recall (%)': 100 * values['recall'],
            'F1 (%)': 100 * values['f1']})
    categories = pd.DataFrame(category_rows)
    summary.to_csv(output_dir / 'notebook_summary.csv', index=False, encoding='utf-8-sig')
    categories.to_csv(output_dir / 'notebook_per_class.csv', index=False, encoding='utf-8-sig')
    lines = ['# Notebook layout evaluation', '',
        f"Evaluated images: {report['image_count']}; taxonomy: " +
        ('10 classes, Title + Section-header merged on GT and predictions.' if merged_headings else '11 original DocLayNet classes.'), '',
        'Accuracy is conditional on localized one-to-one matches. This is not OCR accuracy.',
        'No directly comparable Docling/M2Doc runs are available; no external benchmark scores or mAP are fabricated.', '',
        '| ' + ' | '.join(summary.columns) + ' |', '| ' + ' | '.join('---' for _ in summary.columns) + ' |']
    for row in summary.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(f'{v:.2f}' if isinstance(v, float) else str(v) for v in row) + ' |')
    lines.extend(['', f'Per-class results: notebook_per_class.csv (IoU {iou}).',
        'All thresholds/taxonomies and snapshot sources: metrics.json.',
        'The companion report.md retains the evaluator literal 11-class primary report.', ''])
    (output_dir / 'notebook_report.md').write_text('\n'.join(lines), encoding='utf-8')
    return report, summary, categories


def layout_metrics_figure(report, *, merged_headings=True, iou=.5):
    """Same strict/loose protocol as the shared evaluator; scores on a 0..1 axis."""
    import numpy as np
    import matplotlib.pyplot as plt
    if iou not in (.5, .75):
        raise ValueError('IoU must be 0.5 or 0.75')
    mode = 'final_headings_merged' if merged_headings else 'final_literal'
    selected = report['metrics'][str(iou)][mode]
    strict, loose = selected['strict'], selected['localization_only']
    categories = [10, 1, 5, 7, 8, 2, 3, 4, 6, 9] + ([] if merged_headings else [11])
    if report.get('unmapped_labels'):
        categories.append(0)
    def values(metrics, category):
        return metrics['per_category'].get(category, metrics['per_category'].get(str(category), {}))
    def scores(metrics, name):
        return [values(metrics, category).get(name, 0.) for category in categories]
    labels = ['Heading' if merged_headings and c == 8 else DOCLAYNET_CLASSES.get(c, 'Unmapped')
              for c in categories]
    y = np.arange(len(categories))
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6.5, len(categories) * .43 + 2)), sharey=True)
    axes[0].barh(y - .23, scores(strict, 'precision'), height=.22, label='Precision', color='#438dff')
    axes[0].barh(y, scores(strict, 'recall'), height=.22, label='Recall', color='#ff7f1c')
    axes[0].barh(y + .23, scores(strict, 'f1'), height=.22, label='F1-score', color='#13ba8b')
    axes[1].barh(y - .17, scores(strict, 'f1'), height=.32,
                 label='Strict F1 (class + location)', color='#13ba8b')
    axes[1].barh(y + .17, scores(loose, 'f1'), height=.32,
                 label='Loose F1 (location only)', color='#8d5bff')
    axes[0].set_yticks(y, labels=labels)
    axes[0].invert_yaxis()
    axes[0].set_title('Strict metrics (location & class)', fontweight='bold')
    axes[1].set_title('F1 comparison: strict vs loose (localization)', fontweight='bold')
    for ax, xlabel, overall in zip(axes, ('Score', 'F1-score'), (strict['overall'], loose['overall'])):
        ax.set_xlim(0, 1.03)
        ax.set_xlabel(xlabel)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.grid(axis='x', linestyle='--', alpha=.35)
        ax.set_axisbelow(True)
        ax.legend(loc='lower right', fontsize=8)
        ax.text(.01, -.12, f"Micro: P={overall['precision']:.3f} | R={overall['recall']:.3f} | F1={overall['f1']:.3f}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor='white', edgecolor='#cccccc', boxstyle='round,pad=.3'))
    # Absent categories are not measured zero-quality classes.
    for index, category in enumerate(categories):
        data = values(strict, category)
        if not data.get('support', 0) and not data.get('fp', 0):
            axes[0].text(.015, index, 'N/A (no GT/prediction)', va='center', color='#999999', fontsize=8)
            axes[1].text(.015, index, 'N/A', va='center', color='#999999', fontsize=8)
    taxonomy = '10 classes (Title + Section-header merged)' if merged_headings else '11 original classes'
    accuracy = selected['classification_on_localized_matches']
    accuracy_text = 'N/A' if accuracy['accuracy'] is None else f"{100 * accuracy['accuracy']:.2f}%"
    fig.suptitle(f"Dual-Skip v4 Layout Evaluation (DocLayNet, IoU >= {iou:g})\n"
                 f"{report['image_count']} saved real images | {taxonomy}", fontweight='bold', fontsize=13)
    fig.text(.5, .025, f"Classification accuracy on localized matches: {accuracy_text} "
             f"({accuracy['correct']}/{accuracy['matched']}). Not OCR accuracy.\n"
             'Loose: class-agnostic matching; matched TP/FN grouped by GT class, unmatched FP by predicted class.',
             ha='center', fontsize=9, color='#444444')
    fig.tight_layout(rect=(0, .10, 1, .92))
    return fig


def _overlay(image, entries):
    from ..notebook_visuals import _canvas, label_color
    import numpy as np
    import cv2
    canvas, draw, font = _canvas(image)
    for entry in entries:
        color = label_color(entry['canonical'])
        x1, y1, x2, y2 = entry['bbox']
        draw.rectangle((x1, y1, x2, y2), outline=color, width=max(2, image.shape[1] // 500))
        text = entry['display']
        bounds = draw.textbbox((0, 0), text, font=font)
        height = bounds[3] - bounds[1] + 6
        top = max(0, y1 - height)
        draw.rectangle((x1, top, min(image.shape[1], x1 + bounds[2] + 6), top + height), fill=color)
        draw.text((x1 + 3, top - bounds[1] + 2), text, font=font, fill='white')
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def final_gt_figures(report, image_dir, snapshot_dir, *, merged_headings=True, max_images=3, documents=None):
    """Yield figures from the exact snapshot evaluated above, never live result files."""
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from ..notebook_visuals import label_color
    if type(max_images) is not int or max_images < 1:
        raise ValueError('max_images must be a positive integer')
    annotation = json.loads(Path(report['annotation_file']).read_text(encoding='utf-8'))
    images = {image['id']: image for image in annotation['images']}
    sources = sorted(report['sources'], key=lambda source: str(source['file_name']))
    if documents is not None:
        requested = {str(doc) for doc in documents}
        sources = [source for source in sources if Path(source['file_name']).stem in requested]
        if not sources:
            raise ValueError('Selected documents have no evaluated saved predictions.')
    elif len(sources) > max_images:
        sources = [sources[int(i)] for i in np.linspace(0, len(sources) - 1, max_images)]
    def canonical(category):
        return 'Heading' if merged_headings and category in (8, 11) else DOCLAYNET_CLASSES.get(category, 'Unmapped')
    for source in sources[:max_images]:
        image_id = source['image_id']
        metadata = images[image_id]
        path = Path(image_dir) / metadata['file_name']
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (metadata['height'], metadata['width']):
            raise ValueError(f'Image/annotation coordinate frame mismatch: {path}')
        result = json.loads((Path(snapshot_dir) / str(image_id) / 'result.json').read_text(encoding='utf-8'))
        ground_truth, predictions = [], []
        for ann in annotation['annotations']:
            if ann['image_id'] != image_id:
                continue
            x, y, width, height = ann['bbox']
            label = canonical(ann['category_id'])
            raw = DOCLAYNET_CLASSES[ann['category_id']]
            ground_truth.append(dict(bbox=(x, y, x + width, y + height), canonical=label,
                                     display=label if label == raw else f'{label} [{raw}]'))
        for region in result['final_regions']:
            raw = region['label_final']
            label = canonical(doclaynet_category(raw))
            predictions.append(dict(bbox=region['bbox_final'], canonical=label, display=f'{label} [{raw}]'))
        fig, axes = plt.subplots(1, 2, figsize=(17, 9))
        for ax, entries, title in zip(axes, (predictions, ground_truth),
                ('Final prediction (mapped class [pipeline label])', 'Ground truth DocLayNet')):
            ax.imshow(cv2.cvtColor(_overlay(image, entries), cv2.COLOR_BGR2RGB))
            ax.set_title(f"{metadata['file_name']} - {title}")
            ax.axis('off')
        labels = sorted({e['canonical'] for e in predictions + ground_truth})
        fig.legend(handles=[Patch(color=label_color(label), label=label) for label in labels],
                   loc='lower center', ncol=min(6, len(labels)))
        fig.tight_layout(rect=(0, .07, 1, 1))
        yield Path(metadata['file_name']).stem, fig
