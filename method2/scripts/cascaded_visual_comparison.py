"""Render saved predictions against GT without loading any model."""
from pathlib import Path
import csv
import hashlib
from collections import defaultdict


def export_comparisons(report_dir, gt_path=None, limit=20, display_count=5):
    from src.cascaded.runner import read_json, image_read
    from src.cascaded.report import shared_metrics, convert
    from src.cascaded.types import category
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from IPython.display import display, Image

    if limit < 1 or not 0 <= display_count <= 5:
        raise ValueError('limit must be positive; display_count must be 0..5')
    report_dir = Path(report_dir)
    report = read_json(report_dir/'metrics.json')
    project = Path(__file__).resolve().parents[2]
    candidates = [Path(gt_path)] if gt_path else list((project/'datasets').glob(
        'doclaynet*/processed/annotations/*.json'))
    matches = [p for p in candidates if p.is_file() and
               hashlib.sha256(p.read_bytes()).hexdigest() == report['annotation_sha256']]
    if len(matches) != 1:
        raise ValueError('Cannot find unique matching GT; provide gt_path explicitly.')
    gt = read_json(matches[0])
    merged = report['merged_headings']
    mapped = lambda id: 8 if merged and id == 11 else id
    names = {mapped(c['id']): c['name'] for c in gt['categories']}
    if merged:
        names[8] = 'Heading'
    annotations = defaultdict(list)
    for a in gt['annotations']:
        annotations[a['image_id']].append(dict(a, category_id=mapped(a['category_id'])))
    output = report_dir/'visual_comparison_final'
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    metric = shared_metrics()
    cmap = plt.get_cmap('tab20')
    for image_id in report['image_ids'][:limit]:
        result = read_json(report_dir/'snapshot'/f'{image_id}.json')
        if result['mode'] != 'REAL':
            raise ValueError('Quality comparison requires REAL predictions.')
        image_path = Path(result['record']['image'])
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != result['image_sha256']:
            raise ValueError(f'Image differs from evaluated snapshot: {image_path}')
        image = image_read(image_path)[:,:,::-1]
        regions = result['stages']['final']
        predictions = convert(regions, merged)
        truth = annotations[image_id]
        pairs, missing, false = metric.match_image(truth, predictions, .5, True)
        fig, axes = plt.subplots(1, 2, figsize=(18, 10))
        try:
            for ax in axes:
                ax.imshow(image)
                ax.set_xlim(0, result['width'])
                ax.set_ylim(result['height'], 0)
                ax.axis('off')
            for ax, boxes, is_gt in ((axes[0], truth, True), (axes[1], regions, False)):
                for item in boxes:
                    if is_gt:
                        x,y,w,h = item['bbox']
                        cid = item['category_id']
                        label = names.get(cid, str(cid))
                    else:
                        x,y,right,bottom = item['bbox']
                        w,h = right-x,bottom-y
                        cid = category(item['label'], merged)
                        label = names.get(cid, 'Unmapped')+' ['+item['label']+']'
                    color = cmap(cid % 20) if cid else 'gray'
                    ax.add_patch(Rectangle((x,y), w,h, fill=False, edgecolor=color, linewidth=1.5))
                    ax.text(x, y, label, fontsize=6, va='bottom', color='black',
                            bbox={'facecolor':color, 'alpha':.75, 'edgecolor':'none', 'pad':.5})
            axes[0].set_title(f'Ground truth ({len(truth)} boxes)')
            axes[1].set_title(f'Final prediction ({len(regions)} boxes)')
            fig.suptitle(f"{report['variant'].upper()} | image {image_id} | "
                         f"strict IoU 0.50: TP={len(pairs)}, FP={len(false)}, FN={len(missing)}\n"
                         'Same class = same color; gray = Unmapped; review evidence is not a detection')
            fig.tight_layout(rect=(0,0,1,.94))
            path = output/f'{image_id}_gt_vs_prediction.png'
            fig.savefig(path, dpi=140, bbox_inches='tight')
        finally:
            plt.close(fig)
        rows.append({'image_id':image_id, 'tp':len(pairs), 'fp':len(false), 'fn':len(missing),
                     'gt_boxes':len(truth), 'prediction_boxes':len(regions), 'file':str(path)})
    with (output/'index.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['image_id','tp','fp','fn','gt_boxes','prediction_boxes','file'])
        writer.writeheader()
        writer.writerows(rows)
    for row in rows[:display_count]:
        display(Image(filename=row['file']))
    print(f'Saved {len(rows)} comparisons; displayed {min(len(rows), display_count)}. Folder: {output}')
    return rows
