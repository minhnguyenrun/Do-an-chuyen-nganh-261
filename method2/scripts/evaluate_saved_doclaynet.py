"""Evaluate a frozen snapshot of saved results; never run or interrupt inference."""
import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('shared_layout_metrics', ROOT / 'src/common/metrics.py')
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)

mapping_spec = importlib.util.spec_from_file_location('dual_skip_label_mapping', ROOT / 'method2/src/label_mapping.py')
label_mapping = importlib.util.module_from_spec(mapping_spec)
mapping_spec.loader.exec_module(label_mapping)
LABELS = label_mapping.PIPELINE_TO_DOCLAYNET



def xywh(bbox):
    x1, y1, x2, y2 = bbox
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def predictions(result, *, final=True, heading_mode='literal'):
    output = []
    for region in result['final_regions' if final else 'original_regions']:
        label = region['label_final' if final else 'label']
        category = label_mapping.doclaynet_category(label, heading_mode=heading_mode)
        output.append(dict(category_id=category, bbox=xywh(region['bbox_final' if final else 'bbox']), label=label))
    return output


def summarize(gt, pred, names, threshold):
    strict = metrics.evaluate(gt, pred, names, threshold, require_same_category=True)
    loose = metrics.evaluate(gt, pred, names, threshold, require_same_category=False)
    count, correct, confusions = 0, 0, Counter()
    for image_id in gt:
        matches, _, _ = metrics.match_image(gt[image_id], pred[image_id], threshold, require_same_category=False)
        for gi, pi, _ in matches:
            true_class, predicted_class = gt[image_id][gi]['category_id'], pred[image_id][pi]['category_id']
            count += 1
            correct += true_class == predicted_class
            if true_class != predicted_class:
                confusions[(names.get(true_class, str(true_class)), names.get(predicted_class, 'Unmapped'))] += 1
    strict['overall']['macro_f1_gt_present'] = sum(v['f1'] for v in strict['per_category'].values() if v['support']) / max(1, sum(v['support'] > 0 for v in strict['per_category'].values()))
    return dict(strict=strict, localization_only=loose,
        classification_on_localized_matches=dict(correct=correct, matched=count, accuracy=correct / count if count else None),
        top_confusions=[dict(ground_truth=a, predicted=b, count=n) for (a, b), n in confusions.most_common()])


def evaluate_saved(annotation_file, predictions_dir, output_dir):
    ground_truth = json.loads(Path(annotation_file).read_text(encoding='utf-8'))
    images = {Path(image['file_name']).stem: image for image in ground_truth['images']}
    names = {c['id']: c['name'] for c in ground_truth['categories']} | {0: 'Unmapped'}
    output_dir = Path(output_dir)
    snapshot_dir = output_dir / 'snapshot'
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    selected, excluded, sources = {}, [], []
    # Freeze the initial filename list: pages that finish later belong to the next evaluation.
    for path in sorted(Path(predictions_dir).rglob('result.json')):
        try:
            before = path.stat()
            raw = path.read_bytes()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('file changed while reading')
            result = json.loads(raw)
            if any(e.get('parser') == 'glm_mock' for e in result.get('evidence', [])) or any(r.get('content') == 'Mock text' for r in result['final_regions']):
                raise ValueError('mock result excluded')
            image = images.get(str(result['document_id']))
            if image is None or result['page_index'] != 0:
                raise ValueError('no matching DocLayNet image')
            if image['id'] in selected:
                raise ValueError('duplicate image ID')
            # Require the final file to have the same coordinate frame as the sampled GT.
            for r in result['final_regions']:
                x1, y1, x2, y2 = r['bbox_final']
                if not (0 <= x1 < x2 <= image['width'] and 0 <= y1 < y2 <= image['height']):
                    raise ValueError('prediction coordinate frame mismatch')
            selected[image['id']] = result
            page_folder = snapshot_dir / str(image['id'])
            page_folder.mkdir(exist_ok=True)
            (page_folder / 'result.json').write_bytes(raw)
            sources.append(dict(image_id=image['id'], file_name=image['file_name'], prediction_file=str(path.resolve()),
                                modified=datetime.fromtimestamp(after.st_mtime, timezone(timedelta(hours=7))).isoformat()))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            excluded.append(dict(file=str(path), reason=str(exc)))
    if not selected:
        raise ValueError('No saved real predictions match the annotation file')
    gt = {image_id: [] for image_id in selected}
    for ann in ground_truth['annotations']:
        if ann['image_id'] in gt:
            gt[ann['image_id']].append(dict(category_id=ann['category_id'], bbox=ann['bbox']))
    pred = {image_id: predictions(result) for image_id, result in selected.items()}
    baseline = {image_id: predictions(result, final=False) for image_id, result in selected.items()}
    sections = {image_id: predictions(result, heading_mode='section') for image_id, result in selected.items()}
    coarse_gt = {image_id: [g | {'category_id': 8 if g['category_id'] == 11 else g['category_id']} for g in boxes] for image_id, boxes in gt.items()}
    coarse_pred = {image_id: predictions(result, heading_mode='merged') for image_id, result in selected.items()}
    coarse_baseline = {image_id: predictions(result, final=False, heading_mode='merged') for image_id, result in selected.items()}
    modes = {}
    for threshold in (.5, .75):
        modes[str(threshold)] = dict(
            final_literal=summarize(gt, pred, names, threshold),
            baseline_literal=summarize(gt, baseline, names, threshold),
            baseline_headings_merged=summarize(coarse_gt, coarse_baseline, names | {8: 'Heading (Title + Section-header)'}, threshold),
            final_title_as_section=summarize(gt, sections, names, threshold),
            final_headings_merged=summarize(coarse_gt, coarse_pred, names | {8: 'Heading (Title + Section-header)'}, threshold))
    rows = []
    for image_id in sorted(selected):
        value = summarize({image_id: gt[image_id]}, {image_id: pred[image_id]}, names, .5)
        rows.append(dict(image_id=image_id, file_name=next(i['file_name'] for i in sources if i['image_id'] == image_id),
            ground_truth=len(gt[image_id]), predictions=len(pred[image_id]), **value['strict']['overall'],
            localization_f1=value['localization_only']['overall']['f1']))
    report = dict(snapshot_time=datetime.now(timezone(timedelta(hours=7))).isoformat(),
        annotation_file=str(Path(annotation_file).resolve()), image_count=len(selected), sources=sources, excluded=excluded,
        ground_truth_boxes=sum(map(len, gt.values())), final_boxes=sum(map(len, pred.values())),
        baseline_boxes=sum(map(len, baseline.values())),
        unmapped_labels=dict(Counter(p['label'] for boxes in pred.values() for p in boxes if p['category_id'] == 0)),
        revisions=dict(Counter(f"{r['action']}:{r['status']}" for result in selected.values() for r in result.get('revisions', []))),
        label_mapping=LABELS, official_sdk_mapping=label_mapping.mapping_records(), metrics=modes, per_image=rows,
        notes=[
            'Only saved real results in the frozen snapshot are evaluated; unfinished pages are not treated as failures.',
            'GT is the downloaded DocLayNet test COCO annotation; viewer image dimensions and rescaled GT boxes are used.',
            'Primary mapping is literal title -> Title (11); the pipeline collapsed doc_title/paragraph_title, so section-only and merged-heading sensitivities are also reported.',
            'residual -> Text; labels without a defined mapping are retained as strict false positives, not dropped.',
            'Matching uses repository src/common/metrics.py: greedy global descending IoU, one-to-one within each image.',
            'Classification accuracy is conditional on localization-only matched boxes, not whole-page or OCR accuracy.',
            'No OCR CER/WER, revision accuracy or final COCO mAP is claimed: transcript/revision GT and calibrated final confidence scores are absent.',
            'This is a small ordered test sample, not the full DocLayNet benchmark.',
        ])
    (output_dir / 'metrics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    with (output_dir / 'per_image.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    categories = modes['0.5']['final_literal']['strict']['per_category']
    with (output_dir / 'per_category.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=['category_id', *next(iter(categories.values())).keys()])
        writer.writeheader()
        writer.writerows(dict(category_id=k, **v) for k, v in categories.items())
    return report


def write_markdown_report(report, output_dir):
    def percent(value):
        return '—' if value is None else f'{100 * value:.2f}%'
    variants = report['metrics']['0.5']
    primary = variants['final_literal']
    strict, loose = primary['strict']['overall'], primary['localization_only']['overall']
    accuracy = primary['classification_on_localized_matches']
    lines = [
        '# Đánh giá tạm thời Dual-Skip trên DocLayNet', '',
        f"Thời điểm chốt snapshot: **{report['snapshot_time']}**.", '',
        f"Đánh giá **{report['image_count']} ảnh đã lưu kết quả thật**, "
        f"**{report['ground_truth_boxes']} bbox ground truth**, **{report['final_boxes']} bbox cuối**. "
        'Không chạy lại suy luận và không can thiệp notebook đang chạy.', '',
        '## Kết quả tại IoU ≥ 0,50', '',
        '| Chỉ số | Kết quả |', '|---|---:|',
        f"| Precision, đúng bbox và nhãn | {percent(strict['precision'])} |",
        f"| Recall, đúng bbox và nhãn | {percent(strict['recall'])} |",
        f"| Micro-F1, đúng bbox và nhãn | {percent(strict['f1'])} |",
        f"| Macro-F1, trung bình các lớp có GT | {percent(strict['macro_f1_gt_present'])} |",
        f"| F1 chỉ xét bbox, bỏ qua nhãn | {percent(loose['f1'])} |",
        f"| Accuracy phân loại trên bbox đã ghép được | {percent(accuracy['accuracy'])} |", '',
        f"Strict: **TP={strict['tp']}, FP={strict['fp']}, FN={strict['fn']}**. "
        f"Accuracy có điều kiện: **{accuracy['correct']}/{accuracy['matched']}** bbox đúng nhãn "
        'trong các cặp đã khớp vị trí; không phải accuracy OCR hoặc accuracy trên toàn bộ tài liệu.', '',
        '## So sánh trước/sau trên cùng snapshot', '',
        '| Kết quả | Precision | Recall | Micro-F1 | F1 chỉ bbox |', '|---|---:|---:|---:|---:|',
    ]
    for key, name in [('baseline_literal', 'PP-DocLayout ban đầu'), ('final_literal', 'Sau Dual-Skip')]:
        value = variants[key]
        s = value['strict']['overall']
        lines.append(f"| {name} | {percent(s['precision'])} | {percent(s['recall'])} | {percent(s['f1'])} | {percent(value['localization_only']['overall']['f1'])} |")
    lines += ['', '## Ảnh hưởng của việc gộp nhãn heading', '',
        'Adapter hiện chuẩn hóa cả `doc_title` và `paragraph_title` thành `title`. '
        'Output đã lưu không còn đủ thông tin để khôi phục chắc chắn hai lớp gốc. '
        'Vì vậy các điểm theo nhãn dưới đây là **tạm thời, phụ thuộc quy đổi**, không chọn quy đổi có điểm cao nhất làm kết quả chính.', '',
        '| Quy ước | Micro-F1 |', '|---|---:|']
    for key, name in [('final_literal', '`title` → Title (quy ước chính theo tên nhãn)'),
                      ('final_title_as_section', '`title` → Section-header (phân tích độ nhạy)'),
                      ('final_headings_merged', 'Gộp Title + Section-header ở cả GT và prediction (taxonomy 10 lớp)')]:
        lines.append(f"| {name} | {percent(variants[key]['strict']['overall']['f1'])} |")
    lines += ['', '## Theo lớp, quy ước chính', '',
              '| Lớp | Số GT | Precision | Recall | F1 |', '|---|---:|---:|---:|---:|']
    for value in primary['strict']['per_category'].values():
        lines.append(f"| {value['name']} | {value['support']} | {percent(value['precision'])} | {percent(value['recall'])} | {percent(value['f1'])} |")
    lines += ['', '## Giới hạn và cách tính', '',
        '- Ghép một-một theo IoU giảm dần bằng evaluator dùng chung của repository, trên đúng ảnh có prediction.',
        '- `residual` được quy đổi thành Text. Nhãn chưa quy đổi vẫn tính FP: ' + json.dumps(report['unmapped_labels'], ensure_ascii=False) + '.',
        '- Trạng thái revision: ' + json.dumps(report['revisions'], ensure_ascii=False) + '. '
        'Nếu chỉ có KEEP/abstained, thay đổi số bbox đến từ residual recovery, chưa phải bằng chứng semantic sửa layout thành công.',
        '- Bộ GT đang dùng là annotation COCO từ mẫu DocLayNet test đã tải, với tọa độ được đổi theo kích thước ảnh thực tế.',
        '- Chưa có GT transcript/revision để tính OCR accuracy, CER/WER hoặc accuracy cho MERGE/SPLIT. Không giả tạo confidence cho bbox cuối để báo mAP.',
        '- Đây là tập mẫu nhỏ theo thứ tự, chưa đại diện toàn bộ DocLayNet. Ảnh đang chạy và chưa lưu không bị tính FN.', '',
        '## File chi tiết', '', '- `metrics.json`: đầy đủ cấu hình, quy đổi nhãn, IoU 0,50/0,75 và danh sách ảnh.',
        '- `per_category.csv`, `per_image.csv`: chỉ số theo lớp và theo ảnh.',
        '- `snapshot/`: bản sao cố định của các prediction đã đánh giá.', '']
    (Path(output_dir) / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, default=ROOT / 'method2/outputs/doclaynet_sample_test/annotations/test.json')
    parser.add_argument('--predictions', type=Path, default=ROOT / 'method2/outputs/notebook')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / 'method2/outputs/evaluation' / datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d_%H%M%S')
    report = evaluate_saved(args.annotations, args.predictions, output)
    write_markdown_report(report, output)
    print(json.dumps({k: report[k] for k in ['snapshot_time', 'image_count', 'ground_truth_boxes', 'baseline_boxes', 'final_boxes', 'unmapped_labels', 'revisions']}, indent=2))
    for name, values in report['metrics']['0.5'].items():
        print(name, json.dumps(dict(strict=values['strict']['overall'], localization=values['localization_only']['overall'],
                                   classification=values['classification_on_localized_matches'])))
    print('REPORT_DIR', output)


if __name__ == '__main__':
    main()
