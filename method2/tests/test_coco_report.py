import json

import pytest

from src.evaluation.coco_report import evaluate_coco_snapshot, _final_score


def fixture_snapshot(tmp_path, *, only_unmapped=False, missing_line_score=False):
    annotation = tmp_path / 'gt.json'
    annotation.write_text(json.dumps(dict(images=[dict(id=1, file_name='1.png', width=100, height=100)],
        categories=[dict(id=10, name='Text')], annotations=[dict(id=1, image_id=1,
        category_id=10, bbox=[10, 10, 20, 20])])), encoding='utf-8')
    original = [dict(id='b', bbox=[10, 10, 30, 30], label='text', score=.9),
                dict(id='n', bbox=[80, 80, 90, 90], label='number', score=.6)]
    final = [dict(id='b', bbox_final=[10, 10, 30, 30], label_final='text', source_region_ids=['b']),
             dict(id='r', bbox_final=[50, 50, 60, 60], label_final='text',
                  source_region_ids=['recovered:0'], source_line_ids=['L1']),
             dict(id='n', bbox_final=[80, 80, 90, 90], label_final='number', source_region_ids=['n'])]
    if only_unmapped:
        original, final = original[-1:], final[-1:]
    folder = tmp_path / 'snapshot/1'
    folder.mkdir(parents=True)
    result = dict(original_regions=original, final_regions=final,
        recovered_regions=[dict(id='recovered:0', score=1.)],
        original_lines=[] if missing_line_score else [dict(line_id='L1', score=.7)],
        evidence=[dict(parser='glm_local')])
    (folder / 'result.json').write_text(json.dumps(result), encoding='utf-8')
    return dict(annotation_file=str(annotation), sources=[dict(image_id=1)]), folder.parent


def test_coco_ap_uses_measured_scores_and_strict_f1_keeps_unmapped(tmp_path):
    pytest.importorskip('pycocotools')
    report, snapshot = fixture_snapshot(tmp_path)
    output, table, per_class = evaluate_coco_snapshot(report, snapshot, tmp_path / 'output')
    final = output['modes']['final']
    assert final['coco']['overall']['map50'] == pytest.approx(1.)
    assert final['strict']['overall']['f1'] == pytest.approx(.5)
    assert final['strict']['overall']['fp'] == 2
    assert final['ap_excluded_unmapped'] == {'number': 1}
    assert final['score_origins'] == {'inherited_layout_min': 1, 'inherited_detector_min': 1}
    assert not final['missing_scores']
    assert table.iloc[1]['AP50 (%)'] == pytest.approx(100.)
    assert table.iloc[1]['Strict Micro-F1 (%)'] == pytest.approx(50.)
    assert (tmp_path / 'output/coco_metrics.json').is_file() and not per_class.empty
    assert _final_score(dict(source_region_ids=['recovered:0'], source_line_ids=['L1']), {}, {'L1': .7})[0] == .7


def test_missing_measured_score_does_not_fabricate_ap_or_drop_predictions(tmp_path):
    report, snapshot = fixture_snapshot(tmp_path, missing_line_score=True)
    output, _, _ = evaluate_coco_snapshot(report, snapshot, tmp_path / 'output')
    final = output['modes']['final']
    assert final['coco'] is None and len(final['missing_scores']) == 1
    assert final['strict']['overall']['fp'] == 2
    assert 'no score is fabricated' in final['ap_error']


def test_shared_coco_map_handles_no_mapped_predictions(tmp_path):
    pytest.importorskip('pycocotools')
    report, snapshot = fixture_snapshot(tmp_path, only_unmapped=True)
    output, _, _ = evaluate_coco_snapshot(report, snapshot, tmp_path / 'output')
    for mode in output['modes'].values():
        assert mode['coco']['overall']['map50'] == 0
        assert mode['strict']['overall']['f1'] == 0
        assert mode['strict']['overall']['fp'] == 1
        assert mode['strict']['overall']['fn'] == 1
