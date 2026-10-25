import json
import pytest
from src.evaluation.evaluate_refinement import run_ablation_suite,ABLATIONS
from src.evaluation.evaluate_detection import compute_detection_metrics
from src.evaluation.evaluate_parsing import compute_parsing_metrics


def test_ablation_uses_measured_predictions_and_requires_missing_runs(tmp_path):
    record = dict(ground_truth=dict(revisions=[],reading_order=['a'],text='hello',regions=[dict(bbox=[0,0,20,20],label='text')]),
        predictions=dict(A9=dict(revisions=[],final_regions=[dict(id='a',bbox_final=[0,0,20,20],label_final='text')],markdown='hello',glm_requests=2)))
    (tmp_path/'page.json').write_text(json.dumps(record),encoding='utf-8')
    result = run_ablation_suite(tmp_path,['A9'])['A9']
    assert result['reading_order_ned'] == 1 and result['cer'] == 0
    assert result['detection_f1'] == 1 and result['glm_requests'] == 2
    with pytest.raises(ValueError,match='missing measured'):
        run_ablation_suite(tmp_path,['A0'])
    with pytest.raises(ValueError,match='unknown ablation'):
        run_ablation_suite(tmp_path,['A20'])


def test_missing_gt_does_not_fabricate_order_accuracy(tmp_path):
    record = dict(ground_truth=dict(revisions=[]),predictions=dict(A9=dict(revisions=[])))
    (tmp_path/'page.json').write_text(json.dumps(record),encoding='utf-8')
    assert 'reading_order_ned' not in run_ablation_suite(tmp_path,['A9'])['A9']


def test_detection_counts_duplicates_and_wrong_labels():
    gt = [dict(bbox=[0,0,20,20],label='table')]
    assert compute_detection_metrics(gt+gt,gt)['precision'] == .5
    assert compute_detection_metrics([dict(bbox=[0,0,20,20],label='text')],gt)['recall'] == 0
    assert compute_parsing_metrics('abc','abc') == dict(cer=0,wer=0)


def test_all_ablation_configurations_are_defined():
    assert set(ABLATIONS) == {f'A{i}' for i in range(10)}
    assert ABLATIONS['A7']['validation']['use_skip_l'] is False
    assert ABLATIONS['A8']['validation']['use_skip_h'] is False
