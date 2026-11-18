"""Meaningful replay audit and literal metric fixtures; no model dependency."""
from pathlib import Path
import importlib.util
import pytest


def replay_module():
    script=Path(__file__).resolve().parents[2]/'scratch/fast_cascaded_v3/replay_evaluation.py'
    assert script.is_file(), 'paired replay verification harness missing'
    spec=importlib.util.spec_from_file_location('fast_cascaded_replay',script)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def test_replay_audit_rejects_duplicate_ownership_and_lost_baseline_lines():
    audit=replay_module().audit_line_accounting
    baseline=[{'id':'p','line_ids':['a']}]
    with pytest.raises(AssertionError,match='duplicate'):
        audit(baseline,[{'id':'p','line_ids':['a']},{'id':'q','line_ids':['a']}],['a'],[])
    with pytest.raises(AssertionError,match='baseline'):
        audit(baseline,[],['a'],[{'line_ids':['a']}])
    audit(baseline,baseline,['a','b'],[{'line_ids':['b']}])


def test_literal_evaluator_keeps_unknown_as_fp_and_distinguishes_iou_thresholds():
    evaluate=replay_module().evaluate_final_predictions
    annotations={'images':[{'id':1,'width':100,'height':100}],
                 'categories':[{'id':10,'name':'Text'}],
                 'annotations':[{'id':1,'image_id':1,'category_id':10,'bbox':[0,0,100,100]}]}
    results=[{'record':{'image_id':1},'stages':{'final':[
        {'id':'p','bbox':[0,0,100,60],'label':'text','source':'mineru'},
        {'id':'u','bbox':[80,80,90,90],'label':'unknown','source':'mineru'}]}}]
    out=evaluate(results,annotations)
    half=out['0.5']['strict']['overall'];high=out['0.75']['strict']['overall']
    assert (half['tp'],half['fp'],half['fn'])==(1,1,0)
    assert half['f1']==pytest.approx(2/3)
    assert (high['tp'],high['fp'],high['fn'])==(0,2,1)
    assert out['0.5']['strict']['per_category'][0]['fp']==1
    assert out['0.5']['source_errors']['fp_by_source']=={'mineru':1}
