"""Geometry, model boundary and reproducible notebook regressions; no weights needed."""
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
import pytest
from src.cascaded.types import Config, Proposal, Line, category, label_for
from src.cascaded.core import (coverage, prepare, finish, context_box, to_page_bbox, deduplicate)
from src.cascaded.runner import Experiment, write_json


def line(id='l', bbox=(10, 10, 80, 20)):
    a, b, c, d = bbox
    return Line(id, [[a,b],[c,b],[c,d],[a,d]], 'OCR', .9)


def proposal(id='p', bbox=(0, 0, 100, 100), label='text', score=.9, source='paddle'):
    return Proposal(id, list(bbox), label, source, label, score)


def test_coverage_is_line_area_not_iou_and_visuals_survive():
    ls = [line()]
    assert coverage(ls[0], [0,0,1000,1000]) == 1
    first, second, targets = prepare([proposal(bbox=(0,0,1000,1000)),
        proposal('fig', (1100,0,1200,200), 'image')], ls, Config())
    assert next(p for p in first if p.id == 'p').line_ids == ['l'] and len(first) == 2 and not targets
    assert first == second


def test_dual_threshold_and_weak_recovery():
    ps = [proposal('ignored', (0,0,90,30), score=.09), proposal('weak', (0,0,90,30), score=.1)]
    first, second, targets = prepare(ps, [line()], Config())
    assert first == [] and second[0].id == 'weak' and second[0].stage == 'recovery' and not targets
    assert ps[1].line_ids == []  # immutable inputs


def test_nms_tie_prefers_paddle_and_keeps_lineage():
    ps = [proposal('y', source='yolo'), proposal('p')]
    kept = deduplicate(ps, Config())
    assert len(kept) == 1 and kept[0].id == 'p' and set(kept[0].source_ids) == {'p','y'}


def test_cross_label_conflict_is_not_decided_by_confidence():
    first, second, targets = prepare([proposal(), proposal('q', label='paragraph_title', score=.5)], [line()], Config())
    assert not first and not second and targets[0]['reason'] == 'detector_conflict'
    final, audit = finish(second, targets, [line()], {}, Config())
    assert len(final) == 1 and final[0].label == 'unknown' and final[0].needs_review


def test_unique_ownership_smaller_bbox_wins():
    ps = [proposal('big'), proposal('small', (0,0,90,30), 'caption')]
    first, _, _ = prepare(ps, [line()], Config())
    assert next(p for p in first if p.id == 'small').line_ids == ['l']
    assert next(p for p in first if p.id == 'big').line_ids == []


def test_adjacent_recovery_and_two_column_guard():
    ls = [line('a', (10,10,90,20)), line('b', (10,25,90,35)), line('right', (150,25,220,35))]
    _, second, targets = prepare([proposal(bbox=(10,10,90,20))], ls, Config())
    assert second[0].bbox == [10,10,90,35]
    assert set(second[0].line_ids) == {'a','b'}
    assert targets[0]['line_ids'] == ['right']


def test_multiple_adjacent_owners_abstain():
    ls = [line('a', (10,0,80,10)), line('b', (10,30,80,40)), line('middle', (10,15,80,25))]
    _, second, targets = prepare([proposal('a',(10,0,80,10)), proposal('b',(10,30,80,40))], ls, Config())
    assert sum(len(p.line_ids) for p in second) == 2
    assert targets[0]['line_ids'] == ['middle']


def test_visual_blocks_bbox_expansion():
    ls = [line('a',(10,0,80,10)), line('b',(10,15,80,25))]
    _, second, _ = prepare([proposal('p',(10,0,80,10)), proposal('fig',(0,12,100,14),'image')], ls, Config())
    assert next(p for p in second if p.id == 'p').bbox == [10,0,80,10]


def test_context_crop_clamp_and_coordinate_roundtrip():
    t = {'bbox': [0,0,80,20], 'line_ids': ['l']}
    crop = context_box(t, [line()], 100, 100)
    assert crop == [0,0,100,40]
    assert to_page_bbox([.1,.25,.8,.5], crop) == [10,10,80,20]
    with pytest.raises(ValueError):
        to_page_bbox([0,0,1.1,1], crop)


@pytest.mark.parametrize('status', ['error','incomplete','invalid_output','budget_exhausted'])
def test_failed_crop_keeps_unknown_and_line(status):
    _, second, targets = prepare([], [line()], Config())
    final, _ = finish(second, targets, [line()], {targets[0]['id']: {'status': status, 'blocks': []}}, Config())
    assert final[0].line_ids == ['l'] and final[0].label == 'unknown'


def test_mineru_acceptance_excludes_context_and_preserves_ocr():
    ls = [line()]
    _, second, targets = prepare([], ls, Config())
    prediction = {'status': 'ok', 'crop_bbox': [0,0,100,100],
                  'blocks': [{'bbox':[.1,.1,.8,.2], 'label':'paragraph_title'},
                             {'bbox':[0,.5,1,.8], 'label':'text'}]}
    final, _ = finish(second, targets, ls, {targets[0]['id']:prediction}, Config())
    assert len(final) == 1 and final[0].label == 'paragraph_title' and final[0].score is None
    assert final[0].bbox == [10,10,80,20] and ls[0].text == 'OCR'


def test_malformed_mineru_bbox_and_conflicting_labels_rejected():
    ls = [line()]
    _, second, targets = prepare([], ls, Config())
    for blocks in ([{'bbox':[0,0,2,1], 'label':'text'}],
                   [{'bbox':[0,0,1,1], 'label':label} for label in ('text','caption')]):
        final, _ = finish(second, targets, ls, {targets[0]['id']:{'status':'ok', 'crop_bbox':[0,0,100,100],
                                                                'blocks':blocks}}, Config())
        assert final[0].label == 'unknown'


def test_heading_mapping_is_not_collapsed():
    assert label_for('doc_title','paddle') == 'doc_title'
    assert category('doc_title') == 11 and category('paragraph_title') == 8
    assert category('title') == 0 and category('title', True) == 8


def test_mineru_adapter_calls_layout_only_counts_tokens_and_restores_generate(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import sys
    from src.cascaded.adapters import MinerUAdapter
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(inference_mode=nullcontext))
    class Tokens:
        def detach(self): return self
        def cpu(self): return self
        def tolist(self): return [1,2,3]
    class Sequences:
        def __getitem__(self, key): return Tokens()
    class Model:
        generation_config = SimpleNamespace(eos_token_id=9)
        def generate(self, **kwargs): return Sequences()
    adapter = MinerUAdapter(Config(max_tokens=3))
    adapter.model = Model()
    original = adapter.model.generate
    adapter.processor = SimpleNamespace(tokenizer=SimpleNamespace(
        decode=lambda *a, **kw:'<|box_start|>0 0 1000 1000<|box_end|><|ref_start|>text<|ref_end|>'))
    calls = []
    class Client:
        def layout_detect(self, image):
            calls.append(image.mode)
            adapter.model.generate(input_ids=np.zeros((1,2)))
            return [SimpleNamespace(bbox=[0,0,1,1], type='text')]
    adapter.client = Client()
    result = adapter.predict(np.zeros((50,50,3), np.uint8))
    assert calls == ['RGB'] and result['tokens'] == 3 and result['status'] == 'incomplete'
    assert adapter.model.generate == original


def test_mineru_cannot_merge_separate_columns():
    ls = [line('left',(0,0,50,10)), line('right',(100,0,150,10))]
    target = {'id':'t','bbox':[0,0,150,10],'line_ids':['left','right'],'source_ids':[]}
    final, _ = finish([], [target], ls, {'t':{'status':'ok','crop_bbox':[0,0,150,10],
        'blocks':[{'bbox':[0,0,1,1],'label':'text'}]}}, Config())
    assert len(final) == 2 and all(p.label == 'unknown' for p in final)


def test_phase_lifecycle_cache_budget_and_image_change(tmp_path):
    import cv2
    from src.cascaded.adapters import MockAdapter
    image = tmp_path/'ảnh.png'
    cv2.imencode('.png', np.full((400,300,3),255,np.uint8))[1].tofile(image)
    records = [{'key':'page','image':str(image),'document_id':'page','page_index':0,'image_id':1}]
    events = []
    class Tracked(MockAdapter):
        def initialize(self):
            events.append('load:'+self.role)
        def close(self):
            events.append('close:'+self.role)
    experiment = Experiment(Config(mock=True), records, tmp_path/'outputs', Tracked)
    for phase in ('paddle','yolo','mineru'):
        experiment.run_phase(phase, lambda x:None)
    assert events == ['load:paddle','close:paddle','load:yolo','close:yolo','load:mineru','close:mineru']
    result = experiment.results()[0]
    assert result['mode'] == 'MOCK' and result['schema_version']=='cascaded-layout-v2.1'
    assert result['review_evidence'] and all(p['label'] != 'unknown' for p in result['final_detections'])
    before = len(events)
    for phase in ('paddle','yolo','mineru'):
        experiment.run_phase(phase, lambda x:None)
    assert len(events) == before
    cv2.imencode('.png', np.zeros((400,300,3),np.uint8))[1].tofile(image)
    assert experiment.cached(records[0], 'paddle') is None


def test_no_targets_means_no_mineru_load(tmp_path):
    import cv2
    from src.cascaded.adapters import MockAdapter
    image = tmp_path/'page.png'
    cv2.imencode('.png', np.zeros((100,100,3),np.uint8))[1].tofile(image)
    record = {'key':'p','image':str(image),'document_id':'p','page_index':0,'image_id':1}
    class Adapter(MockAdapter):
        def initialize(self):
            assert self.role != 'mineru'
        def predict(self, image):
            return {'proposals':[], 'lines':[]} if self.role == 'paddle' else {'proposals':[]}
    exp = Experiment(Config(mock=True), [record], tmp_path/'out', Adapter)
    for phase in ('paddle','yolo','mineru'):
        exp.run_phase(phase, lambda x:None)
    assert exp.results()[0]['evidence'] == []


def test_zero_budget_never_loads_mineru_and_oom_is_evidence(tmp_path):
    import cv2
    from src.cascaded.adapters import MockAdapter
    image = tmp_path/'page.png'
    cv2.imencode('.png', np.zeros((400,300,3),np.uint8))[1].tofile(image)
    record = {'key':'p','image':str(image),'document_id':'p','page_index':0,'image_id':1}
    class Adapter(MockAdapter):
        def initialize(self):
            if self.role == 'mineru' and self.cfg.max_crops == 0:
                raise AssertionError('must not load')
        def __init__(self, cfg, role):
            super().__init__(cfg, role)
            self.cfg = cfg
        def predict(self, image):
            if self.role == 'mineru':
                raise RuntimeError('CUDA out of memory')
            return super().predict(image)
    for budget in (0, 1):
        exp = Experiment(Config(mock=True, max_crops=budget), [record], tmp_path/f'out{budget}', Adapter)
        for phase in ('paddle','yolo','mineru'):
            exp.run_phase(phase, lambda x: None)
        result = exp.results()[0]
        assert all(p['label'] != 'unknown' for p in result['stages']['final'])
        assert any(r['reason']=='orphan_abstained' for r in result['review_evidence'])
        assert result['evidence'][0]['status'] == ('error' if budget else 'budget_exhausted')
        if budget:
            assert 'out of memory' in result['evidence'][0]['error']


def test_report_same_taxonomy_and_comparison_rejects_different_images(tmp_path):
    from src.cascaded.report import evaluate, compare
    r = {'mode':'REAL','width':100,'height':100,'record':{'image_id':1}, 'config':{'variant':'s'},
         'image_sha256':'abc', 'lines':[], 'evidence':[], 'timing':{'paddle':{'inference_seconds':.1}},
         'stages':{name:[asdict(proposal(label='doc_title'))] for name in ('paddle','ensemble','recovery','final')}}
    annotation = {'images':[{'id':1,'file_name':'1.png','width':100,'height':100}],
                  'annotations':[{'id':1,'image_id':1,'bbox':[0,0,100,100],'category_id':11}],
                  'categories':[{'id':11,'name':'Title'}]}
    write_json(tmp_path/'gt.json', annotation)
    report = evaluate([r], tmp_path/'gt.json', tmp_path/'report')
    assert report['rows'][0]['f1'] == 1
    assert report['details']['final']['ap'] is None
    report['variant'] = 'v3'
    report['sources'][0]['image_sha256'] = 'different'
    write_json(tmp_path/'v3.json', report)
    with pytest.raises(ValueError, match='datasets'):
        compare([tmp_path/'report/metrics.json',tmp_path/'v3.json'],tmp_path/'compare')
    r['mode'] = 'MOCK'
    with pytest.raises(ValueError, match='REAL'):
        evaluate([r], tmp_path/'gt.json', tmp_path/'bad')


@pytest.mark.parametrize('variant', ['S','V3'])
def test_notebook_fresh_kernel_mock(tmp_path, variant):
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client.kernelspec import KernelSpecManager
    import sys
    root = Path(__file__).resolve().parents[1]
    nb = nbformat.read(root/f'Cascaded_Layout_{variant}_YOLO_MinerU.ipynb', as_version=4)
    for cell in nb.cells:
        if cell.cell_type == 'code' and cell.id == 'config':
            cell.source = cell.source.replace('MOCK_MODE = False', 'MOCK_MODE = True')
            cell.source = cell.source.replace('RUN_COUNT = 20', 'RUN_COUNT = 1')
            cell.source += f'\nOUTPUT_ROOT = Path({str(tmp_path / "outputs")!r})\n'
    kernel = tmp_path/'kernels/cascade-test'
    kernel.mkdir(parents=True)
    write_json(kernel/'kernel.json', {'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
                                     'display_name':'Cascade test','language':'python'})
    client = NotebookClient(nb, timeout=180, kernel_name='cascade-test',
        resources={'metadata':{'path':str(root)}})
    client.km = __import__('jupyter_client').KernelManager(kernel_name='cascade-test',
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(tmp_path/'kernels')]))
    client.execute()
    nbformat.write(nb, tmp_path/f'{variant}_executed_MOCK.ipynb')
    assert any('MOCK' in str(c.get('outputs',[])) for c in nb.cells)
