"""Adaptive request contracts: deterministic crop geometry and per-call budget.

No model download/inference. Fakes replace only adapter model boundary, while
real Experiment cache/signature/evidence behavior remains under test.
"""
from copy import deepcopy
import importlib
import importlib.util
import numpy as np
import pytest
from src.cascaded.types import Config,Line


def config(adaptive=True):
    c=Config(mock=True)
    c.enable_adaptive_inference=adaptive
    return c


def fixture(count=1,sources=('p',)):
    lines=[Line(str(i),[[100,100+i*25],[300,100+i*25],[300,120+i*25],[100,120+i*25]],
                'Several ordinary words in this paragraph.',.95,.95) for i in range(count)]
    return {'id':'t','bbox':[100,100,300,120+max(0,count-1)*25],
            'line_ids':[l.id for l in lines],'source_ids':list(sources),
            'fallback_baseline':[],'reason':'orphan'},lines


def request(target,lines,cfg=None,width=2000,height=2000):
    name='src.cascaded.inference_policy'
    assert importlib.util.find_spec(name) is not None, 'Adaptive inference policy API is missing'
    return importlib.import_module(name).build_inference_request(target,lines,width,height,cfg or config())


def test_legacy_request_retains_context_crop_square_size_and_token_cap():
    target,lines=fixture()
    out=request(target,lines,config(False))
    assert out['crop_bbox']==[60,60,340,160]
    assert tuple(out['layout_size'])==(768,768) and out['max_new_tokens']==512


@pytest.mark.parametrize('count,want',[(1,96),(3,116),(25,732),(40,768)])
def test_adaptive_token_budget_fits_layout_complexity_and_caps_generation(count,want):
    target,lines=fixture(count)
    assert request(target,lines)['max_new_tokens']==want


def test_source_proposal_count_is_unique_and_can_outweigh_line_count():
    target,lines=fixture(1,('a','b','c','d','a'))
    assert request(target,lines)['max_new_tokens']==144


def test_adaptive_crop_uses_tight_line_height_limited_padding_without_mutation():
    target,lines=fixture();before=deepcopy((target,lines))
    out=request(target,lines)
    assert out['crop_bbox']==[80,98,320,122]
    assert (target,lines)==before


@pytest.mark.parametrize('bbox',[(0,0,2,1),(1980,1980,2000,2000),(0,500,2000,501)])
def test_adaptive_edge_and_thin_crops_are_nonempty_in_page_and_grid_budgeted(bbox):
    target,lines=fixture();target['bbox']=list(bbox)
    out=request(target,lines)
    a,b,c,d=out['crop_bbox'];w,h=out['layout_size']
    assert 0<=a<c<=2000 and 0<=b<d<=2000
    assert w>=28 and h>=28 and w%28==0 and h%28==0
    assert w*h<=448*448


def test_adaptive_layout_preserves_wide_crop_aspect_instead_of_forcing_square():
    target,lines=fixture();target['bbox']=[100,100,500,200]
    out=request(target,lines);w,h=out['layout_size'];a,b,c,d=out['crop_bbox']
    assert w>h
    assert abs(w/h-(c-a)/(d-b))<.5
    assert w*h<=448*448


def test_adaptive_profile_has_distinct_cache_identity_from_legacy(tmp_path):
    from src.cascaded.runner import Experiment
    record={'key':'p','document_id':'p','page_index':0,'image':'unused.png','image_id':1}
    old=Experiment(config(False),[record],tmp_path/'out')
    new=Experiment(config(True),[record],tmp_path/'out')
    assert old.signature!=new.signature and old.root!=new.root


@pytest.mark.parametrize('adaptive',[False,True])
def test_runner_supplies_request_overrides_only_for_adaptive_profile(tmp_path,adaptive):
    import cv2
    from src.cascaded.adapters import MockAdapter
    from src.cascaded.runner import Experiment
    image=tmp_path/'page.png'
    cv2.imencode('.png',np.full((400,300,3),255,np.uint8))[1].tofile(image)
    record={'key':'p','document_id':'p','page_index':0,'image':str(image),'image_id':1}
    seen=[]
    class ModelBoundary(MockAdapter):
        def predict(self,image,**kwargs):
            if self.role=='mineru':seen.append(kwargs)
            return super().predict(image)
    exp=Experiment(config(adaptive),[record],tmp_path/'out',ModelBoundary)
    for role in ('paddle','yolo','mineru'):exp.run_phase(role,lambda _:None)
    assert len(seen)==1
    if adaptive:
        assert seen[0]['max_new_tokens']==96
        assert len(seen[0]['layout_image_size'])==2
        assert seen[0]['layout_image_size'][0]*seen[0]['layout_image_size'][1]<=448*448
    else:assert seen==[{}]


@pytest.mark.parametrize('fail',[False,True])
@pytest.mark.parametrize('had_layout_entry',[False,True])
def test_mineru_override_uses_official_helper_settings_and_restores_them(monkeypatch,fail,had_layout_entry):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import inspect
    import sys
    from src.cascaded.adapters import MinerUAdapter
    assert 'max_new_tokens' in inspect.signature(MinerUAdapter.predict).parameters, 'per-call token override missing'
    assert 'layout_image_size' in inspect.signature(MinerUAdapter.predict).parameters, 'per-call aspect override missing'
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(inference_mode=nullcontext))
    # Sampling object carries untouched fields, and restoration must retain
    # the original object identity rather than rebuilding a lossy approximation.
    class Sampling:
        def __init__(self,max_new_tokens=512,**kwargs):
            self.max_new_tokens=max_new_tokens
            self.temperature=0
            self.top_k=1
    monkeypatch.setitem(sys.modules,'mineru_vl_utils.mineru_client',SimpleNamespace(MinerUSamplingParams=Sampling))
    class Tokens:
        def detach(self):return self
        def cpu(self):return self
        def tolist(self):return [1]*96
    class Sequences:
        def __getitem__(self,key):return Tokens()
    class Model:
        generation_config=SimpleNamespace(eos_token_id=9)
        def generate(self,**kwargs):return Sequences()
    adapter=MinerUAdapter(Config(max_tokens=512))
    adapter.model=Model();original_generate=adapter.model.generate
    adapter.processor=SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda *a,**kw:'<|box_start|>0 0 1000 1000<|box_end|><|ref_start|>text<|ref_end|>'))
    original_sampling=Sampling()
    original_helper_size=(768,768)
    seen=[]
    class OfficialClientBoundary:
        def __init__(self):
            self.sampling_params={'[layout]':original_sampling} if had_layout_entry else {}
            self.helper=SimpleNamespace(layout_image_size=original_helper_size)
        def layout_detect(self,image,priority=None,scored=None):
            active=self.sampling_params.get('[layout]',original_sampling)
            seen.append((active.max_new_tokens,self.helper.layout_image_size,image.mode))
            if fail:raise RuntimeError('synthetic model failure')
            adapter.model.generate(input_ids=np.zeros((1,2)),max_new_tokens=active.max_new_tokens)
            return [SimpleNamespace(bbox=[0,0,1,1],type='text')]
    adapter.client=OfficialClientBoundary()
    output=adapter.predict(np.zeros((40,300,3),np.uint8),max_new_tokens=96,layout_image_size=(448,56))
    assert seen==[(96,(448,56),'RGB')]
    assert adapter.client.helper.layout_image_size==original_helper_size
    assert ('[layout]' in adapter.client.sampling_params)==had_layout_entry
    if had_layout_entry:assert adapter.client.sampling_params['[layout]'] is original_sampling
    assert adapter.model.generate==original_generate
    if fail:assert output['status']=='error' and 'synthetic model failure' in output['error']
    else:assert output['tokens']==96 and output['status']=='incomplete'


def test_invalid_request_before_model_call_does_not_count_as_inference_or_poison_tokens():
    from src.cascaded.report import timing_report
    result={'lines':[],'stages':{'ensemble':[],'recovery':[],'final':[]},
            'timing':{'mineru':{'inference_seconds':.001,'peak_memory':None}},
            'evidence':[{'status':'invalid_request','tokens':None,'blocks':[]}]}
    report=timing_report([result])
    assert report['mineru_crop_calls']==0
    assert report['mineru_tokens']==0


@pytest.mark.parametrize('status,tokens',[('ok',12),('incomplete',512),('invalid_output',3),('error',None)])
def test_real_model_calls_keep_their_timing_and_unknown_token_accounting(status,tokens):
    from src.cascaded.report import timing_report
    result={'lines':[],'stages':{'ensemble':[],'recovery':[],'final':[]},
            'timing':{'mineru':{'inference_seconds':.001,'peak_memory':None}},
            'evidence':[{'status':status,'tokens':tokens,'blocks':[]}]}
    report=timing_report([result])
    assert report['mineru_crop_calls']==1
    assert report['mineru_tokens']==tokens


def _instrumented_adapter(monkeypatch,tokens,fail_after_generate=False):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import sys
    from src.cascaded.adapters import MinerUAdapter
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(inference_mode=nullcontext))
    class Tokens:
        def detach(self):return self
        def cpu(self):return self
        def tolist(self):return tokens
    class Sequences:
        def __getitem__(self,key):return Tokens()
    class Model:
        generation_config=SimpleNamespace(eos_token_id=9)
        def generate(self,**kwargs):return Sequences()
    adapter=MinerUAdapter(Config(max_tokens=512));adapter.model=Model()
    adapter.processor=SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda *a,**kw:'<|box_start|>0 0 1000 1000<|box_end|><|ref_start|>text<|ref_end|>'))
    class Client:
        sampling_params={'[layout]':SimpleNamespace(max_new_tokens=512)}
        helper=SimpleNamespace(layout_image_size=(768,768))
        def __init__(self):self.seen=[]
        def layout_detect(self,image,priority=None,scored=None):
            self.seen.append((self.sampling_params['[layout]'].max_new_tokens,self.helper.layout_image_size))
            adapter.model.generate(input_ids=np.zeros((1,2)))
            if fail_after_generate:raise RuntimeError('post-generation API failure')
            return [SimpleNamespace(bbox=[0,0,1,1],type='text')]
    adapter.client=Client()
    return adapter


@pytest.mark.parametrize('cap,tokens,fail_after,want',[
    (96,[1]*95+[9],False,'ok'),
    (768,[1]*512,False,'ok'),
    (96,[1]*96,True,'error'),
    (None,[1]*96,True,'invalid_output'),
])
def test_effective_generation_cap_eos_and_postgeneration_error_precedence(monkeypatch,cap,tokens,fail_after,want):
    adapter=_instrumented_adapter(monkeypatch,tokens,fail_after)
    crop=np.zeros((40,300,3),np.uint8)
    output=adapter.predict(crop,max_new_tokens=cap,layout_image_size=(448,56)) if cap else adapter.predict(crop)
    assert output['status']==want and output['tokens']==len(tokens)
    if fail_after:assert 'post-generation API failure' in output['error']


def test_adaptive_call_followed_by_legacy_call_restores_cap_and_size_without_settings_leak(monkeypatch):
    adapter=_instrumented_adapter(monkeypatch,[1]*96)
    crop=np.zeros((40,300,3),np.uint8)
    first=adapter.predict(crop,max_new_tokens=96,layout_image_size=(448,56))
    second=adapter.predict(crop)
    assert first['status']=='incomplete' and second['status']=='ok'
    assert adapter.client.seen==[(96,(448,56)),(512,(768,768))]
    assert 'inference_override' not in second


def test_unsupported_adaptive_client_gets_full_legacy_crop_positional_call_and_visible_metadata(tmp_path):
    import cv2
    from src.cascaded.adapters import MockAdapter
    from src.cascaded.runner import Experiment
    image=tmp_path/'page.png'
    cv2.imencode('.png',np.full((400,300,3),255,np.uint8))[1].tofile(image)
    record={'key':'p','document_id':'p','page_index':0,'image':str(image),'image_id':1}
    seen=[]
    class UnsupportedClient(MockAdapter):
        def inference_override_capability(self):return False,'test_unsupported_client_settings'
        def predict(self,image):
            if self.role=='mineru':seen.append(image.shape)
            return super().predict(image)
    exp=Experiment(config(True),[record],tmp_path/'out',UnsupportedClient)
    for role in ('paddle','yolo','mineru'):exp.run_phase(role,lambda _:None)
    evidence=exp.results()[0]['evidence'][0]
    assert seen==[(100,282,3)]
    assert evidence['crop_bbox']==[0,200,282,300]
    assert evidence['adaptive_inference_requested']
    assert evidence['requested_inference_request']['max_new_tokens']==96
    assert evidence['inference_request']['max_new_tokens']==512
    assert evidence['inference_effective']['applied'] is False
    assert tuple(evidence['inference_effective']['effective_layout_image_size'])==(768,768)
    assert 'unsupported' in evidence['inference_fallback_reason']
