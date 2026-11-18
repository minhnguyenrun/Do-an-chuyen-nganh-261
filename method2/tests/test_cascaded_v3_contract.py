"""V3 public optional features: real-format contracts, no model loading."""
from copy import deepcopy
import importlib
import importlib.util
import json
import numpy as np
import pytest
from shapely.geometry import Polygon,box
from src.cascaded import types
from src.cascaded.types import Proposal,Config,Line


def normalize(raw,variant='v3',proposal_id='paddle:7'):
    name='src.cascaded.v3_features'
    assert importlib.util.find_spec(name) is not None,'V3 public feature normalizer is missing'
    return importlib.import_module(name).normalize_paddle_box(raw,width=100,height=100,variant=variant,proposal_id=proposal_id)


def serialize(p):
    fn=getattr(types,'serialize_proposal',None)
    assert callable(fn),'backward-compatible proposal serializer is missing'
    return fn(p)


def raw_box(**extra):
    raw={'coordinate':[10.7,20.2,90.2,80.1],'label':'figure_title','score':.81,
         'order':2,'polygon_points':[[9.5,21.5],[90.75,20.25],[90.5,80.5],[10.25,80.75]]}
    raw.update(extra);return raw


def legacy_payload():
    return {'id':'p','bbox':[10,10,110,30],'label':'text','source':'paddle','raw_label':'text',
            'score':.8,'line_ids':['l'],'source_ids':['p'],'stage':'detector','reasons':[],'needs_review':False}


def test_v3_public_order_polygon_pairing_preserves_fractional_page_geometry():
    raw=raw_box();before=deepcopy(raw)
    p,audit=normalize(raw)
    assert p.id=='paddle:7' and p.reading_order_rank==1
    assert p.bbox==[10,20,91,81] and p.label=='caption' and p.score==.81
    assert Polygon(p.polygon).equals(Polygon(raw['polygon_points']))
    assert set(map(tuple,p.polygon))==set(map(tuple,raw['polygon_points'])) or set(map(tuple,p.polygon[:-1]))==set(map(tuple,raw['polygon_points']))
    assert raw==before
    assert 'order_status' in audit and 'polygon_status' in audit
    json.dumps({'proposal':serialize(p),'audit':audit},allow_nan=False)


def test_v3_skipped_public_order_stays_none_without_index_or_coordinate_fallback():
    p,_=normalize(raw_box(order=None),proposal_id='paddle:99')
    assert p.reading_order_rank is None and p.polygon is not None


@pytest.mark.parametrize('order',[True,False,0,-1,1.0,1.5,float('nan'),float('inf'),'2'])
def test_invalid_order_abstains_and_audit_is_strict_json_safe(order):
    p,audit=normalize(raw_box(order=order))
    assert p is not None and p.reading_order_rank is None
    assert audit['reasons']
    json.dumps({'proposal':serialize(p),'audit':audit},allow_nan=False)


@pytest.mark.parametrize('points',[
    [[10,10],[90,90],[10,90],[90,10]],
    [[10,10],[10,10],[10,10]],
    [[10,10],[90,float('nan')],[90,90],[10,90]],
    [[10,10],[30,10],[30,120],[70,120],[70,10],[90,10],[90,140],[10,140]],
])
def test_invalid_or_multicomponent_polygon_abstains_without_fabricating_bbox_mask(points):
    p,audit=normalize(raw_box(polygon_points=points))
    assert p is not None and p.polygon is None and p.reading_order_rank==1
    assert audit['reasons']
    json.dumps({'proposal':serialize(p),'audit':audit},allow_nan=False)


def test_single_component_polygon_clips_to_page_without_replacing_with_bbox():
    points=[[-10,10],[80,10],[80,120],[-10,120]]
    p,_=normalize(raw_box(coordinate=[-10,-5,80,120],polygon_points=points))
    assert p.bbox==[0,0,80,100]
    assert Polygon(p.polygon).equals(box(0,10,80,100))


def test_numpy_public_scalars_and_arrays_never_break_strict_cache_json():
    p,audit=normalize(raw_box(coordinate=np.array([10.7,20.2,90.2,80.1]),
                            polygon_points=np.array([[10.5,21.5],[90.5,21.5],[90.5,80.5],[10.5,80.5]]),
                            score=np.float32(.81),order=np.int64(2)))
    assert p is not None
    assert p.reading_order_rank==1
    json.dumps({'proposal':serialize(p),'audit':audit},allow_nan=False)


def test_missing_public_features_abstain_with_explicit_audit_reasons():
    raw=raw_box();raw.pop('order');raw.pop('polygon_points')
    p,audit=normalize(raw)
    assert p is not None and p.reading_order_rank is None and p.polygon is None
    assert audit['order_status'] and audit['polygon_status'] and audit['reasons']
    json.dumps({'proposal':serialize(p),'audit':audit},allow_nan=False)


def test_s_ignores_synthesized_order_and_layout_polygon():
    p,_=normalize(raw_box(),variant='s')
    assert p is not None and p.reading_order_rank is None and p.polygon is None


def test_old_proposal_roundtrip_keeps_legacy_dictionary_exact():
    payload=legacy_payload();p=Proposal(**payload)
    assert serialize(p)==payload
    assert serialize(Proposal(**serialize(p)))==payload


def test_old_positional_proposal_constructor_keeps_all_prior_argument_positions():
    p=Proposal('p',[10,10,110,30],'text','paddle','text',.8,['l'],['source'],'strong',['why'],True)
    assert p.line_ids==['l'] and p.source_ids==['source'] and p.stage=='strong'
    assert p.reasons==['why'] and p.needs_review is True
    assert getattr(p,'reading_order_rank','missing') is None
    assert getattr(p,'polygon','missing') is None


def test_measured_rank_zero_and_polygon_survive_serialization_roundtrip():
    p,_=normalize(raw_box(order=1))
    payload=serialize(p)
    assert payload['reading_order_rank']==0 and payload['polygon']
    restored=Proposal(**payload)
    assert serialize(restored)==payload


def test_phase2_features_default_off_without_enabling_failed_phase1_profiles():
    cfg=Config()
    for name in ('enable_v3_polygons','enable_v3_reading_order','enable_semantic_arbitration','enable_semantic_quality',
                 'enable_geometry_refinement','enable_adaptive_inference'):
        assert getattr(cfg,name,'missing') is False
    assert getattr(cfg,'semantic_artifact_path','missing') is None


def test_legacy_conflict_fallback_and_decision_dicts_omit_new_none_fields():
    from src.cascaded.fusion import prepare_layout
    from src.cascaded.attribution import finish_layout
    p=Proposal(**legacy_payload())
    q=Proposal('y',[10,10,110,30],'caption','yolo','caption',.65)
    p.score=.6
    line=Line('l',[[10,10],[110,10],[110,30],[10,30]],'Figure 1. Revenue',.9,.9)
    state=prepare_layout([p,q],[line],Config(),1000,1000)
    expected=legacy_payload();expected.update(score=.6,stage='strong')
    assert state['targets'][0]['fallback_baseline']==[expected]
    _,_,_,decisions=finish_layout(state,[line],{},Config(),1000,1000)
    assert decisions[-1]['baseline']==[expected]


def test_adapter_worker_preserves_real_format_features(monkeypatch):
    """Replacing only the model boundary leaves the actual child loop running.

    Dropping coordinate/order/polygon in the adapter, independently sorting
    them, or gating evidence capture behind consumer flags must fail here.
    """
    import sys
    from types import SimpleNamespace
    from src.cascaded.adapters import _paddle_child

    rows=[raw_box(order=2),raw_box(order=None,coordinate=[20,1,40,9],
          polygon_points=[[20,1],[40,1],[40,9],[20,9]]),
          raw_box(order=1,coordinate=[1,80,8,99],
          polygon_points=[[1,80],[8,80],[8,99],[1,99]])]
    class PredictorBoundary:
        def process(self, image, layout_shape_mode='auto'):
            raise AssertionError('metadata must inspect the declared signature only')
    class LayoutBoundary:
        def __init__(self,**kwargs):
            assert kwargs['model_name']=='PP-DocLayoutV3'
            self.paddlex_predictor = PredictorBoundary()
        def predict(self,image,**kwargs):
            return [{'res':{'boxes':deepcopy(rows)}}]
    class DetectorBoundary:
        def __init__(self,**kwargs):pass
        def predict(self,image):
            return [{'res':{'dt_polys':[],'dt_scores':[]}}]
    class RecognizerBoundary:
        def __init__(self,**kwargs):pass
        def predict(self,crops):
            raise AssertionError('empty OCR response must not invoke recognition')
    monkeypatch.setitem(sys.modules,'src.adapters.paddle_worker',
                        SimpleNamespace(_prepare_paddle_process=lambda:None))
    monkeypatch.setitem(sys.modules,'paddleocr',SimpleNamespace(
        LayoutDetection=LayoutBoundary,TextDetection=DetectorBoundary,
        TextRecognition=RecognizerBoundary,__version__='fixture-ocr'))
    monkeypatch.setitem(sys.modules,'paddle',SimpleNamespace(__version__='fixture-paddle'))
    class Pipe:
        def __init__(self):self.sent=[];self.inputs=iter([np.zeros((100,100,3),np.uint8),None]);self.closed=False
        def recv(self):return next(self.inputs)
        def send(self,value):self.sent.append(value)
        def close(self):self.closed=True
    pipe=Pipe()
    cfg=Config(variant='v3',mock=True)
    _paddle_child(pipe,cfg)
    assert pipe.closed
    assert [status for status,_ in pipe.sent]==['ready','ok'],pipe.sent
    output=pipe.sent[-1][1]
    assert [p.get('reading_order_rank') for p in output['proposals']]==[1,None,0]
    assert [p['bbox'] for p in output['proposals']]==[[10,20,91,81],[20,1,40,9],[1,80,8,99]]
    for p,row in zip(output['proposals'],rows):
        assert Polygon(p['polygon']).equals(Polygon(row['polygon_points']))
    features=output.get('layout_features')
    assert features is not None,'worker discarded the raw public feature audit'
    assert features['raw_boxes']==rows
    assert features['schema_version']
    assert features['layout_shape_mode_default'] == 'auto'
    assert 'layout_shape_mode' not in features
    assert pipe.sent[0][1]['layout_shape_mode_default'] == 'auto'
    assert 'layout_shape_mode' not in pipe.sent[0][1]
    assert features['rank_count']==2 and features['polygon_count']==3
    assert [r['proposal_id'] for r in features['records']]==[p['id'] for p in output['proposals']]
    assert [r['raw_order'] for r in features['records']]==[2,None,1]
    json.dumps(output,allow_nan=False)


def test_cache_missing_features_explicit_without_rewriting_old_payload(tmp_path):
    from src.cascaded.runner import Experiment,write_json
    image=tmp_path/'image.bin';image.write_bytes(b'image-fingerprint-only')
    record={'key':'p','document_id':'d','page_index':0,'image':str(image),'image_id':1}
    exp=Experiment(Config(variant='v3',mock=True),[record],tmp_path/'out')
    path=exp.folder(record)/'paddle.json'
    write_json(path,{'signature':exp.signature,'image_sha256':exp.fingerprint(record),
                     'output':{'proposals':[legacy_payload()],'lines':[]},
                     'inference_seconds':0,'peak_memory':None})
    before=path.read_bytes()
    cached=exp.cached(record,'paddle')
    features=cached['output'].get('layout_features')
    assert features is not None,'old cache must explicitly report missing feature evidence'
    assert features['status']=='unavailable'
    assert features['rank_count']==0 and features['polygon_count']==0 and features['reasons']
    assert cached['output']['proposals']==[legacy_payload()]
    assert path.read_bytes()==before


def test_v3_usage_flags_have_distinct_cache_namespaces(tmp_path):
    from src.cascaded.runner import Experiment
    record={'key':'p','document_id':'d','page_index':0,'image':'unused','image_id':1}
    base=Experiment(Config(variant='v3',mock=True),[record],tmp_path/'out')
    cfg=Config(variant='v3',mock=True);cfg.enable_v3_polygons=True;cfg.enable_v3_reading_order=True
    enabled=Experiment(cfg,[record],tmp_path/'out')
    assert base.signature!=enabled.signature and base.root!=enabled.root
