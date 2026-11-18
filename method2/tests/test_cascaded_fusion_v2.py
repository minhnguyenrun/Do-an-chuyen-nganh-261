"""V2 fusion acceptance tests; no weights, no internet."""
from copy import deepcopy
from dataclasses import asdict
import pytest
from src.cascaded.types import Config, Proposal, Line
from src.cascaded.fusion import prepare_layout, finish_layout, quality, compatible
from src.cascaded.semantic_gate import semantic_score


def line(id='l', bbox=(10,10,100,30), text='Several ordinary words in this paragraph', rec=.9):
    a,b,c,d = bbox
    return Line(id,[[a,b],[c,b],[c,d],[a,d]],text,.95,rec)


def p(id='p',bbox=(10,10,100,30),label='text',score=.8,source='paddle'):
    return Proposal(id,list(bbox),label,source,label,score)


def prep(ps,ls,cfg=None):
    return prepare_layout(ps,ls,cfg or Config(),1000,1000)


def finish(state,ls,pred=None,cfg=None):
    return finish_layout(state,ls,pred or {},cfg or Config(),1000,1000)


def response(t,blocks):
    # A full-page context makes the normalized fixture coordinates unambiguous.
    return {t['id']:{'status':'ok','crop_bbox':[0,0,1000,1000],'blocks':blocks}}


def test_low_yolo_cannot_veto_high_pp():
    ls=[line()]
    state=prep([p(),p('y',label='table',score=.11,source='yolo')],ls)
    final,_,_,ds=finish(state,ls)
    assert len(final)==1 and final[0].id=='p' and not state['targets']
    assert any(d['reason']=='pp_dominance' for d in ds)


def test_generic_title_supports_pp_subtype_but_subtypes_not_aliases():
    ls=[line(text='A DOCUMENT HEADING')]
    state=prep([p(label='paragraph_title'),p('y',label='title',source='yolo')],ls)
    assert not state['targets'] and state['ensemble'][0].label=='paragraph_title'
    assert compatible('title','doc_title') and not compatible('doc_title','paragraph_title')
    assert not compatible('table','image') and not compatible('header','footer')


def test_yolo_small_box_limits_and_nested_parent_preserved():
    ls=[line()]
    state=prep([p(bbox=(0,0,700,800)),p('nested',source='yolo'),
                p('large',(0,850,600,950),source='yolo')],ls)
    assert [r.id for r in state['ensemble']]==['p']
    assert {r['reason'] for r in state['review']}=={'nested_in_pp_block','outside_small_box_role'}


def test_small_boundary_and_single_small_yolo_supplement():
    from src.cascaded.fusion import small
    cfg=Config()
    assert small(p(bbox=(0,0,250,80)),cfg,1000,1000)
    assert not small(p(bbox=(0,0,251,80)),cfg,1000,1000)
    assert not small(p(bbox=(0,0,10,81)),cfg,1000,1000)
    state=prep([p('y',source='yolo')],[line()])
    assert len(state['ensemble'])==1 and state['ensemble'][0].stage=='yolo_small'


@pytest.mark.parametrize('status',['error','incomplete','invalid_output','budget_exhausted','disabled'])
def test_conflict_model_failure_keeps_pp(status):
    ls=[line(text='Figure 1. Company revenue')]
    state=prep([p(score=.6),p('y',label='caption',score=.65,source='yolo')],ls)
    assert len(state['targets'])==1
    t=state['targets'][0]
    assert t['fallback_baseline'][0]['id']=='p'
    final,_,reviews,ds=finish(state,ls,{t['id']:{'status':status,'blocks':[]}})
    assert [r.id for r in final]==['p'] and final[0].bbox==[10,10,100,30]
    assert ds[-1]['action']=='KEEP' and reviews


def test_verified_caption_replace_beats_keep_and_ocr_text_unchanged():
    ls=[line(text='Figure 1. Company revenue')]
    state=prep([p(score=.6),p('y',label='caption',score=.65,source='yolo')],ls)
    final,_,_,ds=finish(state,ls,response(state['targets'][0],[{'bbox':[.01,.01,.10,.03],'label':'caption'}]))
    assert len(final)==1 and final[0].label=='caption' and final[0].score is None
    assert ds[-1]['action']=='REPLACE' and ds[-1]['replace_score']>ds[-1]['keep_score']+.1
    assert ls[0].text=='Figure 1. Company revenue'


def test_same_quality_cannot_replace_and_missing_line_restores_baseline():
    ls=[line('a'),line('b',(10,35,100,55))]
    state=prep([p(bbox=(10,10,100,55),score=.6),p('y',(10,10,100,55),'caption',.65,'yolo')],ls)
    t=state['targets'][0]
    final,_,_,ds=finish(state,ls,response(t,[{'bbox':[.01,.01,.10,.03],'label':'text'}]))
    assert final[0].id=='p' and ds[-1]['action']=='KEEP'
    assert 'missing_target_lines' in ds[-1]['rejection_reasons']


def test_orphan_failed_or_picture_output_stays_review_not_detection():
    ls=[line()]
    state=prep([],ls)
    final,_,reviews,ds=finish(state,ls)
    assert not final and reviews and ds[-1]['action']=='REVIEW'
    final,_,reviews,ds=finish(state,ls,response(state['targets'][0],[{'bbox':[.01,.01,.10,.03],'label':'image'}]))
    assert not final and 'semantic_gate_failed' in ds[-1]['rejection_reasons']


def test_verified_orphan_can_be_added_without_default_text():
    ls=[line()]
    state=prep([],ls)
    final,_,_,ds=finish(state,ls,response(state['targets'][0],[{'bbox':[.01,.01,.10,.03],'label':'text'}]))
    assert len(final)==1 and final[0].source=='mineru' and ds[-1]['action']=='ADD'


def test_conflict_component_preserves_all_fallbacks():
    ls=[line('a',(10,10,100,30)),line('b',(10,35,100,55))]
    state=prep([p('a',(10,10,100,30),score=.6),p('b',(10,35,100,55),score=.6),
                p('y',(10,10,100,55),'caption',.6,'yolo')],ls)
    # IoU individual boxes = 20/45, below threshold; use explicit lower IoU ablation.
    cfg=Config(nms_iou=.4)
    state=prep([p('a',(10,10,100,30),score=.6),p('b',(10,35,100,55),score=.6),
                p('y',(10,10,100,55),'caption',.6,'yolo')],ls,cfg)
    assert {r['id'] for r in state['targets'][0]['fallback_baseline']}=={'a','b'}
    final,_,_,_=finish(state,ls,cfg=cfg)
    assert {r.id for r in final}=={'a','b'}


def test_recovery_needs_recognition_score_and_semantic_evidence():
    state=prep([p(score=.3)],[line(rec=None)])
    assert not state['recovery']
    state=prep([p(score=.3)],[line()])
    assert len(state['recovery'])==1
    state=prep([p(label='table',score=.3)],[line()])
    assert not state['recovery']


@pytest.mark.parametrize('score', [.10, .30, .449])
def test_weak_paddle_image_recovers_without_ocr(score):
    state = prep([p('image', (10, 10, 100, 100), 'image', score)], [])
    image, = state['recovery']
    assert (image.id, image.stage, image.score, image.line_ids) == ('image', 'recovery', score, [])
    final, _, _, _ = finish(state, [])
    assert [(region.id, region.score) for region in final] == [('image', score)]


def test_weak_image_requires_no_positive_area_overlap_with_retained_regions():
    state = prep([
        p('strong', (0, 0, 100, 100), 'image', .9),
        p('overlap', (99, 0, 199, 100), 'image', .3),
        p('touch', (100, 100, 200, 200), 'image', .3),
        p('weak_overlap', (199, 150, 250, 200), 'image', .3),
        p('below_low', (300, 0, 400, 100), 'image', .099),
    ], [])
    assert {region.id for region in state['recovery']} == {'strong', 'touch'}


def test_wbf_default_off_and_rejects_loss_of_line():
    ls=[line()]
    ps=[p(score=.5),p('y',(30,10,100,30),'text',.9,'yolo')]
    state=prep(ps,ls)
    assert state['ensemble'][0].bbox==[10,10,100,30]
    state=prep(ps,ls,Config(enable_wbf=True))
    assert state['ensemble'][0].bbox[0]>10
    # A narrower candidate pulls a boundary through >20% of the source line.
    ps=[p(score=.45),p('y',(50,10,100,30),'text',.99,'yolo')]
    state=prep(ps,ls,Config(enable_wbf=True))
    assert state['ensemble'][0].bbox==[10,10,100,30]
    assert state['decisions'][0]['reason']=='wbf_rejected_geometry_keep_pp'


def test_quality_coverage_is_union_not_sum_and_split_penalty_is_logged():
    ls=[line()]
    cfg=Config()
    score,features=quality([p(),p('duplicate')],ls,ls,1000,1000,cfg)
    assert features['coverage']==1
    _,features=quality([p(bbox=(10,10,50,30))],ls,ls,1000,1000,cfg)
    assert features['split_penalty']==1


def test_no_ground_truth_needed_and_inputs_are_not_mutated():
    ps,ls=[p(),p('y',label='image',source='yolo')],[line()]
    original=deepcopy(ps)
    state=prep(ps,ls)
    finish(state,ls)
    assert ps==original


def test_candidate_cannot_overwrite_locked_neighbour():
    ls=[line('locked',(125,10,149,30)),line('orphan',(150,10,250,30))]
    state=prep([p(bbox=(125,10,149,30))],ls)
    target=state['targets'][0]
    final,_,_,ds=finish(state,ls,response(target,[{'bbox':[.125,.01,.25,.03],'label':'text'}]))
    assert [r.id for r in final]==['p']
    assert 'overlaps_locked_region' in ds[-1]['rejection_reasons']


def test_disconnected_columns_cannot_replace_a_baseline_block():
    ls=[line('left',(10,10,100,30)),line('right',(200,10,290,30))]
    baseline=p(bbox=(10,10,290,30))
    baseline.line_ids=['left','right']
    state={'recovery':[baseline],'review':[],'decisions':[],
           'targets':[{'id':'t','bbox':baseline.bbox,'line_ids':['left','right'],
                       'source_ids':['p'],'fallback_baseline':[asdict(baseline)],'reason':'true_conflict'}]}
    final,_,_,ds=finish(state,ls,response(state['targets'][0],[{'bbox':[.01,.01,.29,.03],'label':'text'}]))
    assert [r.id for r in final]==['p']
    assert 'cross_column_or_disconnected_lines' in ds[-1]['rejection_reasons']


def test_review_does_not_count_as_detection_but_missing_gt_is_still_fn(tmp_path):
    from src.cascaded.report import evaluate
    from src.cascaded.runner import write_json
    baseline=p()
    r={'mode':'REAL','schema_version':'cascaded-layout-v2','config':asdict(Config()),
       'record':{'image_id':1},'image_sha256':'fixture','width':1000,'height':1000,
       'lines':[asdict(line())],'evidence':[],'timing':{'paddle':{'inference_seconds':1}},
       'stages':{'paddle':[asdict(baseline)],'ensemble':[],'recovery':[],'final':[]},
       'review_evidence':[{'id':'review','bbox':baseline.bbox,'label':'unknown','line_ids':['l'],
                           'reason':'orphan_abstained'}],'decisions':[]}
    gt={'images':[{'id':1,'file_name':'1.png','width':1000,'height':1000}],
        'annotations':[{'id':1,'image_id':1,'category_id':10,'bbox':[10,10,90,20]}],
        'categories':[{'id':10,'name':'Text'}]}
    write_json(tmp_path/'gt.json',gt)
    report=evaluate([r],tmp_path/'gt.json',tmp_path/'report')
    final=next(row for row in report['rows'] if row['stage']=='final' and row['iou']==.5)
    assert final['tp']==0 and final['fp']==0 and final['fn']==1
    assert report['review_summary']['unresolved_line_count']==1
    assert report['details']['final']['0.5']['source_errors']['fn_by_gt_class']=={'Text':1}
