"""Fast Cascaded Step2 safety regressions, with no model or dataset dependency.

The copied Sasol fixture is from cached valid V3 image_3/target:0 (source
cc544104...): the model box covers line172 and only grazes line171.
These tests catch indiscriminate bisection rejection and unsafe blanket refits.
"""
from copy import deepcopy
from dataclasses import asdict
import importlib
import importlib.util
import pytest
from src.cascaded.types import Config, Line, Proposal
from src.cascaded.attribution import finish_layout
from src.cascaded.core import coverage


def line(id='core', bbox=(10,10,110,30), text='Several ordinary words in this paragraph.'):
    a,b,c,d=bbox
    return Line(id,[[a,b],[c,b],[c,d],[a,d]],text,.95,.95)


def cfg(enabled=True):
    c=Config()
    c.enable_geometry_refinement=enabled
    return c


def refine(bbox, target, all_lines=None, owners=None, confirmed=None, label='text'):
    # Test-local API lookup keeps the baseline RED as an assertion, not a
    # collection/import error; all actual checks exercise the real geometry.
    name='src.cascaded.geometry_refine'
    assert importlib.util.find_spec(name) is not None, 'Step2 geometry refinement API is missing'
    return importlib.import_module(name).refine_candidate_bbox(
        bbox,label,target,all_lines or target,owners or {},confirmed or [],cfg())


def test_cached_sasol_grazing_line_is_excluded_while_core_is_retained():
    grazing=Line('line:171',[[768,66],[889,65],[889,80],[768,81]],'reaching new frontiers',.7943,.9890)
    core=Line('line:172',[[769,45],[886,45],[886,67],[769,67]],'SasOL',.8857,.6733)
    bbox=[757.634,38.61,890.228,66.15]
    before=deepcopy((bbox,[grazing,core]))
    out=refine(bbox,[grazing,core],label='title')
    assert out['valid'] and out['changed']
    assert out['core_line_ids']==['line:172']
    assert 'line:171' in out['excluded_line_ids']
    assert coverage(core,out['bbox'])>=.8
    assert coverage(grazing,out['bbox'])==0
    assert (bbox,[grazing,core])==before


def test_no_core_midline_bisection_is_not_rescued_by_envelope_expansion():
    out=refine([10,22,110,30],[line()])
    assert not out['valid']


def test_refit_removes_blank_margin_but_keeps_every_core_line():
    a,b=line('a'),line('b',(10,40,110,60))
    out=refine([0,0,180,70],[a,b])
    assert out['valid'] and set(out['core_line_ids'])=={'a','b'}
    assert out['bbox']==pytest.approx([10,10,110,60])
    assert coverage(a,out['bbox'])>=.8 and coverage(b,out['bbox'])>=.8


def test_carve_protected_picture_in_blank_space_without_cutting_target_ink():
    target=line(bbox=(10,30,80,50))
    picture=Proposal('picture',[100,0,200,80],'image','paddle','image',.9)
    out=refine([0,0,120,80],[target],confirmed=[picture],label='paragraph_title')
    assert out['valid'] and out['changed']
    assert out['bbox'][2]<=100 and coverage(target,out['bbox'])>=.8


def test_protected_picture_with_real_target_ink_overlap_is_not_carved_through():
    target=line(bbox=(60,30,120,50))
    picture=Proposal('picture',[100,0,200,80],'image','paddle','image',.9)
    out=refine([0,0,140,80],[target],confirmed=[picture])
    assert not out['valid']


def test_nontext_candidate_does_not_collapse_to_ocr_text_envelope():
    out=refine([0,0,150,80],[line()],label='table')
    assert out['bbox']==[0,0,150,80]


def _orphan_state(target,baseline=()):
    return {'recovery':list(baseline),'review':[],'decisions':[],
            'targets':[{'id':'t','bbox':[10,10,110,60],'line_ids':[l.id for l in target],
                        'source_ids':[],'fallback_baseline':[],'reason':'orphan'}]}


def test_grazing_orphan_add_runs_existing_semantic_and_line_guards_after_refinement():
    cores=[line(str(i),(10,10+i*25,110,30+i*25)) for i in range(4)]
    grazing=line('grazing',(10,110,110,130))
    state=_orphan_state(cores+[grazing]);before=deepcopy(state)
    pred={'t':{'status':'ok','crop_bbox':[0,0,1000,1000],
               'blocks':[{'bbox':[.01,.01,.11,.115],'label':'text'}]}}
    final,audit,reviews,decisions=finish_layout(state,cores+[grazing],pred,cfg(),1000,1000)
    assert len(final)==1 and final[0].line_ids==['0','1','2','3']
    assert decisions[-1]['action']=='ADD'
    assert any('grazing' in r.get('line_ids',[]) for r in reviews)
    assert state==before and cores[0].text=='Several ordinary words in this paragraph.'


def test_foreign_owned_line_is_retained_once_after_geometry_refinement():
    core=line('core');foreign=line('foreign',(10,40,110,60))
    baseline=Proposal('p',[10,40,110,60],'text','paddle','text',.9,['foreign'])
    state=_orphan_state([core],[baseline]);before=deepcopy(state)
    pred={'t':{'status':'ok','crop_bbox':[0,0,1000,1000],
               'blocks':[{'bbox':[.01,.01,.11,.045],'label':'text'}]}}
    final,_,_,ds=finish_layout(state,[core,foreign],pred,cfg(),1000,1000)
    assert 'foreign' in {p.id:p.line_ids for p in final}.get('p',[])
    assert sum('core' in p.line_ids for p in final)==1
    assert len([lid for p in final for lid in p.line_ids])==2
    assert state==before


def test_refined_candidate_still_fails_unverified_table_semantics():
    core=line('core')
    state=_orphan_state([core])
    pred={'t':{'status':'ok','crop_bbox':[0,0,1000,1000],
               'blocks':[{'bbox':[0,0,.15,.08],'label':'table'}]}}
    final,audit,_,ds=finish_layout(state,[core],pred,cfg(),1000,1000)
    assert not final and ds[-1]['action']=='REVIEW'
    assert 'semantic_gate_failed' in audit[0]['boxes'][0]['codes']


def test_disabled_refinement_preserves_existing_grazing_rejection():
    core=line('core');grazing=line('grazing',(10,40,110,60))
    pred={'t':{'status':'ok','crop_bbox':[0,0,1000,1000],
               'blocks':[{'bbox':[.01,.01,.11,.045],'label':'text'}]}}
    final,audit,_,_=finish_layout(_orphan_state([core,grazing]),[core,grazing],pred,cfg(False),1000,1000)
    assert not final and 'LINE_BISECTION' in audit[0]['boxes'][0]['codes']


@pytest.mark.filterwarnings('ignore:invalid value encountered in linearrings:RuntimeWarning')
@pytest.mark.parametrize('polygon',[
    [[10,10],[110,30],[10,30],[110,10]],
    [[10,10],[10,10],[10,10]],
    [[10,10],[110,float('nan')],[110,30],[10,30]],
])
def test_invalid_polygon_cannot_bypass_refinement_and_atomic_fallback(polygon):
    invalid=Line('bad',polygon,'Several ordinary words in this paragraph.',.95,.95)
    baseline=Proposal('p',[10,10,110,30],'text','paddle','text',.9,['bad'])
    state=_orphan_state([invalid],[baseline])
    state['targets'][0]['fallback_baseline']=[asdict(baseline)]
    state['targets'][0]['reason']='true_conflict'
    pred={'t':{'status':'ok','crop_bbox':[0,0,1000,1000],
               'blocks':[{'bbox':[0,0,.15,.08],'label':'text'}]}}
    final,audit,_,ds=finish_layout(state,[invalid],pred,cfg(),1000,1000)
    assert final==[baseline] and ds[-1]['action']=='KEEP'
    assert not any(h.get('selected_partition') for h in audit[0]['boxes'])


def test_rotated_partial_line_does_not_get_silently_excluded():
    core=line('core')
    rotated=Line('rotated',[[10,40],[110,60],[107,74],[7,54]],'Rotated words',.9,.9)
    assert not refine([0,0,120,45],[core,rotated])['valid']


def test_refit_cannot_hide_substantial_foreign_ink_in_blank_outer_extent():
    core=line('core');foreign=line('foreign',(200,10,300,30))
    assert not refine([0,0,400,80],[core],[core,foreign])['valid']


def test_core_lines_in_separate_columns_are_not_refit_into_one_region():
    left=line('left');right=line('right',(300,10,400,30))
    assert not refine([0,0,420,50],[left,right])['valid']


def test_partial_interior_line_between_cores_is_not_deleted_to_make_box_valid():
    top=line('top');bottom=line('bottom',(10,60,110,80))
    middle=line('middle',(0,35,500,45))
    assert not refine([10,10,110,80],[top,middle,bottom])['valid']


def test_disabled_helper_keeps_geometry_and_all_inputs_immutable():
    from src.cascaded.geometry_refine import refine_candidate_bbox
    core=line();bbox=[0,0,150,80];before=deepcopy((bbox,[core]))
    out=refine_candidate_bbox(bbox,'text',[core],[core],{},[],cfg(False))
    assert out['bbox']==bbox and not out['changed']
    assert (bbox,[core])==before


def test_geometry_enabled_complete_fallback_is_kept_when_grazing_line_is_excluded():
    cores=[line(str(i),(10,10+i*25,110,30+i*25)) for i in range(4)]
    grazing=line('grazing',(10,110,110,130));all_lines=cores+[grazing]
    baseline=Proposal('p',[10,10,110,130],'text','paddle','text',.9,[l.id for l in all_lines])
    state=_orphan_state(all_lines,[baseline])
    state['targets'][0]['fallback_baseline']=[asdict(baseline)]
    state['targets'][0]['reason']='true_conflict';before=deepcopy(state)
    pred={'t':{'status':'ok','crop_bbox':[0,0,1000,1000],
               'blocks':[{'bbox':[.01,.01,.11,.115],'label':'text'}]}}
    final,audit,_,ds=finish_layout(state,all_lines,pred,cfg(),1000,1000)
    assert final==[baseline] and ds[-1]['action']=='KEEP'
    assert audit[0]['boxes'][0]['geometry_refinement']['changed']
    assert 'STEP3_INCOMPLETE_TARGET_SET_COVERAGE' in audit[0]['checkpoints']
    assert 'replace_score' not in ds[-1] and state==before


def test_geometry_enabled_valid_refined_partition_cannot_override_saturated_keep_quality():
    core=line('core')
    baseline=Proposal('p',[10,10,110,30],'text','paddle','text',.9,['core'])
    state=_orphan_state([core],[baseline])
    state['targets'][0]['fallback_baseline']=[asdict(baseline)]
    state['targets'][0]['reason']='true_conflict'
    pred={'t':{'status':'ok','crop_bbox':[0,0,1000,1000],
               'blocks':[{'bbox':[0,0,.18,.08],'label':'text'}]}}
    final,audit,_,ds=finish_layout(state,[core],pred,cfg(),1000,1000)
    assert final==[baseline] and ds[-1]['action']=='KEEP'
    assert audit[0]['boxes'][0]['geometry_refinement']['changed']
    assert audit[0]['quality_diagnostics']['partition_feasible']
    assert audit[0]['quality_diagnostics']['blocked_by_legacy_q']
    assert ds[-1]['keep_score']>=.60 and ds[-1]['replace_score']<=.70
