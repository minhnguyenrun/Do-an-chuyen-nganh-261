"""V2.1 ownership/partition contracts. No models, weights or network."""
from copy import deepcopy
from dataclasses import asdict
import pytest
from src.cascaded.types import Config, Line, Proposal
from src.cascaded.attribution import finish_layout, line_link


def line(id='l', bbox=(10,10,110,30), text='Several ordinary words in this paragraph'):
    a,b,c,d = bbox
    return Line(id, [[a,b],[c,b],[c,d],[a,d]], text, .95, .95)


def region(id='p', bbox=(10,10,110,30), label='text', ids=('old',)):
    return Proposal(id, list(bbox), label, 'paddle', label, .8, list(ids))


def state(ls, baseline=(), fallback=(), id='t'):
    from src.cascaded.core import envelope, line_box
    target = {'id': id, 'bbox': envelope([line_box(l) for l in ls]),
              'line_ids': [l.id for l in ls], 'source_ids': [],
              'fallback_baseline': [asdict(p) for p in fallback], 'reason': 'orphan'}
    return {'recovery': list(baseline), 'review': [], 'decisions': [], 'targets': [target]}


def response(blocks, id='t', status='ok'):
    return {id: {'status':status, 'crop_bbox':[0,0,1000,1000], 'blocks':
                 [{'bbox':[v/1000 for v in bbox], 'label':label} for bbox,label in blocks]}}


def run(s, ls, blocks=(), cfg=None, status='ok'):
    return finish_layout(s, ls, response(blocks, status=status), cfg or Config(), 1000, 1000)


@pytest.mark.parametrize('legacy', [False, True])
def test_recovered_image_preserves_spanning_line_owner_in_final(legacy):
    from src.cascaded.fusion import prepare_layout, finish_layout as legacy_finish
    cfg = Config(line_coverage=.4)
    lines = [line('span', (0, 10, 100, 30))]
    text = region('text', (0, 0, 40, 40), ids=[])
    image = Proposal('image', [40, 0, 100, 40], 'image', 'paddle', 'image', .3)
    prepared = prepare_layout([text, image], lines, cfg, 1000, 1000)
    assert {p.id: p.line_ids for p in prepared['recovery']} == {'text': ['span'], 'image': []}
    finalizer = legacy_finish if legacy else finish_layout
    final, _, _, _ = finalizer(prepared, lines, {}, cfg, 1000, 1000)
    assert {p.id: p.line_ids for p in final} == {'text': ['span'], 'image': []}
    assert next(p for p in final if p.id == 'image').score == .3


def test_wide_context_box_with_only_target_ink_is_accepted_not_clipped():
    ls = [line()]
    final, audit, _, ds = run(state(ls), ls, [([0,0,250,100], 'text')])
    assert final[0].bbox == [0,0,250,100] and final[0].line_ids == ['l']
    assert final[0].score is None and ds[-1]['action'] == 'ADD'
    assert audit[0]['boxes'][0]['attribution']['non_target_ink_ratio'] == 0


def test_pure_context_box_does_not_poison_valid_candidate():
    ls = [line()]
    final, audit, _, _ = run(state(ls), ls, [([500,500,600,550], 'image'), ([10,10,110,30], 'text')])
    assert len(final) == 1 and 'PURE_CONTEXT_BOX' in audit[0]['boxes'][0]['codes']


def test_empty_checkpoint_does_not_compare_quality():
    ls = [line()]
    _, audit, _, ds = run(state(ls), ls, [([500,500,600,550], 'text')])
    assert audit[0]['checkpoints'] == ['STEP1_NO_BOX_TOUCHED_TARGET']
    assert 'replace_score' not in ds[-1] and 'keep_score' not in ds[-1]


def test_vertical_snap_only_y_and_line_bisection_rejected():
    ls = [line()]
    final, audit, _, _ = run(state(ls), ls, [([5,15,115,27], 'text')])
    assert final[0].bbox == [5,10,115,30]
    assert 'VERTICAL_SNAP_APPLIED' in audit[0]['boxes'][0]['codes']
    final, audit, _, ds = run(state(ls), ls, [([5,20,115,30], 'text')])
    assert not final and 'LINE_BISECTION' in audit[0]['boxes'][0]['codes']
    assert 'replace_score' not in ds[-1]


def test_rotated_line_requires_polygon_evidence_not_vertical_snap():
    l = Line('r', [[10,10],[110,40],[104,60],[4,30]], 'Several ordinary paragraph words', .9, .9)
    link = line_link(l, [0,25,120,50], Config())
    assert not link['horizontal'] and not link['snap']


def test_horizontal_split_is_review_without_transcript_slicing():
    ls = [line()]
    text = ls[0].text
    final, audit, reviews, _ = run(state(ls), ls, [([10,10,60,30], 'text'), ([60,10,110,30], 'text')])
    assert not final and reviews and ls[0].text == text
    assert all('HORIZONTAL_SPLIT_UNVERIFIED' in h['codes'] for h in audit[0]['boxes'])


def test_partition_heading_and_body_can_cover_three_lines():
    ls = [line('a', (10,10,110,30), 'A DOCUMENT HEADING'),
          line('b', (10,35,110,55), 'SECOND HEADING LINE'), line('c', (10,65,110,85))]
    final, audit, _, ds = run(state(ls), ls, [([10,10,110,55], 'paragraph_title'),
                                            ([10,65,110,85], 'text')])
    assert {p.label for p in final} == {'paragraph_title','text'}
    assert ds[-1]['count_coverage'] == 1 and audit[0]['partition_search']['beam_size'] <= 32
    assert len([lid for p in final for lid in p.line_ids]) == 3


def test_orphan_eighty_percent_count_keeps_remaining_line_in_review():
    ls = [line(str(i), (10,10+i*25,110,30+i*25)) for i in range(5)]
    final, _, reviews, ds = run(state(ls), ls, [([10,10,110,105], 'text')])
    assert ds[-1]['count_coverage'] == .8 and len(final[0].line_ids) == 4
    assert any(r['reason'] == 'unresolved_ocr_lines' and r['line_ids'] == ['4'] for r in reviews)
    final, audit, _, ds = run(state(ls), ls, [([10,10,110,80], 'text')])
    assert not final and 'STEP3_INCOMPLETE_TARGET_SET_COVERAGE' in audit[0]['checkpoints']
    assert 'replace_score' not in ds[-1]


def test_all_fallback_lines_required_even_when_eighty_percent_would_pass():
    ls = [line(str(i), (10,10+i*25,110,30+i*25)) for i in range(5)]
    p = region(bbox=(10,10,110,130), ids=[l.id for l in ls])
    final, _, _, ds = run(state(ls, [p], [p]), ls, [([10,10,110,105], 'text')])
    assert final == [p] and ds[-1]['action'] == 'KEEP' and 'replace_score' not in ds[-1]


def test_multiple_fallbacks_preserved_atomically():
    ls = [line('a', (10,10,110,30), 'Figure 1. Revenue chart'),
          line('b', (10,40,110,60), 'Figure 2. Revenue chart')]
    ps = [region('p1', (10,10,110,30), ids=['a']), region('p2', (10,40,110,60), ids=['b'])]
    final, _, _, ds = run(state(ls, ps, ps), ls, [([10,10,110,30], 'caption'), ([10,40,110,60], 'caption')])
    assert ds[-1]['action'] == 'REPLACE' and {lid for p in final for lid in p.line_ids} == {'a','b'}
    final, _, _, _ = run(state(ls, ps, ps), ls, [([10,10,110,30], 'caption')])
    assert final == ps


def test_foreign_multi_neighbor_and_other_orphans_are_contamination():
    old1, old2, target = line('a'), line('b', (130,10,230,30)), line('t', (10,40,110,60))
    ps = [region('p1', ids=['a']), region('p2', (130,10,230,30), ids=['b'])]
    final, audit, _, _ = run(state([target], ps), [old1,old2,target], [([0,0,240,70], 'text')])
    assert final == ps and 'FOREIGN_CONTAMINATION' in audit[0]['boxes'][0]['codes']
    # 1/8 of OCR area is foreign: the 10%-15% gap must never pass silently.
    target = line('t', (10,10,710,30))
    orphan = line('x', (10,35,110,55))
    final, audit, _, _ = run(state([target]), [target,orphan], [([0,0,720,60], 'text')])
    assert not final and audit[0]['boxes'][0]['attribution']['non_target_ink_ratio'] == pytest.approx(.125)


def test_nontext_overlap_uses_smaller_area_even_for_huge_model_box():
    ls = [line()]
    image = region('img', (200,200,220,220), 'image', [])
    final, audit, _, _ = run(state(ls, [image]), ls, [([0,0,900,900], 'text')])
    assert final == [image]
    assert audit[0]['boxes'][0]['attribution']['nontext_overlap'] == 1
    assert 'PROTECTED_NONTEXT_OVERLAP' in audit[0]['boxes'][0]['codes']


def test_absorb_uses_envelope_keeps_id_label_old_lines_and_input_immutable():
    old, target = line('old'), line('t', (10,40,110,60))
    p = region()
    s = state([target], [p])
    before = deepcopy(s)
    final, audit, _, ds = run(s, [old,target], [([0,0,130,70], 'text')])
    assert s == before
    assert final[0].id == 'p' and final[0].label == 'text' and final[0].source == 'paddle'
    assert final[0].bbox == [10,10,110,60] and final[0].line_ids == ['old','t']
    assert final[0].score is None and ds[-1]['action'] == 'ABSORB'
    assert audit[0]['checkpoints'] == ['ACCEPTED_NEIGHBOR_MERGE']


def test_absorb_low_gain_rolls_back_entire_neighbor():
    old = [line(str(i), (10,10+i*25,110,30+i*25)) for i in range(5)]
    target = line('t', (10,135,110,155))
    p = region(bbox=(10,10,110,130), ids=[l.id for l in old])
    final, audit, _, ds = run(state([target], [p]), old+[target], [([0,0,130,165], 'text')])
    assert final == [p] and ds[-1]['action'] == 'REVIEW'
    assert audit[0]['checkpoints'] == ['STEP4_LOST_TO_KEEP_BASELINE']


def test_absorb_wrong_column_or_heading_font_is_rejected():
    old = line('old')
    p = region()
    for target in [line('t', (130,40,230,60)), line('t', (10,40,110,70))]:
        final, audit, _, _ = run(state([target], [p]), [old,target], [([0,0,250,80], 'text')])
        assert final == [p] and any(c.startswith('ABSORB_') or c == 'semantic_gate_failed'
                                   for c in audit[0]['boxes'][0]['codes'])


def test_crop_translation_preserves_page_coordinates_and_edge_snap():
    ls = [line('l', (900,960,1000,980))]
    pred = {'t': {'status':'ok', 'crop_bbox':[850,900,1000,1000],
                  'blocks':[{'bbox':[1/3,.6,1.,.8], 'label':'text'}]}}
    final, _, _, _ = finish_layout(state(ls), ls, pred, Config(), 1000, 1000)
    assert final[0].bbox == pytest.approx([900,960,1000,980])


@pytest.mark.parametrize('pred', [
    {'status':'ok', 'blocks':None},
    {'status':'ok', 'blocks':[], 'crop_bbox':[0,0,1001,1000]},
    {'status':'ok', 'blocks':[], 'crop_bbox':[0,0,float('nan'),1000]},
])
def test_malformed_response_stops_without_quality(pred):
    ls = [line()]
    final, audit, _, ds = finish_layout(state(ls), ls, {'t':pred}, Config(), 1000, 1000)
    assert not final and audit[0]['status'] == 'invalid_output'
    assert audit[0]['checkpoints'] == ['MODEL_INVALID_OUTPUT'] and 'replace_score' not in ds[-1]


def test_neighbor_can_only_be_edited_once_per_page():
    old, a, b = line('old'), line('a', (10,40,110,60)), line('b', (10,70,110,90))
    s = state([a], [region()])
    s['targets'] += state([b], id='u')['targets']
    pred = response([([0,0,120,65], 'text')]) | response([([0,0,120,95], 'text')], id='u')
    final, audit, _, ds = finish_layout(s, [old,a,b], pred, Config(), 1000, 1000)
    assert ds[0]['action'] == 'ABSORB' and final[0].line_ids == ['old','a']
    assert 'NEIGHBOR_ALREADY_EDITED' in audit[1]['boxes'][0]['codes']


@pytest.mark.parametrize('status', ['incomplete','invalid_output','error','budget_exhausted','disabled'])
def test_model_failure_stops_before_geometry_or_quality(status):
    ls = [line()]
    final, audit, _, ds = run(state(ls), ls, [([10,10,110,30], 'text')], status=status)
    assert not final and audit[0]['boxes'] == []
    assert audit[0]['checkpoints'] == ['MODEL_'+status.upper()] and 'replace_score' not in ds[-1]


def test_candidate_limit_and_beam_are_deterministic():
    ls = [line()]
    blocks = [([10,10,110,30], 'text')]*26
    a = run(state(ls), ls, blocks)
    b = run(state(ls), ls, blocks)
    assert a == b and a[1][0]['partition_search']['candidate_count'] == 24
    assert sum('PARTITION_CANDIDATE_LIMIT' in h['codes'] for h in a[1][0]['boxes']) == 2


def test_snap_rechecks_foreign_ink_and_duplicate_line_partition():
    target = line(text='Several ordinary words in this paragraph.')
    foreign = line('x', (10,10,110,14))
    final, audit, _, _ = run(state([target]), [target,foreign], [([10,15,110,27], 'text')])
    assert not final
    h = audit[0]['boxes'][0]
    assert 'VERTICAL_SNAP_APPLIED' in h['codes'] and 'FOREIGN_CONTAMINATION' in h['codes']
    # Two valid boxes owning the same line cannot both enter a partition.
    final, _, _, _ = run(state([target]), [target], [([10,10,110,30], 'text'), ([0,0,120,35], 'text')])
    assert len(final) == 1 and final[0].line_ids == ['l']


def test_box_table_exposes_checkpoint_without_quality_on_empty():
    from src.cascaded.report import attribution_rows, review_summary
    ls = [line()]
    final, audit, reviews, ds = run(state(ls), ls, [([500,500,600,550], 'text')])
    result = {'document_id':'page', 'evidence':audit, 'decisions':ds, 'lines':[asdict(l) for l in ls],
              'review_evidence':reviews, 'stages':{'final':[asdict(p) for p in final]}}
    assert attribution_rows([result])[0]['codes'] == 'PURE_CONTEXT_BOX'
    assert review_summary([result])['mineru_checkpoints'] == {'STEP1_NO_BOX_TOUCHED_TARGET':1}


def test_post_snap_ownership_includes_newly_touched_target_line():
    big = line('big', text='Several ordinary words in this paragraph.')
    small = line('small', (10,10,110,14), 'A small supporting line.')
    final, audit, _, _ = run(state([big,small]), [big,small], [([10,15,110,27], 'text')])
    assert final[0].line_ids == ['big','small']
    assert {v['line_id'] for v in audit[0]['boxes'][0]['post_snap_links']} == {'big','small'}
