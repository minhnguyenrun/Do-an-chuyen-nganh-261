"""Exercise V3 shape and reading-order evidence through the active cascade."""

from collections import Counter
from copy import deepcopy

import pytest

from src.cascaded.attribution import finish_layout
from src.cascaded.core import coverage
from src.cascaded.fusion import prepare_layout
from src.cascaded.types import Config, Line, Proposal


PAGE_SIZE = 1000
IMAGE_BBOX = (100, 100, 300, 300)
IMAGE_AWAY_FROM_TEXT = [[300, 100], [300, 300], [100, 300]]
IMAGE_TOUCHING_TEXT = [[100, 100], [300, 100], [300, 300]]


def rectangle(bbox):
    x1, y1, x2, y2 = bbox
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def line(id, bbox, text='These are ordinary words in a paragraph.'):
    return Line(id, rectangle(bbox), text, .95, .95)


def proposal(id, bbox, *, label='text', source='paddle', score=.8, rank=None, polygon=None):
    return Proposal(id, list(bbox), label, source, label, score,
                    reading_order_rank=rank, polygon=polygon)


def prepare(proposals, lines, cfg):
    return prepare_layout(proposals, lines, cfg, PAGE_SIZE, PAGE_SIZE)


def prediction(target, bbox):
    return {target['id']: {'status': 'ok', 'crop_bbox': [0, 0, PAGE_SIZE, PAGE_SIZE],
                           'blocks': [{'bbox': [xy / PAGE_SIZE for xy in bbox], 'label': 'text'}]}}


def finish(state, lines, cfg, predictions):
    return finish_layout(state, lines, predictions, cfg, PAGE_SIZE, PAGE_SIZE)


def target_for(state, line_id):
    return next((target for target in state['targets'] if line_id in target['line_ids']), None)


def regions_by_id(regions):
    return {region.id: region for region in regions}


def assert_accounting(regions, reviews, lines):
    owned = Counter(line_id for region in regions for line_id in region.line_ids)
    assert all(count == 1 for count in owned.values())
    unresolved = {line_id for review in reviews if review['reason'] == 'unresolved_ocr_lines'
                  for line_id in review['line_ids']}
    assert set(owned).isdisjoint(unresolved)
    assert set(owned) | unresolved == {line.id for line in lines}


@pytest.mark.parametrize('geometry_refinement', [False, True])
def test_measured_polygon_releases_false_image_collision(geometry_refinement):
    text_bbox = (120, 130, 190, 150)
    lines = [line('orphan', text_bbox)]
    image = proposal('image', IMAGE_BBOX, label='image', polygon=IMAGE_AWAY_FROM_TEXT)
    cfg = Config(variant='v3', enable_v3_polygons=True, enable_geometry_refinement=geometry_refinement)
    state = prepare([image], lines, cfg)
    target = target_for(state, 'orphan')
    assert target is not None
    assert regions_by_id(state['recovery'])['image'].line_ids == []

    final, audit, reviews, decisions = finish(state, lines, cfg, prediction(target, text_bbox))

    assert decisions[-1]['action'] == 'ADD'
    assert audit[0]['boxes'][0]['attribution']['nontext_overlap'] == 0
    assert regions_by_id(final)['image'].bbox == list(IMAGE_BBOX)
    assert regions_by_id(final)['image'].polygon == IMAGE_AWAY_FROM_TEXT
    assert [region.bbox for region in final if region.label == 'text'] == [list(text_bbox)]
    assert_accounting(final, reviews, lines)


@pytest.mark.parametrize('geometry_refinement', [False, True])
def test_measured_polygon_still_blocks_real_image_overlap(geometry_refinement):
    text_bbox = (120, 130, 190, 150)
    lines = [line('orphan', text_bbox)]
    image = proposal('image', IMAGE_BBOX, label='image', polygon=IMAGE_TOUCHING_TEXT)
    cfg = Config(variant='v3', enable_v3_polygons=True, enable_geometry_refinement=geometry_refinement)
    state = prepare([image], lines, cfg)
    # The triangle covers 5/7 of the OCR line, below assignment's 80% threshold.
    target = target_for(state, 'orphan')
    assert target is not None

    final, audit, reviews, decisions = finish(state, lines, cfg, prediction(target, text_bbox))

    assert decisions[-1]['action'] == 'REVIEW'
    assert [region.id for region in final] == ['image']
    box = audit[0]['boxes'][0]
    if geometry_refinement:
        assert not box['geometry_refinement']['valid']
        assert any(code.startswith('PROTECTED_NONTEXT') for code in box['codes'])
    else:
        assert box['attribution']['nontext_overlap'] == pytest.approx(5 / 7)
        assert 'PROTECTED_NONTEXT_OVERLAP' in box['codes']
    assert_accounting(final, reviews, lines)


@pytest.mark.parametrize(('variant', 'enabled', 'polygon'), [
    ('s', True, IMAGE_AWAY_FROM_TEXT),
    ('v3', False, IMAGE_AWAY_FROM_TEXT),
    ('v3', True, None),
    ('v3', True, [[100, 100], [300, 300], [100, 300], [300, 100]]),
])
def test_ineligible_or_invalid_polygons_use_legacy_ownership_and_targets(variant, enabled, polygon):
    lines = [line('text_in_rectangle', (120, 130, 190, 150)),
             line('outside', (450, 450, 580, 470))]
    image = proposal('image', IMAGE_BBOX, label='image', polygon=polygon)
    cfg = Config(variant=variant, enable_v3_polygons=enabled)

    state = prepare([image], lines, cfg)
    legacy = prepare([proposal('image', IMAGE_BBOX, label='image')], lines, Config(variant=variant))

    assert [(p.id, p.bbox, p.line_ids) for p in state['recovery']] == [
        (p.id, p.bbox, p.line_ids) for p in legacy['recovery']]
    assert state['targets'] == legacy['targets']
    assert state['recovery'][0].line_ids == ['text_in_rectangle']


def test_polygon_mode_keeps_frozen_foreign_owners():
    lines = [line('a', (115, 112, 160, 130)), line('b', (165, 114, 210, 132)),
             line('orphan', (115, 155, 190, 175))]
    proposals = [proposal('image', IMAGE_BBOX, label='image', polygon=IMAGE_AWAY_FROM_TEXT),
                 proposal('first', (115, 112, 160, 130)), proposal('second', (165, 114, 210, 132))]
    cfg = Config(variant='v3', enable_v3_polygons=True, enable_recovery=False)
    state = prepare(proposals, lines, cfg)
    target = target_for(state, 'orphan')
    assert target is not None

    final, audit, reviews, decisions = finish(state, lines, cfg, prediction(target, (110, 105, 215, 180)))

    assert audit[0]['frozen_ownership']['confirmed_line_owners'] == {'a': 'first', 'b': 'second'}
    assert audit[0]['boxes'][0]['attribution']['nontext_overlap'] == 0
    assert 'FOREIGN_CONTAMINATION' in audit[0]['boxes'][0]['codes']
    assert decisions[-1]['action'] == 'REVIEW'
    assert regions_by_id(final)['first'].line_ids == ['a']
    assert regions_by_id(final)['second'].line_ids == ['b']
    assert_accounting(final, reviews, lines)


def test_wbf_geometry_change_clears_measured_mask_and_preserves_ocr():
    bbox = (10, 10, 100, 30)
    paddle = proposal('p', bbox, score=.5, rank=2, polygon=rectangle(bbox))
    yolo = proposal('y', (30, 10, 100, 30), source='yolo', score=.9)
    old = line('old', bbox)
    cfg = Config(variant='v3', enable_v3_polygons=True, enable_wbf=True)

    state = prepare([paddle, yolo], [old], cfg)

    fused = regions_by_id(state['recovery'])['p']
    assert fused.stage == 'wbf' and fused.bbox[0] > bbox[0]
    assert set(fused.source_ids) == {'p', 'y'}
    assert fused.polygon is None and paddle.polygon == rectangle(bbox)
    assert fused.line_ids == ['old'] and coverage(old, fused.bbox) >= cfg.line_coverage


def test_aligned_boundary_expansion_clears_mask_before_reassigning_lines():
    bbox = (100, 100, 300, 120)
    p = proposal('p', bbox, rank=1, polygon=rectangle(bbox))
    lines = [line('old', (110, 100, 290, 120)), line('extra', (100, 125, 300, 145))]
    cfg = Config(variant='v3', enable_v3_polygons=True)

    state = prepare([p], lines, cfg)

    assert regions_by_id(state['ensemble'])['p'].polygon == rectangle(bbox)
    expanded = regions_by_id(state['recovery'])['p']
    assert expanded.bbox == [100, 100, 300, 145] and expanded.polygon is None
    assert expanded.reading_order_rank == 1
    assert set(expanded.line_ids) == {'old', 'extra'} and not state['targets']
    assert p.bbox == list(bbox) and p.polygon == rectangle(bbox)


def paragraph_boundary():
    before = proposal('before', (100, 100, 320, 128), rank=4)
    recipient_bbox = (100, 174, 320, 194)
    recipient = proposal('recipient', recipient_bbox, rank=5, polygon=rectangle(recipient_bbox))
    lines = [line('earlier', (100, 108, 310, 128), 'That is the previous paragraph.'),
             line('first_line', (145, 149, 310, 169), 'A new paragraph starts here.'),
             line('continuation', (100, 174, 315, 194), 'The remaining words continue here.')]
    return [before, recipient], lines


def test_order_recovers_only_the_indented_first_line_in_same_column():
    proposals, lines = paragraph_boundary()
    legacy = prepare(proposals, lines, Config(variant='v3', enable_v3_polygons=True))
    assert target_for(legacy, 'first_line') is not None
    cfg = Config(variant='v3', enable_v3_polygons=True, enable_v3_reading_order=True)

    state = prepare(proposals, lines, cfg)

    assert target_for(state, 'first_line') is None
    detected = regions_by_id(state['recovery'])
    assert detected['before'].line_ids == ['earlier']
    assert set(detected['recipient'].line_ids) == {'first_line', 'continuation'}
    assert detected['recipient'].bbox == [100, 149, 320, 194]
    assert detected['recipient'].polygon is None and detected['recipient'].reading_order_rank == 5


def test_order_expansion_cannot_transfer_a_previously_owned_continuation():
    proposals, lines = paragraph_boundary()
    proposals.append(proposal('competing_unranked', (90, 174, 330, 194)))
    cfg = Config(variant='v3', enable_v3_reading_order=True)

    state = prepare(proposals, lines, cfg)

    ensemble = regions_by_id(state['ensemble'])
    assert ensemble['recipient'].line_ids == ['continuation']
    assert ensemble['competing_unranked'].line_ids == []
    final, _, reviews, _ = finish(state, lines, cfg, {})
    assert set(regions_by_id(final)['recipient'].line_ids) == {'continuation', 'first_line'}
    assert 'continuation' not in regions_by_id(final)['competing_unranked'].line_ids
    assert_accounting(final, reviews, lines)


def test_order_requires_same_column_overlap_after_wbf_changes_an_anchor():
    proposals, lines = paragraph_boundary()
    proposals[1].bbox = [150, 174, 370, 194]
    proposals[1].polygon = rectangle(proposals[1].bbox)
    proposals.append(proposal('yolo', (200, 174, 420, 194), source='yolo', score=.9))
    lines[1].polygon = rectangle((200, 149, 310, 169))
    lines[2].polygon = rectangle((190, 174, 365, 194))
    cfg = Config(variant='v3', enable_v3_reading_order=True, enable_wbf=True)

    state = prepare(proposals, lines, cfg)

    recipient = regions_by_id(state['ensemble'])['recipient']
    assert recipient.stage == 'wbf'
    assert recipient.bbox[0] == pytest.approx((.8 * 150 + .9 * 200) / 1.7)
    assert target_for(state, 'first_line') is not None
    assert regions_by_id(state['recovery'])['recipient'].line_ids == ['continuation']


@pytest.mark.parametrize('failure', ['cross_column', 'pagewide_heading', 'duplicate_rank', 'missing_rank',
                                     'inconsistent_direction', 's_variant'])
def test_order_abstains_without_unique_directional_same_column_evidence(failure):
    proposals, lines = paragraph_boundary()
    if failure == 'cross_column':
        proposals[0].bbox = [560, 100, 800, 128]
        lines[0].polygon = rectangle((560, 108, 780, 128))
    elif failure == 'pagewide_heading':
        proposals[0] = proposal('heading', (30, 100, 970, 128), label='doc_title', rank=4)
        proposals.append(proposal('other_column', (620, 63, 800, 85), rank=3))
        lines.append(line('other', (620, 64, 795, 84)))
    elif failure == 'duplicate_rank':
        proposals.append(proposal('duplicate', (620, 100, 800, 128), rank=4))
        lines.append(line('duplicate_line', (620, 108, 795, 128)))
    elif failure == 'missing_rank':
        proposals[0].reading_order_rank = None
    elif failure == 'inconsistent_direction':
        proposals[0].reading_order_rank, proposals[1].reading_order_rank = 5, 4
    cfg = Config(variant='s' if failure == 's_variant' else 'v3', enable_v3_reading_order=True)

    state = prepare(proposals, lines, cfg)

    assert target_for(state, 'first_line') is not None
    assert regions_by_id(state['recovery'])['recipient'].line_ids == ['continuation']


def test_graph_abstains_when_two_adjacent_regions_can_take_the_orphan():
    from src.cascaded.reading_order import build_reading_graph, graph_neighbor

    proposals, lines = paragraph_boundary()
    # Put both body regions less than one line height from the orphan.
    proposals[0].bbox = [125, 105, 345, 145]
    lines[0].polygon = rectangle((125, 125, 335, 145))
    cfg = Config(variant='v3', enable_v3_reading_order=True, enable_recovery=False)
    confirmed = prepare(proposals, lines, cfg)['recovery']
    graph = build_reading_graph(confirmed, lines, cfg)

    association = graph_neighbor(regions_by_id(confirmed)['recipient'], [lines[1]], graph, lines, confirmed)

    assert not association['supported']
    assert association['neighbor_id'] is None
    assert association['reasons']


def test_reading_graph_abstains_when_the_other_anchor_is_relabelled():
    from src.cascaded.reading_order import build_reading_graph, graph_neighbor

    proposals, lines = paragraph_boundary()
    cfg = Config(variant='v3', enable_v3_reading_order=True, enable_recovery=False)
    confirmed = prepare(proposals, lines, cfg)['recovery']
    graph = build_reading_graph(confirmed, lines, cfg)
    recipient = regions_by_id(confirmed)['recipient']
    assert graph_neighbor(recipient, [lines[1]], graph, lines, confirmed)['supported']

    regions_by_id(confirmed)['before'].label = 'caption'
    association = graph_neighbor(recipient, [lines[1]], graph, lines, confirmed)

    assert not association['supported']
    assert association['neighbor_id'] is None


def test_order_allows_absorb_once_and_preserves_frozen_ownership():
    proposals, lines = paragraph_boundary()
    lines.append(line('another_orphan', (100, 201, 310, 221)))
    legacy_cfg = Config(variant='v3', enable_v3_polygons=True, enable_recovery=False)
    cfg = Config(variant='v3', enable_v3_polygons=True, enable_v3_reading_order=True, enable_recovery=False)
    state = prepare(proposals, lines, cfg)
    original_state = deepcopy(state)
    first = target_for(state, 'first_line')
    second = target_for(state, 'another_orphan')
    assert first is not None and second is not None and first['id'] != second['id']
    first_response = prediction(first, (99, 147, 320, 197))
    second_response = prediction(second, (99, 173, 320, 223))
    legacy = prepare(proposals, lines, legacy_cfg)
    _, legacy_audit, _, legacy_decisions = finish(legacy, lines, legacy_cfg, first_response)
    assert legacy_decisions[-2]['action'] == 'REVIEW'
    assert 'ABSORB_COLUMN_MISMATCH' in legacy_audit[0]['boxes'][0]['codes']

    final, audit, reviews, decisions = finish(state, lines, cfg, first_response | second_response)

    assert state == original_state
    assert [decision['action'] for decision in decisions[-2:]] == ['ABSORB', 'REVIEW']
    assert 'NEIGHBOR_ALREADY_EDITED' in audit[1]['boxes'][0]['codes']
    detected = regions_by_id(final)
    assert detected['before'].line_ids == ['earlier']
    recipient = detected['recipient']
    assert recipient.bbox == [100, 149, 320, 194]
    assert set(recipient.line_ids) == {'continuation', 'first_line'}
    assert recipient.polygon is None and recipient.reading_order_rank == 5
    assert all(coverage(owned, recipient.bbox) == 1 for owned in lines if owned.id in recipient.line_ids)
    assert_accounting(final, reviews, lines)


def test_graph_absorb_cannot_edit_a_neighbor_expanded_during_preparation():
    proposals, lines = paragraph_boundary()
    lines.insert(0, line('aligned_bottom', (100, 201, 310, 221)))
    cfg = Config(variant='v3', enable_v3_reading_order=True)
    state = prepare(proposals, lines, cfg)
    target = target_for(state, 'first_line')
    assert target is not None
    prepared_recipient = regions_by_id(state['recovery'])['recipient']
    assert set(prepared_recipient.line_ids) == {'continuation', 'aligned_bottom'}
    assert prepared_recipient.bbox == [100, 174, 320, 221]

    final, _, reviews, decisions = finish(state, lines, cfg, prediction(target, (99, 147, 320, 197)))

    assert regions_by_id(final)['recipient'] == prepared_recipient
    assert decisions[-1]['action'] == 'REVIEW'
    assert_accounting(final, reviews, lines)
