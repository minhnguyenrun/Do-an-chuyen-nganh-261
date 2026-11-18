"""Independent semantic routing and hard-gate tests using a fixed oracle."""

import math

import pytest

from src.cascaded.attribution import finish_layout
from src.cascaded.fusion import prepare_layout, quality
from src.cascaded.types import Config, Line, Proposal


PAGE = 1000


def rectangle(bbox):
    a, b, c, d = bbox
    return [[a, b], [c, b], [c, d], [a, d]]


def line(id='l', bbox=(20, 10, 180, 30), text='Figure 1. Quarterly revenue.', rec=.95):
    return Line(id, rectangle(bbox), text, .95, rec)


def proposal(id, bbox, label='text', score=.6, source='paddle', **features):
    return Proposal(id, list(bbox), label, source, label, score, **features)


def conflict(pp_score=.6, yolo_score=.65):
    bbox = (10, 10, 210, 30)
    return [proposal('p', bbox, score=pp_score, polygon=rectangle(bbox), reading_order_rank=5),
            proposal('y', (12, 10, 196, 30), 'caption', yolo_score, 'yolo')]


def prepared(proposals, lines, cfg, *, semantic=None):
    args = {} if semantic is None else {'semantic': semantic}
    return prepare_layout(proposals, lines, cfg, PAGE, PAGE, **args)


def response(target, bbox, label):
    return {target['id']: {'status': 'ok', 'crop_bbox': [0, 0, PAGE, PAGE],
                           'blocks': [{'bbox': [xy / PAGE for xy in bbox], 'label': label}]}}


class SemanticOracle:
    def __init__(self, probabilities=None, *, reason=None, available=None):
        self.probabilities = probabilities or {}
        self.reason = reason
        self.available = (reason is None) if available is None else available
        self.calls = []

    def predict(self, label, lines, page_lines, bbox, width, height):
        self.calls.append((label, [line.id for line in lines], list(bbox)))
        assert (width, height) == (PAGE, PAGE)
        return {'available': self.available, 'probabilities': dict(self.probabilities),
                'supported_classes': sorted(self.probabilities), 'abstain_reason': self.reason,
                'calibration_status': 'calibrated' if self.available else 'unavailable',
                'model_fingerprint': 'fixture-model', 'feature_fingerprint': 'fixture-features',
                'cpu_seconds': 0.0}


def test_arbitration_resolves_one_to_one_conflict_and_preserves_measured_paddle_geometry():
    proposals, lines = conflict(), [line()]
    cfg = Config(variant='v3', enable_semantic_arbitration=True)
    oracle = SemanticOracle({'text': .05, 'caption': .95})

    state = prepared(proposals, lines, cfg, semantic=oracle)

    assert oracle.calls and not state['targets']
    region, = state['recovery']
    assert (region.id, region.source, region.label, region.score) == ('p', 'paddle', 'caption', .6)
    assert region.bbox == proposals[0].bbox and region.line_ids == ['l']
    assert 'p' in region.source_ids
    assert region.polygon == proposals[0].polygon and region.reading_order_rank == 5
    assert proposals[0].label == 'text'


@pytest.mark.parametrize(('pp_score', 'yolo_score', 'probability', 'blocking_gate'), [
    (.60, .65, .60, 'joint'),
    (.75, .78, .56, 'margin'),
])
def test_arbitration_requires_both_joint_score_and_margin(pp_score, yolo_score, probability, blocking_gate):
    pp_joint = .55 * pp_score + .45 * (1 - probability)
    yolo_joint = .55 * yolo_score + .45 * probability
    if blocking_gate == 'joint':
        assert yolo_joint < .65
    else:
        assert yolo_joint >= .65 and yolo_joint - pp_joint < .15
    lines = [line()]
    proposals = conflict(pp_score, yolo_score)
    oracle = SemanticOracle({'text': 1 - probability, 'caption': probability})
    cfg = Config(variant='v3', enable_semantic_arbitration=True)

    state = prepared(proposals, lines, cfg, semantic=oracle)

    assert oracle.calls
    assert state['targets'][0]['reason'] == 'true_conflict'
    assert state['recovery'][0].label == 'text'


def test_arbitration_preserves_multiple_fallback_owners():
    proposals = [proposal('a', (10, 10, 100, 30)), proposal('b', (10, 35, 100, 55)),
                 proposal('y', (10, 10, 100, 55), 'caption', .65, 'yolo')]
    lines = [line('a_line', (10, 10, 100, 30)), line('b_line', (10, 35, 100, 55))]
    cfg = Config(variant='v3', enable_semantic_arbitration=True, nms_iou=.40)

    state = prepared(proposals, lines, cfg, semantic=SemanticOracle({'text': .01, 'caption': .99}))

    assert {region.id: region.line_ids for region in state['recovery']} == {'a': ['a_line'], 'b': ['b_line']}
    target, = state['targets']
    assert {fallback['id'] for fallback in target['fallback_baseline']} == {'a', 'b'}


def test_semantic_quality_flag_does_not_enable_prepare_arbitration():
    cfg = Config(variant='v3', enable_semantic_quality=True)
    oracle = SemanticOracle({'text': .01, 'caption': .99})

    state = prepared(conflict(), [line()], cfg, semantic=oracle)

    assert not oracle.calls
    assert state['targets'][0]['reason'] == 'true_conflict'


def test_unavailable_prediction_leaves_same_review_target_and_fallback():
    proposals, lines = conflict(), [line()]
    legacy = prepared(proposals, lines, Config(variant='v3'))
    cfg = Config(variant='v3', enable_semantic_arbitration=True)

    state = prepared(proposals, lines, cfg, semantic=SemanticOracle(reason='unsupported_class'))

    assert state['recovery'] == legacy['recovery']
    assert [target['line_ids'] for target in state['targets']] == [target['line_ids'] for target in legacy['targets']]
    assert [target['fallback_baseline'] for target in state['targets']] == [
        target['fallback_baseline'] for target in legacy['targets']]


@pytest.mark.parametrize(('probability', 'recovered'), [(.59, False), (.60, True)])
def test_weak_recovery_retains_probability_threshold(probability, recovered):
    cfg = Config(variant='v3', enable_semantic_arbitration=True)
    region = proposal('weak', (10, 10, 210, 30), score=.30)
    lines = [line(text='These are ordinary words in a paragraph.')]
    oracle = SemanticOracle({'text': probability, 'caption': 1 - probability})

    state = prepared([region], lines, cfg, semantic=oracle)

    assert oracle.calls
    assert bool(state['recovery']) is recovered


@pytest.mark.parametrize(('cfg_minimum', 'probability', 'abstain_reason'), [
    (.50, .59, None),
    (0., None, 'insufficient_ocr_confidence'),
])
def test_custom_legacy_threshold_cannot_relax_learned_weak_gate(cfg_minimum, probability, abstain_reason):
    cfg = Config(variant='v3', enable_semantic_arbitration=True, min_semantic_score=cfg_minimum)
    region = proposal('weak', (10, 10, 210, 30), score=.30)
    lines = [line(text='These are ordinary words in a paragraph.')]
    distribution = {'text': probability, 'caption': 1 - probability} if probability is not None else None
    oracle = SemanticOracle(distribution, reason=abstain_reason, available=True)

    state = prepared([region], lines, cfg, semantic=oracle)

    assert oracle.calls and not state['recovery']
    assert state['targets'][0]['line_ids'] == ['l']


def test_quality_flag_changes_only_semantic_term_and_retains_original_weights():
    region = proposal('p', (10, 10, 210, 30))
    lines = [line(text='These are ordinary words in a paragraph.')]
    oracle = SemanticOracle({'text': .92, 'caption': .08})
    arbitration_cfg = Config(variant='v3', enable_semantic_arbitration=True)
    quality_cfg = Config(variant='v3', enable_semantic_quality=True)

    legacy_q, legacy = quality([region], lines, lines, PAGE, PAGE, arbitration_cfg, semantic=oracle)
    assert not oracle.calls
    learned_q, learned = quality([region], lines, lines, PAGE, PAGE, quality_cfg, semantic=oracle)

    assert learned['coverage'] == legacy['coverage'] == 1
    assert learned['split_penalty'] == legacy['split_penalty'] == 0
    assert learned['semantic'] == pytest.approx(.92)
    assert legacy_q == pytest.approx(.45 + .25 * .8)
    assert learned_q == pytest.approx(.45 + .25 * .92)


def test_learned_quality_does_not_lower_replacement_gain():
    proposals, lines = conflict(), [line()]
    cfg = Config(variant='v3', enable_semantic_quality=True)
    oracle = SemanticOracle({'text': .34, 'caption': .66})
    state = prepared(proposals, lines, cfg)
    target, = state['targets']
    predictions = response(target, (20, 10, 180, 30), 'caption')

    final, audit, _, decisions = finish_layout(state, lines, predictions, cfg, PAGE, PAGE, semantic=oracle)

    assert [region.label for region in final] == ['text']
    assert decisions[-1]['action'] == 'KEEP'
    assert audit[0]['checkpoints'] == ['STEP4_LOST_TO_KEEP_BASELINE']
    assert 0 < decisions[-1]['replace_score'] - decisions[-1]['keep_score'] < .10


def test_confident_semantics_cannot_override_protected_nontext_polygon():
    bbox = (120, 130, 190, 150)
    lines = [line('orphan', bbox, 'These are ordinary words in a paragraph.')]
    image = proposal('image', (100, 100, 300, 300), 'image', .9,
                     polygon=[[100, 100], [300, 100], [300, 300]])
    cfg = Config(variant='v3', enable_v3_polygons=True, enable_semantic_quality=True)
    state = prepared([image], lines, cfg)
    target, = state['targets']
    oracle = SemanticOracle({'text': .99, 'caption': .01})

    final, audit, _, decisions = finish_layout(state, lines, response(target, bbox, 'text'),
                                               cfg, PAGE, PAGE, semantic=oracle)

    assert [region.id for region in final] == ['image']
    assert decisions[-1]['action'] == 'REVIEW'
    assert 'PROTECTED_NONTEXT_OVERLAP' in audit[0]['boxes'][0]['codes']
    assert not oracle.calls


@pytest.mark.parametrize('exists', [False, True])
def test_missing_or_invalid_artifact_abstains_without_a_fake_probability(tmp_path, exists):
    from src.cascaded.semantic import SemanticArbitrator

    path = tmp_path / 'model'
    if exists:
        path.mkdir()
        (path / 'model_manifest.json').write_text('{"schema_version":"not-a-real-schema"}', encoding='utf-8')
    lines = [line()]
    arbitrator = SemanticArbitrator.from_artifact(path)

    prediction = arbitrator.predict('caption', lines, lines, (10, 10, 210, 30), PAGE, PAGE)

    assert not prediction['available']
    assert prediction['abstain_reason']
    assert not prediction['probabilities']
    assert math.isfinite(prediction['cpu_seconds'])
