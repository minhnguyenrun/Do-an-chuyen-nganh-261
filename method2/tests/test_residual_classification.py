import asyncio
import json

import numpy as np
import pytest

from src.pipeline import parse_page
from src.schemas.evidence import ParseEvidence, RegionClassification
from src.schemas.hypothesis import RegionHypothesis
from src.schemas.region import LayoutRegion
from src.semantic.residual_prompt import (build_classification_prompt, is_classification_prompt,
    parse_classification, parse_classification_with_note, classification_instruction)
from src.semantic.residual_labeling import build_context_crop, residual_role_evidence
from src.semantic.probe_scheduler import ProbeScheduler
from src.skip.dual_skip_validator import owned_lines
from src.skip.high_spatial_skip import HighSpatialSkip
from src.skip.low_spatial_skip import LowSpatialSkip
from src.spatial.region_graph import build_region_graph
from test_spatial import line


class Layout:
    def detect(self, image):
        return []


class Detector:
    def __init__(self, bbox=(20, 5, 150, 20), text='Annual report company'):
        self.lines = [line('line', bbox, text)]
    def detect_lines(self, image):
        return self.lines


class Light:
    def __init__(self, text='Annual report company'):
        self.text = text
    def recognize(self, crops):
        return [self.text for _ in crops]


class Classifier:
    def __init__(self, label='header', text='Annual report company', raw=None, status='ok'):
        self.label, self.text, self.raw, self.status = label, text, raw, status
        self.prompts = []
    async def recognize_batch(self, requests):
        result = []
        for id, image, prompt in requests:
            self.prompts.append(prompt)
            if is_classification_prompt(prompt):
                assert image.ndim == 3 and image.shape[2] == 3 and max(image.shape[:2]) <= 1280
                content = self.raw if self.raw is not None else json.dumps(dict(label=self.label, text=self.text))
            else:
                content = self.text
            result.append(ParseEvidence(hypothesis_id=id, parser='glm_local', prompt=prompt,
                                        content=content, status=self.status))
        return result


def run(classifier, *, bbox=(20, 5, 150, 20), text='Annual report company', config=None):
    return asyncio.run(parse_page(np.full((200, 200, 3), 255, np.uint8), config or {},
                                 Layout(), Detector(bbox, text), Light(text), classifier))


def test_header_classification_is_applied_with_provenance_and_ocr_preserved():
    classifier = Classifier()
    result = run(classifier)
    assert len(result.final_regions) == 1
    final = result.final_regions[0]
    assert final.label_final == 'header' and final.content == 'Annual report company'
    assert final.source_region_ids == (result.recovered_regions[0].id,)
    assert final.source_line_ids == ('line',)
    assert result.recovered_regions[0].label == 'residual'
    assert result.revisions[0].status == 'accepted'
    assert 'top_page_margin' in result.revisions[0].reasons
    assert result.glm_requests == 1
    assert any(e.classification and e.classification.label == 'header' for e in result.evidence)
    assert result.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize('classifier', [Classifier(raw='not JSON'), Classifier(label='unknown'),
    Classifier(label='header', text='hallucinated unrelated content'), Classifier(status='incomplete'),
    Classifier(raw='{"label":"not_a_layout_label","text":"Annual report company"}')])
def test_uncertain_invalid_or_mismatched_classification_abstains(classifier):
    result = run(classifier)
    assert result.final_regions[0].label_final == 'residual'
    assert result.final_regions[0].needs_review
    assert result.revisions[0].status == 'abstained'
    assert result.final_regions[0].content == 'Annual report company'


def test_classifier_cannot_make_middle_of_page_text_a_footer():
    result = run(Classifier(label='footer'), bbox=(20, 80, 150, 95))
    assert result.final_regions[0].label_final == 'residual'
    assert 'classification_spatial_mismatch:footer' in result.revisions[0].reasons


@pytest.mark.parametrize('label,bbox,text', [
    ('footer', (20, 180, 150, 195), 'Annual report company'),
    ('list_item', (20, 80, 150, 95), '1. Some important item'),
    ('text', (20, 80, 150, 95), 'This body paragraph contains several important words'),
])
def test_supported_roles_are_applied(label, bbox, text):
    result = run(Classifier(label=label, text=text), bbox=bbox, text=text)
    assert result.final_regions[0].label_final == label


def test_classification_does_not_replace_existing_ocr_text():
    text = 'Annual report company'
    result = run(Classifier(text='Annual report company.'), text=text)
    assert result.final_regions[0].label_final == 'header'
    assert result.final_regions[0].content == text


@pytest.mark.parametrize('first_status', ['ok', 'incomplete'])
def test_bad_schema_or_truncation_retries_once_and_keeps_original_evidence(first_status):
    class RetryClassifier(Classifier):
        async def recognize_batch(self, requests):
            self.raw = '{"title":"Report","properties":{"label":"text"},"type":"object"}'
            self.status = first_status
            if all(json.loads(prompt).get('retry') for _, _, prompt in requests):
                self.raw, self.status = None, 'ok'
            return await super().recognize_batch(requests)
    result = run(RetryClassifier())
    assert result.final_regions[0].label_final == 'header'
    assert result.glm_requests == 2
    assert any(e.hypothesis_id == 'classify:recovered:0:retry' and e.classification for e in result.evidence)
    assert any(e.status in ('invalid_classification', 'incomplete') for e in result.evidence)


def test_classification_retry_cannot_exceed_page_budget():
    result = run(Classifier(raw='not JSON'), config={'inference': {'max_glm_requests_per_page': 1}})
    assert result.glm_requests == 1
    assert result.final_regions[0].label_final == 'residual'
    assert any(e.hypothesis_id.endswith(':retry') and e.status == 'budget_exhausted' for e in result.evidence)


@pytest.mark.parametrize('text', [
    "While they may be the biggest\nbank in town, they sure\ndon't act like it.\ndon't\nIt's like banking\nwith friends.",
    'While they may be the biggest bank in town, they sure don’t act like it. It’s like banking with friends.',
    'Annual report company',
])
def test_observed_extra_final_quote_is_normalized_without_changing_text(text):
    valid = json.dumps({'label': 'text', 'text': text}, ensure_ascii=False, separators=(',', ':'))
    raw = valid[:-1] + '"}'
    classification, note = parse_classification_with_note(raw)
    assert classification.label == 'text' and classification.text == text
    assert note == 'normalized_json:removed_redundant_final_quote'
    assert parse_classification_with_note(valid)[1] == ''


def test_normalization_keeps_raw_answer_and_does_not_bypass_independent_checks():
    raw = '{"label":"header","text":"Annual report company""}'
    result = run(Classifier(raw=raw))
    assert result.glm_requests == 1  # No repeated identical inference for this syntax defect.
    assert result.final_regions[0].label_final == 'header'
    evidence = next(e for e in result.evidence if e.hypothesis_id == 'classify:recovered:0')
    assert evidence.content == raw and evidence.error.startswith('normalized_json:')
    mismatch = run(Classifier(raw='{"label":"header","text":"unrelated invented words""}'))
    assert mismatch.final_regions[0].label_final == 'residual'
    assert mismatch.revisions[0].status == 'abstained'


@pytest.mark.parametrize('raw', [
    '{"label":"text" "text":"Report"}',  # missing comma
    '{"label":"text","text":"He said "hello" today"}',  # internal unescaped quotes
    '{"label":"text","text":"Report',  # truncated
    '{"label":"other","text":"Report""}',  # invalid label
    '{"title":"Report","properties":{"label":"text"},"text":"Report""}',
    '{"text":"Report","label":"text""}',  # repair applies only to final text field
])
def test_other_json_errors_are_not_guessed_or_silently_repaired(raw):
    with pytest.raises(ValueError):
        parse_classification(raw)


def test_valid_escaped_quotes_are_preserved():
    text = 'He said "hello". Last character is a quote: "'
    parsed, note = parse_classification_with_note(json.dumps(dict(label='text', text=text)))
    assert parsed.text == text and not note


def test_table_classification_requires_canonical_reparse_and_grid_validation():
    html = '<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>'
    class UnknownLayout:
        def detect(self, image):
            return [LayoutRegion(id='unknown', label='unknown', bbox=(20, 40, 150, 100))]
    class GridDetector:
        def detect_lines(self, image):
            return [line('a', (20, 40, 70, 60), 'A'), line('b', (90, 40, 150, 60), 'B'),
                    line('c', (20, 80, 70, 100), 'C'), line('d', (90, 80, 150, 100), 'D')]
    class GridLight:
        def recognize(self, crops):
            return ['A', 'B', 'C', 'D']
    class TableClassifier(Classifier):
        async def recognize_batch(self, requests):
            if requests[0][2] == 'Table Recognition:':
                self.prompts.append('Table Recognition:')
                return [ParseEvidence(hypothesis_id=id, parser='glm_local', prompt=prompt, content=html, status='ok')
                        for id, _, prompt in requests]
            return await super().recognize_batch(requests)
    classifier = TableClassifier(label='table', text='A B C D')
    result = asyncio.run(parse_page(np.full((200, 200, 3), 255, np.uint8), {}, UnknownLayout(),
                                   GridDetector(), GridLight(), classifier))
    assert result.final_regions[0].label_final == 'table'
    assert result.final_regions[0].content == html
    assert result.glm_requests == 2
    assert 'Table Recognition:' in classifier.prompts


def test_caption_can_be_beside_a_picture_but_not_deep_inside_it():
    visual = LayoutRegion(id='image', bbox=(100, 10, 195, 150), label='image')
    target = LayoutRegion(id='r', bbox=(20, 20, 85, 40), label='residual')
    lines = [line('caption', target.bbox, 'Figure 1. Company headquarters')]
    graph = build_region_graph([visual, target], 200, 200, lines=lines)
    hyp = RegionHypothesis(hypothesis_id='h', action='RELABEL', source_region_ids=['r'],
                           candidate_label='figure_title', candidate_bbox=target.bbox)
    assert residual_role_evidence(hyp, graph, lines)[0]
    inside = hyp.model_copy(update={'candidate_bbox': (110, 20, 180, 40)})
    assert not residual_role_evidence(inside, graph, lines)[0]


def test_page_budget_and_disable_toggle_are_respected():
    result = run(Classifier(), config={'inference': {'max_glm_requests_per_page': 0}})
    assert result.glm_requests == 0 and result.final_regions[0].needs_review
    assert any(e.status == 'budget_exhausted' for e in result.evidence)
    classifier = Classifier()
    result = run(classifier, config={'routing': {'classify_unlabeled_regions': False}})
    assert not classifier.prompts and result.final_regions[0].label_final == 'residual'


def test_schema_task_is_narrow_and_context_does_not_modify_page():
    region = LayoutRegion(id='r', bbox=(20, 5, 150, 20), label='residual')
    prompt = build_classification_prompt(region, (200, 200), 'Annual report company')
    assert is_classification_prompt(prompt) and not is_classification_prompt('Classify anything')
    assert parse_classification('```json\n{"label":"header","text":"Report"}\n```').label == 'header'
    with pytest.raises(ValueError):
        parse_classification('{"label":"header","text":"Report","confidence":1}')
    image = np.full((200, 200, 3), 150, np.uint8)
    assert build_context_crop(image, region.bbox).shape == (45, 390, 3)
    assert np.all(image == 150)


def test_target_crop_excludes_all_other_page_pixels_and_keeps_bgr():
    image = np.full((200, 200, 3), (255, 0, 0), np.uint8)
    image[31:68, 20:79] = (0, 0, 255)
    crop = build_context_crop(image, (20, 31, 79, 68))
    assert crop.shape == (111, 177, 3)
    assert np.all(crop == (0, 0, 255))
    assert np.all(image[:31] == (255, 0, 0))
    region = LayoutRegion(id='r', bbox=(20, 31, 79, 68), label='residual')
    prompt = build_classification_prompt(region, (200, 200), 'Target text')
    instruction = classification_instruction(prompt)
    assert json.loads(prompt)['bbox'] == [.1, .155, .395, .34]
    assert 'only one target region' in instruction
    assert 'left panel' not in instruction and 'right panel' not in instruction


def test_owned_lines_follow_reading_order_even_when_detector_returns_bottom_first():
    region = LayoutRegion(id='r', bbox=(10, 10, 150, 100), label='residual')
    lines = [line('bottom', (20, 60, 140, 80), 'Second line'), line('top', (20, 20, 140, 40), 'First line')]
    graph = build_region_graph([region], 200, 200, lines=lines)
    hyp = RegionHypothesis(hypothesis_id='h', action='KEEP', source_region_ids=['r'],
                           candidate_label='residual', candidate_bbox=region.bbox)
    owned = owned_lines(hyp, HighSpatialSkip([region], graph), LowSpatialSkip(lines, np.zeros((200, 200, 3), np.uint8)))
    assert [l.line_id for l in owned] == ['top', 'bottom']
