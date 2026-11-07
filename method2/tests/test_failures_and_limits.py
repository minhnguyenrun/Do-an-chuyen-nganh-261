import asyncio
import numpy as np
import pytest
from src.pipeline import parse_page,load_config
from src.schemas.evidence import ParseEvidence
from src.semantic.probe_scheduler import ProbeScheduler
from src.semantic.semantic_layout_refiner import refine_hypotheses
from src.semantic.candidate_generator import generate_hypotheses
from src.skip.high_spatial_skip import HighSpatialSkip
from src.skip.low_spatial_skip import LowSpatialSkip
from src.skip.dual_skip_validator import validate_hard_constraints
from src.adapters.glm_ocr_local import LocalGlmOcrAdapter
from test_spatial import region,line
from test_semantic import table_fixture,evidence,HTML


def test_small_score_gain_abstains_even_valid_table():
    rs,ls,g = table_fixture()
    choices = generate_hypotheses(g)
    ev = {h.hypothesis_id:evidence(h,HTML if h.action == 'MERGE' else 'A B C D') for h in choices}
    revision = refine_hypotheses('g',choices,ev,HighSpatialSkip(rs,g),LowSpatialSkip(ls,np.zeros((500,500,3),np.uint8)),{'min_score_gain':.99})
    assert revision.action == 'KEEP' and revision.status == 'abstained'


def test_competing_high_scoring_alternatives_abstain():
    rs,ls,g = table_fixture()
    choices = generate_hypotheses(g)
    alternative = next(h for h in choices if h.action == 'MERGE')
    competing = alternative.model_copy(update={'hypothesis_id':'competing'})
    choices.append(competing)
    ev = {h.hypothesis_id:evidence(h,HTML if h.action == 'MERGE' else 'A B C D') for h in choices}
    rev = refine_hypotheses('g',choices,ev,HighSpatialSkip(rs,g),LowSpatialSkip(ls,np.zeros((500,500,3),np.uint8)),{})
    assert rev.status == 'abstained' and 'ambiguous_alternatives' in rev.reasons


def test_candidate_and_inference_limits_cannot_exceed_v4_caps():
    rs,ls,g = table_fixture()
    hypotheses = generate_hypotheses(g,100,100)
    for group in {h.group_id for h in hypotheses}:
        assert sum(h.group_id == group for h in hypotheses) <= 3
    scheduler = ProbeScheduler(100,2)
    assert scheduler.limit == 30
    with pytest.raises(ValueError):
        load_config({'hypotheses':{'max_iterations':2}})
    with pytest.raises(ValueError):
        load_config({'validation':{'min_score_gain':0}})


def test_budget_shared_between_simultaneous_batches_and_errors_are_recorded():
    class Failing:
        async def recognize_batch(self,requests):
            raise RuntimeError('offline')
    async def run():
        scheduler = ProbeScheduler(3,2)
        req = [(str(i),np.zeros((5,5,3),np.uint8),'Text Recognition:') for i in range(3)]
        batches = await asyncio.gather(scheduler.run(Failing(),req),scheduler.run(Failing(),req))
        assert scheduler.used == 3
        flat = [e for b in batches for e in b]
        assert sum(e.status == 'error' for e in flat) == 3
        assert sum(e.status == 'budget_exhausted' for e in flat) == 3
    asyncio.run(run())


def test_page_budget_includes_original_parsing_and_never_calls_light_when_disabled():
    class Layout:
        def detect(self,image):
            return [region(f'r{i}',(0,i*10,50,i*10+8)) for i in range(35)]
    class Detector:
        def detect_lines(self,image):
            return []
    class ForbiddenLight:
        def recognize(self,crops):
            raise AssertionError('light OCR must be disabled')
    result = asyncio.run(parse_page(np.zeros((400,100,3),np.uint8),{'routing':{'enable_light_ocr':False}},
        Layout(),Detector(),ForbiddenLight(),LocalGlmOcrAdapter(mock_mode=True)))
    assert result.glm_requests == 30 and len(result.final_regions) == 35
    assert sum(not r.content for r in result.final_regions) == 5
    assert any('budget_exhausted' in warning for warning in result.warnings)


def test_residual_line_recovery_has_explicit_lineage():
    class EmptyLayout:
        def detect(self,image):
            return []
    class Detector:
        def detect_lines(self,image):
            return [line('l',(10,10,40,30))]
    class Light:
        def recognize(self,crops):
            return ['Recovered']
    result = asyncio.run(parse_page(np.zeros((100,100,3),np.uint8),{},EmptyLayout(),Detector(),Light(),LocalGlmOcrAdapter(mock_mode=True)))
    assert not result.original_regions and len(result.recovered_regions) == 1
    assert result.final_regions[0].source_region_ids == (result.recovered_regions[0].id,)
    assert result.final_regions[0].source_line_ids == ('l',)


def test_formula_with_valid_syntax_but_hallucinated_tokens_rejected():
    from src.spatial.region_graph import build_region_graph
    from src.schemas.hypothesis import RegionHypothesis
    rs = [region('a',(10,10,150,40))]
    ls = [line('l',rs[0].bbox,r'x^2 + y^2 = z^2')]
    graph = build_region_graph(rs,200,200,lines=ls)
    h = RegionHypothesis(hypothesis_id='h',action='RELABEL',source_region_ids=['a'],candidate_label='formula',candidate_bbox=rs[0].bbox)
    passed,flags = validate_hard_constraints(h,HighSpatialSkip(rs,graph),LowSpatialSkip(ls,np.zeros((200,200,3),np.uint8)),evidence(h,'a+b=c'))
    assert not passed and 'formula_content_mismatch' in flags
