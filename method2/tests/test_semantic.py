import asyncio
import numpy as np
from src.schemas.hypothesis import RegionHypothesis, SplitPart
from src.schemas.evidence import ParseEvidence
from src.spatial.region_graph import build_region_graph
from src.skip.high_spatial_skip import HighSpatialSkip
from src.skip.low_spatial_skip import LowSpatialSkip
from src.skip.dual_skip_validator import validate_hard_constraints
from src.semantic.candidate_generator import generate_hypotheses
from src.semantic.semantic_layout_refiner import refine_hypotheses
from src.semantic.evidence_features import extract_table_features, extract_formula_features, calculate_content_consistency
from src.semantic.probe_scheduler import ProbeScheduler
from src.adapters.glm_ocr_local import LocalGlmOcrAdapter
from test_spatial import region, line


def table_fixture():
    rs = [region(str(i),(50+x*60,50+y*30,100+x*60,70+y*30))
          for i,(x,y) in enumerate([(0,0),(1,0),(0,1),(1,1)])]
    ls = [line(f'l{i}',r.bbox,chr(65+i)) for i,r in enumerate(rs)]
    graph = build_region_graph(rs,500,500,lines=ls)
    return rs,ls,graph


def evidence(h,content):
    return ParseEvidence(hypothesis_id=h.hypothesis_id,parser='glm',prompt='Table Recognition:',content=content,status='ok')


HTML = '<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>'


def test_table_merge_has_geometric_support_and_margin():
    rs,ls,g = table_fixture()
    hyps = generate_hypotheses(g)
    merge = next(h for h in hyps if h.action == 'MERGE')
    assert set(merge.source_region_ids) == {r.id for r in rs}
    choices = [h for h in hyps if h.group_id == merge.group_id]
    baseline = next(h for h in choices if h.action == 'KEEP')
    ev = {baseline.hypothesis_id:evidence(baseline,'A B C D'),merge.hypothesis_id:evidence(merge,HTML)}
    high,low = HighSpatialSkip(rs,g),LowSpatialSkip(ls,np.zeros((500,500,3),np.uint8))
    revision = refine_hypotheses('g',choices,ev,high,low,{})
    assert revision.status == 'accepted' and revision.action == 'MERGE'
    assert revision.score-revision.baseline_score >= .1
    ev[merge.hypothesis_id] = evidence(merge,'broken table')
    revision = refine_hypotheses('g',choices,ev,high,low,{})
    assert revision.status == 'abstained' and revision.needs_review


def test_cross_column_and_syntax_only_table_rejected():
    rs = [region('a',(50,50,200,150)),region('b',(250,50,400,150))]
    g = build_region_graph(rs,500,500)
    h = RegionHypothesis(hypothesis_id='h',action='MERGE',source_region_ids=['a','b'],candidate_label='table',candidate_bbox=(50,50,400,150))
    high,low = HighSpatialSkip(rs,g),LowSpatialSkip([],np.zeros((500,500,3),np.uint8))
    passed,flags = validate_hard_constraints(h,high,low,evidence(h,HTML))
    assert not passed and 'cross_column_violation' in flags
    h = h.model_copy(update={'action':'RELABEL','source_region_ids':('a',),'candidate_bbox':rs[0].bbox})
    assert not validate_hard_constraints(h,high,low,evidence(h,HTML))[0]


def test_split_must_preserve_all_lines_and_nonoverlap():
    r = region('a',(10,10,150,150))
    ls = [line('title',(10,10,80,30),'Heading'),line('body',(10,80,140,100),'Body text')]
    g = build_region_graph([r],200,200,lines=ls)
    hyps = generate_hypotheses(g)
    split = next(h for h in hyps if h.action == 'SPLIT')
    low,high = LowSpatialSkip(ls,np.zeros((200,200,3),np.uint8)),HighSpatialSkip([r],g)
    e = evidence(split,'Heading Body text')
    e = e.model_copy(update={'part_contents':('Heading','Body text')})
    assert validate_hard_constraints(split,high,low,e)[0]
    bad = split.model_copy(update={'split_parts':split.split_parts[:1]})
    assert not validate_hard_constraints(bad,high,low,e)[0]


def test_features_reject_invalid_structure_and_detect_content_loss():
    assert extract_table_features(HTML,4)['cell_count'] == 4
    assert not extract_table_features('<table><tr><td>A',1)['is_valid_html']
    assert extract_formula_features(r'\frac{a}{b}=\sum_i x_i')['is_balanced']
    assert not extract_formula_features(r'\frac{a}{b')['is_balanced']
    assert calculate_content_consistency('Hello World',['Hello','World']) == 1
    assert calculate_content_consistency('A',['A','B','C','D']) < .6


def test_scheduler_budget_and_prompt_validation():
    async def run():
        scheduler = ProbeScheduler(max_glm_requests_per_page=2,max_in_flight=1)
        client = LocalGlmOcrAdapter(mock_mode=True)
        req = [(str(i),np.zeros((10,10,3),np.uint8),'Text Recognition:') for i in range(5)]
        ev = await scheduler.run(client,req)
        assert scheduler.used == 2
        assert sum(e.status == 'budget_exhausted' for e in ev) == 3
        assert all(e.status == 'budget_exhausted' for e in await scheduler.run(client,req))
    asyncio.run(run())
