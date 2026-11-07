import asyncio
import cv2
import numpy as np
from src.pipeline import parse_page
from src.schemas.evidence import ParseEvidence
from src.schemas.hypothesis import RegionHypothesis
from src.skip.high_spatial_skip import HighSpatialSkip
from src.skip.low_spatial_skip import LowSpatialSkip
from src.skip.dual_skip_validator import validate_hard_constraints
from src.spatial.region_graph import build_region_graph
from src.semantic.candidate_generator import generate_hypotheses
from src.semantic.semantic_layout_refiner import refine_hypotheses
from src.adapters.layout_ppdocv3 import LayoutDetectorAdapter
from src.adapters.text_detector_ppocr import TextDetectorAdapter
from src.adapters.light_ocr_ppocr import LightOcrAdapter
from src.adapters.glm_ocr_local import LocalGlmOcrAdapter
from test_spatial import region,line
from test_semantic import table_fixture,HTML


def test_formula_relabel_requires_independent_math_and_score_gain():
    r = region('a',(10,10,150,40))
    ls = [line('l',r.bbox,r'x^2 + y^2 = z^2')]
    graph = build_region_graph([r],200,200,lines=ls)
    choices = generate_hypotheses(graph)
    ev = {h.hypothesis_id:ParseEvidence(hypothesis_id=h.hypothesis_id,parser='glm',prompt='Formula Recognition:',
        content=ls[0].transcript,status='ok') for h in choices}
    high,low = HighSpatialSkip([r],graph),LowSpatialSkip(ls,np.zeros((200,200,3),np.uint8))
    revision = refine_hypotheses('g',choices,ev,high,low,{})
    assert revision.action == 'RELABEL' and revision.status == 'accepted'
    low = LowSpatialSkip([ls[0].model_copy(update={'transcript':'ordinary prose'})],low.image)
    h = next(h for h in choices if h.action == 'RELABEL')
    assert not validate_hard_constraints(h,high,low,ev[h.hypothesis_id])[0]


def test_end_to_end_table_merge_retains_all_lines_once():
    rs,ls,_ = table_fixture()
    class Layout:
        def detect(self,image):
            return rs
    class Detector:
        def detect_lines(self,image):
            return ls
    class Light:
        def recognize(self,crops):
            return ['A','B','C','D']
    class Glm:
        async def recognize_batch(self,requests):
            return [ParseEvidence(hypothesis_id=id,parser='glm',prompt=prompt,content=HTML,status='ok') for id,_,prompt in requests]
    result = asyncio.run(parse_page(np.zeros((500,500,3),np.uint8),{},Layout(),Detector(),Light(),Glm()))
    assert len(result.final_regions) == 1
    final = result.final_regions[0]
    assert final.label_final == 'table'
    assert set(final.source_region_ids) == {r.id for r in rs}
    assert set(final.source_line_ids) == {l.line_id for l in ls}
    assert result.original_regions == tuple(rs)
    assert result.original_lines == tuple(ls)


def test_end_to_end_split_parses_each_child_and_preserves_lineage():
    r = region('a',(10,10,150,150))
    ls = [line('t',(10,10,80,30),'Heading'),line('b',(10,80,140,100),'Body text')]
    class Layout:
        def detect(self,image):
            return [r]
    class Detector:
        def detect_lines(self,image):
            return ls
    class Light:
        def recognize(self,crops):
            return ['Heading','Body text']
    class Glm:
        async def recognize_batch(self,requests):
            return [ParseEvidence(hypothesis_id=id,parser='glm',prompt=prompt,
                content='Heading' if id.endswith(':0') else 'Body text',status='ok') for id,_,prompt in requests]
    result = asyncio.run(parse_page(np.zeros((200,200,3),np.uint8),{},Layout(),Detector(),Light(),Glm()))
    assert len(result.final_regions) == 2
    assert [r.content for r in result.final_regions] == ['Heading','Body text']
    assert all(r.source_region_ids == ('a',) for r in result.final_regions)


def test_notebook_progress_and_document_metadata():
    adapters = (LayoutDetectorAdapter(mock_mode=True),TextDetectorAdapter(mock_mode=True),
                LightOcrAdapter(mock_mode=True),LocalGlmOcrAdapter(mock_mode=True))
    events = []
    result = asyncio.run(parse_page(np.full((100,100,3),255,np.uint8),
        {'document_id':'test','page_index':2},*adapters,progress_callback=lambda stage,message:events.append(stage)))
    assert events[0] == 'spatial' and events[-1] == 'done'
    assert result.document_id == 'test' and result.page_index == 2
