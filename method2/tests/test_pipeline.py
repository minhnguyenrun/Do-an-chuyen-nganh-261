import asyncio
import numpy as np
from src.pipeline import parse_page
from src.adapters.layout_ppdocv3 import LayoutDetectorAdapter
from src.adapters.text_detector_ppocr import TextDetectorAdapter
from src.adapters.light_ocr_ppocr import LightOcrAdapter
from src.adapters.glm_ocr_local import LocalGlmOcrAdapter
from src.skip.revision_assembler import apply_revisions_to_graph, assemble_document
from src.schemas.output import RegionRevision
from src.spatial.region_graph import build_region_graph
from src.evaluation.evaluate_refinement import compute_refinement_metrics, run_ablation_suite
from src.evaluation.evaluate_parsing import compute_reading_order_ned
from test_spatial import region


def test_pipeline_offline():
    result = asyncio.run(parse_page(np.full((600,600,3),255,np.uint8),{},
        LayoutDetectorAdapter(mock_mode=True),TextDetectorAdapter(mock_mode=True),
        LightOcrAdapter(mock_mode=True),LocalGlmOcrAdapter(mock_mode=True)))
    assert result.final_regions and result.markdown
    assert result.original_lines and result.original_regions
    assert result.glm_requests <= 30
    assert result.model_validate_json(result.model_dump_json()) == result


def test_revision_assembly_and_rollback():
    rs = [region('a',(10,10,30,30)),region('b',(35,10,55,30))]
    g = build_region_graph(rs,100,100)
    rev = RegionRevision(revision_id='r',action='MERGE',source_region_ids=['a','b'],target_region_id='t',new_label='table',new_bbox=(10,10,55,30),status='accepted')
    final = apply_revisions_to_graph(g,[rev])
    result = assemble_document(rs,final,{'t':'<table></table>'},[rev],[])
    assert len(result.final_regions) == 1
    assert result.final_regions[0].source_region_ids == ('a','b')
    assert set(g.nodes) == {'a','b'}
    bad = rev.model_copy(update={'new_bbox':(10,10,20,20)})
    rolled = apply_revisions_to_graph(g,[bad])
    assert set(rolled.nodes) == {'a','b'} and rolled.warnings


def test_metrics_count_duplicate_predictions_as_false_positives():
    rev = RegionRevision(revision_id='r',action='RELABEL',source_region_ids=['a'],target_region_id='a',new_label='formula',new_bbox=(0,0,10,10),status='accepted')
    gt = [dict(action='RELABEL',source_region_ids=['a'],new_label='formula')]
    metrics = compute_refinement_metrics([rev,rev],gt)
    assert metrics['relabel_precision'] == .5
    assert metrics['over_correction_count'] == 1
    assert compute_reading_order_ned(['a','b'],['a','b']) == 1
    assert compute_reading_order_ned(['a','b'],['b','a']) < 1
