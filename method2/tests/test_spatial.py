import numpy as np
from src.schemas.region import LayoutRegion, TextLineEvidence
from src.spatial.crop_builder import build_crop, expand_bbox
from src.spatial.line_assignment import assign_lines_to_regions
from src.spatial.residual_regions import cluster_residual_lines, detect_residual_lines
from src.spatial.region_graph import build_region_graph
from src.spatial.reading_order import reconcile_order_as_dag
from src.skip.high_spatial_skip import HighSpatialSkip
from src.skip.low_spatial_skip import LowSpatialSkip


def region(id, bbox, **kwargs):
    return LayoutRegion(id=id, bbox=bbox, label='text', **kwargs)


def line(id, bbox, transcript=''):
    a,b,c,d = bbox
    return TextLineEvidence(line_id=id, polygon=[(a,b),(c,b),(c,d),(a,d)], transcript=transcript)


def test_mask_preserves_lines_and_raw_fallback():
    image = np.full((300,300,3),128,np.uint8)
    bbox = expand_bbox((100,100,200,200), .05,300,300)
    assert bbox == (95,95,205,205)
    poly = [(100,100),(200,100),(200,200),(100,200)]
    crop = build_crop(image,bbox,poly)
    assert crop[0,0,0] == 255 and crop[10,10,0] == 128
    extra = line('l',(96,96,100,100)).polygon
    assert build_crop(image,bbox,poly,line_polygons=[extra])[2,2,0] == 128
    assert np.all(build_crop(image,bbox,poly,mask_certain=False) == 128)


def test_ownership_is_unique_and_recovers_residuals():
    rs = [region('a',(10,10,100,60)),region('b',(10,10,100,60))]
    ls = [line('l',(20,20,90,40)),line('lost',(20,200,90,220))]
    own = assign_lines_to_regions(rs,ls)
    assert sum(len(v) for v in own.values()) == 1
    recovered = cluster_residual_lines(detect_residual_lines(ls,{'l'}))
    assert recovered[0].bbox == (20,200,90,220)


def test_columns_and_cycle_recovery():
    rs = [region('a',(50,50,200,150)),region('b',(50,170,200,300)),
          region('c',(250,50,400,150)),region('d',(250,170,400,300))]
    graph = build_region_graph(rs,500,500)
    assert graph.is_cross_column('a','c')
    assert reconcile_order_as_dag(graph) == ['a','b','c','d']
    graph.edges.add(('d','a'))
    assert set(reconcile_order_as_dag(graph)) == {'a','b','c','d'}
    assert graph.warnings


def test_skip_snapshots_cannot_change():
    rs = [region('a',(1,1,10,10))]
    graph = build_region_graph(rs,20,20)
    high = HighSpatialSkip(rs,graph)
    graph.nodes.clear()
    assert 'a' in high.graph.nodes
    image = np.zeros((20,20,3),np.uint8)
    low = LowSpatialSkip([],image)
    image[:] = 255
    assert low.image[0,0,0] == 0
    import pytest
    with pytest.raises(ValueError):
        low.image[0,0] = 12
