import pytest
from pydantic import ValidationError

from src.schemas.region import LayoutRegion, TextLineEvidence
from src.schemas.output import PageParseResult


def test_deeply_immutable_and_serializable():
    polygon = [(0, 0), (20, 0), (20, 10), (0, 10)]
    region = LayoutRegion(id='r', bbox=(0, 0, 20, 10), polygon=polygon, label='text')
    polygon.clear()
    assert len(region.polygon) == 4
    with pytest.raises(ValidationError):
        region.label = 'table'
    with pytest.raises(TypeError):
        region.polygon[0] = (2, 2)
    result = PageParseResult(original_regions=[region])
    assert PageParseResult.model_validate_json(result.model_dump_json()) == result


def test_invalid_geometry_and_duplicate_ids():
    with pytest.raises(ValidationError):
        LayoutRegion(id='r', bbox=(10, 0, 0, 10), label='text')
    with pytest.raises(ValidationError):
        TextLineEvidence(line_id='l', polygon=[])
