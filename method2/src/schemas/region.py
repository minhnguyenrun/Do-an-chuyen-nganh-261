from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field, AfterValidator, field_validator


def valid_bbox(value):
    x1, y1, x2, y2 = value
    if min(value) < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError('bbox must have positive area in page coordinates')
    return value


BBox = Annotated[tuple[int, int, int, int], AfterValidator(valid_bbox)]
Polygon = tuple[tuple[int, int], ...]


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')


class LayoutRegion(FrozenModel):
    id: str
    bbox: BBox
    polygon: Polygon = ()
    label: str
    order: int = 0
    score: float = Field(default=1.0, ge=0, le=1)


class TextLineEvidence(FrozenModel):
    line_id: str
    polygon: Polygon = Field(min_length=3)
    owner_candidates: tuple[str, ...] = ()
    score: float = Field(default=1.0, ge=0, le=1)
    transcript: str = ''

    @field_validator('polygon')
    @classmethod
    def valid_polygon(cls,value):
        from shapely.geometry import Polygon
        polygon = Polygon(value)
        if not polygon.is_valid or polygon.area <= 0 or any(min(p)<0 for p in value):
            raise ValueError('text line polygon must be valid, nonnegative and nonempty')
        return value


class FinalRegion(FrozenModel):
    id: str
    bbox_final: BBox
    label_final: str
    order_final: int
    content: str = ''
    content_format: str = 'text'
    source_region_ids: tuple[str, ...]
    source_line_ids: tuple[str, ...] = ()
    needs_review: bool = False
