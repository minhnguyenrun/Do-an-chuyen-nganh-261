from enum import Enum
from pydantic import Field
from .region import FrozenModel, BBox, Polygon


class Action(str, Enum):
    KEEP = 'KEEP'
    MERGE = 'MERGE'
    SPLIT = 'SPLIT'
    RELABEL = 'RELABEL'
    REFINE_BBOX = 'REFINE_BBOX'


class RevisionStatus(str, Enum):
    ACCEPTED = 'accepted'
    REJECTED = 'rejected'
    ABSTAINED = 'abstained'
    ROLLED_BACK = 'rolled_back'


class SplitPart(FrozenModel):
    id: str
    bbox: BBox
    label: str
    line_ids: tuple[str, ...]


class RegionHypothesis(FrozenModel):
    hypothesis_id: str
    group_id: str = ''
    action: Action
    source_region_ids: tuple[str, ...] = Field(min_length=1)
    candidate_label: str
    candidate_bbox: BBox
    candidate_polygon: Polygon | None = None
    source_line_ids: tuple[str, ...] = ()
    split_parts: tuple[SplitPart, ...] = ()
    geometry_reasons: tuple[str, ...] = ()
