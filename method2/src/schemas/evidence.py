from .region import FrozenModel
from typing import Literal


class RegionClassification(FrozenModel):
    label: Literal['text', 'title', 'figure_title', 'header', 'footer', 'footnote',
                   'list_item', 'table', 'formula', 'unknown']
    text: str


class ParseEvidence(FrozenModel):
    hypothesis_id: str
    parser: str
    prompt: str
    content: str
    status: str
    error: str = ''
    part_contents: tuple[str, ...] = ()
    classification: RegionClassification | None = None


class GeometryEvidence(FrozenModel):
    coverage: float = 0.0
    alignment: float = 0.0
    reasons: tuple[str, ...] = ()


class ValidationResult(FrozenModel):
    passed: bool
    flags: tuple[str, ...] = ()
