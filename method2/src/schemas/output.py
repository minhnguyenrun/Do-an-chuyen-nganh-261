from pydantic import Field, model_validator
from .region import FrozenModel, LayoutRegion, TextLineEvidence, FinalRegion, BBox
from .hypothesis import Action, RevisionStatus, SplitPart
from .evidence import ParseEvidence


class RegionRevision(FrozenModel):
    revision_id: str
    action: Action
    source_region_ids: tuple[str, ...]
    target_region_id: str
    new_label: str
    new_bbox: BBox
    status: RevisionStatus
    reasons: tuple[str, ...] = ()
    hypothesis_id: str = ''
    baseline_score: float = 0.0
    score: float = 0.0
    split_parts: tuple[SplitPart, ...] = ()
    needs_review: bool = False


class Provenance(FrozenModel):
    region_id: str
    source_region_ids: tuple[str, ...]
    source_line_ids: tuple[str, ...] = ()


class PageParseResult(FrozenModel):
    document_id: str = 'document'
    page_index: int = 0
    original_regions: tuple[LayoutRegion, ...] = ()
    original_lines: tuple[TextLineEvidence, ...] = ()
    recovered_regions: tuple[LayoutRegion, ...] = ()
    final_regions: tuple[FinalRegion, ...] = ()
    revisions: tuple[RegionRevision, ...] = ()
    evidence: tuple[ParseEvidence, ...] = ()
    provenance: tuple[Provenance, ...] = ()
    warnings: tuple[str, ...] = ()
    markdown: str = ''
    glm_requests: int = Field(default=0, ge=0)

    @model_validator(mode='after')
    def check_lineage(self):
        for collection, key in ((self.original_regions, 'id'), (self.original_lines, 'line_id'),
                                (self.recovered_regions, 'id'), (self.final_regions, 'id')):
            ids = [getattr(item, key) for item in collection]
            if len(ids) != len(set(ids)):
                raise ValueError('duplicate evidence or region IDs')
        known = {r.id for r in (*self.original_regions, *self.recovered_regions)}
        seen_lines = set()
        for region in self.final_regions:
            if not region.source_region_ids or not set(region.source_region_ids) <= known:
                raise ValueError('final region has unknown provenance')
            if seen_lines.intersection(region.source_line_ids):
                raise ValueError('duplicate line in final regions')
            seen_lines.update(region.source_line_ids)
        return self
