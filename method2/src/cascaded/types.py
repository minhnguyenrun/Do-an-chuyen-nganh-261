"""JSON contracts for the layout-only experiment (scores are never fabricated)."""
from dataclasses import asdict, dataclass, field
import math


@dataclass
class Config:
    variant: str = 's'
    low: float = .10
    high: float = .45
    nms_iou: float = .50
    line_coverage: float = .80
    max_crops: int = 8
    crop_edge: int = 768
    max_tokens: int = 512
    batch_size: int = 1
    device: str = 'cuda'
    paddle_device: str = 'gpu:0'
    mineru_model: str = 'opendatalab/MinerU2.5-2509-1.2B'
    yolo_model: str = 'juliozhao/DocLayout-YOLO-DocStructBench'
    yolo_file: str = 'doclayout_yolo_docstructbench_imgsz1024.pt'
    mock: bool = False
    yolo_max_area: float = .02
    yolo_max_height: float = .08
    enable_yolo: bool = True
    enable_recovery: bool = True
    enable_mineru: bool = True
    enable_wbf: bool = False
    enable_geometry_refinement: bool = False
    enable_adaptive_inference: bool = False
    min_replace_gain: float = .10
    min_semantic_score: float = .60
    protected_iou: float = .15
    max_foreign_ink: float = .10
    max_nontext_overlap: float = .15
    snap_cov_x: float = .85
    snap_cov_y: float = .50
    snap_center_margin: float = .10
    orphan_line_fraction: float = .80
    partition_iou: float = .30
    partition_candidates: int = 24
    partition_beam: int = 32
    enable_v3_polygons: bool = False
    enable_v3_reading_order: bool = False
    enable_semantic_arbitration: bool = False
    enable_semantic_quality: bool = False
    semantic_artifact_path: str | None = None

    def __post_init__(self):
        if self.variant not in ('s', 'v3') or self.device not in ('cuda', 'cpu'):
            raise ValueError('variant must be s/v3; device must be cuda/cpu')
        if not 0 <= self.low < self.high <= 1 or not 0 < self.line_coverage <= 1:
            raise ValueError('invalid gating thresholds')
        if not 0 < self.nms_iou <= 1 or self.max_crops < 0 or self.crop_edge < 28 or self.max_tokens < 1:
            raise ValueError('invalid crop/inference limits')
        if self.batch_size != 1:
            raise ValueError('Windows 4GB profile uses batch_size=1')
        if not 0 < self.yolo_max_area <= 1 or not 0 < self.yolo_max_height <= 1:
            raise ValueError('invalid small-box limits')
        if not 0 <= self.min_semantic_score <= 1 or self.min_replace_gain < .10 or not 0 < self.protected_iou <= 1:
            raise ValueError('invalid replacement/semantic thresholds')
        if (any(not 0 < value <= 1 for value in (self.max_foreign_ink, self.max_nontext_overlap,
                self.snap_cov_x, self.snap_cov_y, self.snap_center_margin, self.orphan_line_fraction, self.partition_iou))
                or not isinstance(self.partition_candidates, int) or not 1 <= self.partition_candidates <= 24
                or not isinstance(self.partition_beam, int) or not 1 <= self.partition_beam <= 32):
            raise ValueError('invalid OCR attribution/partition limits')

    @property
    def paddle_model(self):
        return 'PP-DocLayout-S' if self.variant == 's' else 'PP-DocLayoutV3'

    @property
    def semantic_enabled(self):
        return self.enable_semantic_arbitration or self.enable_semantic_quality


@dataclass
class Proposal:
    id: str
    bbox: list
    label: str
    source: str
    raw_label: str
    score: float | None = None
    line_ids: list = field(default_factory=list)
    source_ids: list = field(default_factory=list)
    stage: str = 'detector'
    reasons: list = field(default_factory=list)
    needs_review: bool = False
    reading_order_rank: int | None = None
    polygon: list | None = None

    def __post_init__(self):
        if (len(self.bbox) != 4 or not all(math.isfinite(v) for v in self.bbox)
                or min(self.bbox) < 0 or self.bbox[2] <= self.bbox[0] or self.bbox[3] <= self.bbox[1]):
            raise ValueError('invalid page bbox')
        if self.score is not None and (not math.isfinite(self.score) or not 0 <= self.score <= 1):
            raise ValueError('invalid measured score')
        if not self.source_ids:
            self.source_ids = [self.id]


def serialize_proposal(proposal: Proposal) -> dict:
    """Retain measured features while preserving featureless legacy payloads."""
    payload = asdict(proposal)
    for key in ('reading_order_rank', 'polygon'):
        if payload[key] is None:
            del payload[key]
    return payload


@dataclass
class Line:
    id: str
    polygon: list
    text: str = ''
    score: float | None = None
    rec_score: float | None = None

    def __post_init__(self):
        for value in (self.score, self.rec_score):
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError('invalid OCR score')


@dataclass
class Result:
    document_id: str
    page_index: int
    width: int
    height: int
    proposals: list
    lines: list
    stages: dict
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    timing: dict = field(default_factory=dict)
    mode: str = 'REAL'
    schema_version: str = 'cascaded-layout-v2.1'
    final_detections: list = field(default_factory=list)
    review_evidence: list = field(default_factory=list)
    decisions: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


# Deliberately separate from the historical mapping which collapses headings.
ALIASES = {
    'display_formula': 'formula', 'inline_formula': 'formula', 'equation': 'formula',
    'figure': 'image', 'picture': 'image', 'figure_caption': 'caption',
    'figure_title': 'caption', 'table_caption': 'caption', 'image_caption': 'caption',
    'table_footnote': 'footnote', 'page_footnote': 'footnote',
    'plain text': 'text', 'abandon': 'unknown', 'isolate_formula': 'formula',
    'formula_caption': 'formula_number',
    'table_title': 'caption', 'chart_title': 'caption',
    'abstract': 'text', 'content': 'text', 'reference_content': 'text',
    'section_header': 'paragraph_title', 'list': 'list_item', 'list-item': 'list_item',
}
CATEGORIES = {'caption': 1, 'footnote': 2, 'formula': 3, 'list_item': 4,
              'footer': 5, 'header': 6, 'image': 7, 'chart': 7,
              'paragraph_title': 8, 'table': 9, 'text': 10, 'doc_title': 11}


def label_for(raw, source):
    # YOLO's generic title does not reliably distinguish document/section title.
    # Keep it unmapped in literal evaluation; merged-heading evaluation is explicit.
    return ALIASES.get(raw, raw)


def category(label, merged=False):
    if merged and label in ('doc_title', 'paragraph_title', 'title'):
        return 8
    return CATEGORIES.get(label, 0)
