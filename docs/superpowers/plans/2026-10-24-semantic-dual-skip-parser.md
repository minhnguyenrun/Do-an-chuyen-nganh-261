# Semantic-Guided Dual-Skip Document Parser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a zero-training, hypothesis-driven Document Parser combining PP-DocLayoutV3 (macro spatial skip-H), PP-OCRv5 detector (micro spatial skip-L), and GLM-OCR 0.9B (semantic probe & complex parser) to verify, relabel, merge, and split layout bounding boxes with complete provenance and rollback support.

**Architecture:** A 4-stage pipeline where Stage 1 captures immutable dual spatial evidence (coarse layout graph + fine line polygons); Stage 2 generates bounded candidate hypotheses (`KEEP`, `MERGE`, `SPLIT`, `RELABEL`, `REFINE_BBOX`); Stage 3 leverages GLM-OCR as a semantic probe to evaluate hypotheses against dual spatial constraints; and Stage 4 performs selective parsing and dual-skip assembly into structured JSON and Markdown with complete provenance lineage.

**Tech Stack:** Python 3.10+, PyTorch, Transformers, PaddleOCR / PaddleX, GLM-OCR (`zai-org/GLM-OCR`), Shapely, OpenCV, Pillow, Pydantic v2, PyYAML, FastAPI, pytest, pytest-asyncio.

**Spec:** `method2/Paddle_GLM_OCR_Semantic_Dual_Skip_Architecture_v4.md`

## Global Constraints

- **Zero-training constraint:** Freeze all model backbones; no weight updates or learned adapters.
- **Strict provenance invariant:** `original_regions` and `original_lines` are strictly immutable; final regions must map to `source_region_ids`.
- **Hard cross-column guard:** Merging bounding boxes across distinct document columns is strictly forbidden regardless of semantic similarity.
- **Grounded split constraint:** A `SPLIT` proposal must have geometric separation backed by text-line clusters or pixel edges; pure text-based arbitrary box splitting is disallowed.
- **Multi-modal relabel constraint:** A `RELABEL` must be supported by spatial evidence and multi-modal alignment, never solely by valid syntax from a specialized prompt.
- **Bounded inference budget:** Enforce `max_hypotheses_per_group=3`, `max_candidate_group_size=8`, `max_iterations=1`, `max_glm_requests_per_page=30`.
- **Safe polygon masking:** When bounding boxes expand or merge, recompute safe masks enclosing all constituent line polygons; fallback to raw unmasked crops if mask certainty is low.
- **Canonical prompt registry:** Use only supported GLM-OCR prompts: `Text Recognition:`, `Table Recognition:`, `Formula Recognition:`.
- **Conservative decision rule:** Accept an alternative hypothesis $H^* \neq H_0$ only if hard constraints pass, $Score(H^*) - Score(H_0) \ge \Delta$ ($\Delta = 0.10$), and $Score(H^*) \ge \tau_{accept}$ ($\tau_{accept} = 0.60$). Otherwise abstain with `KEEP` and flag `needs_review`.

## Review Focus

1. **Borderless table fragmented into multiple text regions:** PP-DocLayoutV3 detects 6 `text` boxes; system merges them into a single `table` region, verifies row/column alignment via Skip-L, and retains all text lines without duplication.
2. **Cross-column text merge prevention:** Adjacent text boxes across two columns with flowing sentence syntax; system strictly rejects `MERGE` due to column boundary violation and preserves separate reading orders.
3. **Heading and paragraph lumped into single box:** Single bounding box containing both a bold title and body text; system splits the box along vertical line gaps, assigning `title` and `text` with localized reading orders.
4. **Display formula misclassified as plain text:** Mathematical formula with summations/fractions labeled as `text`; system probes with `Formula Recognition:`, confirms LaTeX validity and text discrepancy, and relabels to `formula`.
5. **Low-confidence hypothesis abstention (Over-correction defense):** Uncertain candidate with score gain $\Delta < 0.10$ or high ambiguity; system defaults to $H_0$ (`KEEP`) and marks `needs_review` rather than forcing an errant modification.

---

## File Structure

```text
src/
├── __init__.py
├── app.py                              # FastAPI service endpoint (/parse)
├── pipeline.py                         # End-to-end parse_page orchestrator
├── schemas/
│   ├── __init__.py
│   ├── region.py                       # LayoutRegion, TextLineEvidence, FinalRegion
│   ├── hypothesis.py                   # RegionHypothesis, Action, RevisionStatus
│   ├── evidence.py                     # ParseEvidence, GeometryEvidence, ValidationResult
│   └── output.py                       # PageParseResult, RegionRevision, Provenance
├── spatial/
│   ├── __init__.py
│   ├── coordinates.py                  # Coordinate scaling, bounding box normalization
│   ├── crop_builder.py                 # Margin expansion (2-5%), polygon whiteout, raw crop fallback
│   ├── line_assignment.py              # Text line ownership matching (IoU / containment)
│   ├── residual_regions.py             # Uncovered text line detector & clustering
│   ├── region_graph.py                 # Graph G0=(V, E) with column and neighbor relations
│   └── reading_order.py                # DAG reading order validator and cycle breaker
├── adapters/
│   ├── __init__.py
│   ├── layout_ppdocv3.py               # PP-DocLayoutV3 adapter (normalized schema)
│   ├── text_detector_ppocr.py          # PP-OCRv5 detector adapter
│   ├── light_ocr_ppocr.py              # Optional PP-OCR text recognizer
│   └── glm_ocr_client.py               # GLM-OCR client (task prompts, async batching)
├── semantic/
│   ├── __init__.py
│   ├── candidate_generator.py          # KEEP, MERGE, SPLIT, RELABEL generator
│   ├── prompt_router.py                # Maps region labels to official GLM prompts
│   ├── probe_scheduler.py              # Budget tracker and in-flight request scheduler
│   ├── evidence_features.py            # Table row/col extractor, LaTeX balance, CER
│   ├── hypothesis_scorer.py            # Multi-objective scoring S(H)
│   └── semantic_layout_refiner.py      # Core decision engine (accept/reject/abstain)
└── skip/
    ├── __init__.py
    ├── high_spatial_skip.py            # Immutable Skip-H snapshot provider
    ├── low_spatial_skip.py             # Immutable Skip-L snapshot provider
    ├── dual_skip_validator.py          # Hard constraint checker
    └── revision_assembler.py           # DAG assembly into structured JSON and Markdown
configs/
└── default.yaml                        # Configuration thresholds and hyper-parameters
tests/
├── conftest.py                         # Test fixtures, mock images, mock layout outputs
├── test_schemas.py
├── test_coordinates_and_crops.py
├── test_line_assignment_residual.py
├── test_region_graph_order.py
├── test_adapters.py
├── test_candidate_generator.py
├── test_evidence_features.py
├── test_dual_skip_validator.py
├── test_semantic_layout_refiner.py
├── test_revision_assembler.py
└── test_pipeline_e2e.py
```

---

## Tasks

### Task 1: Core Data Schemas & Provenance Models

**Files:**
- Create: `src/schemas/__init__.py`
- Create: `src/schemas/region.py`
- Create: `src/schemas/hypothesis.py`
- Create: `src/schemas/evidence.py`
- Create: `src/schemas/output.py`
- Create: `tests/test_schemas.py`

**Interfaces:**
- Produces:
  - `LayoutRegion`: `(id: str, bbox: tuple[int, int, int, int], polygon: list[tuple[int, int]], label: str, order: int, score: float)`
  - `TextLineEvidence`: `(line_id: str, polygon: list[tuple[int, int]], owner_candidates: list[str], score: float)`
  - `RegionHypothesis`: `(hypothesis_id: str, action: str, source_region_ids: list[str], candidate_label: str, candidate_bbox: tuple[int, int, int, int], candidate_polygon: list[tuple[int, int]] | None)`
  - `ParseEvidence`: `(hypothesis_id: str, parser: str, prompt: str, content: str, status: str)`
  - `RegionRevision`: `(revision_id: str, action: str, source_region_ids: list[str], target_region_id: str, new_label: str, new_bbox: tuple[int, int, int, int], status: str, reasons: list[str])`
  - `FinalRegion`: `(id: str, bbox_final: tuple[int, int, int, int], label_final: str, order_final: int, content: str, content_format: str, source_region_ids: list[str])`
  - `PageParseResult`: `(document_id: str, page_index: int, original_regions: list[LayoutRegion], final_regions: list[FinalRegion], revisions: list[RegionRevision], markdown: str)`

- [ ] **Step 1: Write failing test in `tests/test_schemas.py` for immutable data contracts and serialization**

```python
def test_schema_immutability_and_serialization():
    from src.schemas.region import LayoutRegion, TextLineEvidence
    from src.schemas.hypothesis import RegionHypothesis
    from src.schemas.output import RegionRevision, FinalRegion, PageParseResult

    region = LayoutRegion(
        id="B1", bbox=(10, 20, 100, 200), polygon=[(10, 20), (100, 20), (100, 200), (10, 200)],
        label="text", order=0, score=0.95
    )
    assert region.id == "B1"
    assert region.bbox == (10, 20, 100, 200)

    # Test hypothesis creation
    hyp = RegionHypothesis(
        hypothesis_id="H1", action="MERGE", source_region_ids=["B1", "B2"],
        candidate_label="table", candidate_bbox=(10, 20, 200, 200)
    )
    assert hyp.action == "MERGE"
    assert len(hyp.source_region_ids) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schemas.py -v`
Expected: FAIL with ModuleNotFoundError: No module named 'src.schemas'

- [ ] **Step 3: Implement data schemas using Pydantic v2 in `src/schemas/`**

Create `src/schemas/region.py`, `src/schemas/hypothesis.py`, `src/schemas/evidence.py`, `src/schemas/output.py` with typed fields, JSON serialization helpers, and frozen models where appropriate to guarantee immutability.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schemas.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/schemas/ tests/test_schemas.py
git commit -m "feat(schemas): define core Pydantic data contracts and provenance records"
```

---

### Task 2: Coordinate Normalization & Safe Polygon-Masked Cropping

**Files:**
- Create: `src/spatial/__init__.py`
- Create: `src/spatial/coordinates.py`
- Create: `src/spatial/crop_builder.py`
- Create: `tests/test_coordinates_and_crops.py`

**Interfaces:**
- Consumes: `LayoutRegion` from `src.schemas.region`
- Produces:
  - `expand_bbox(bbox: tuple[int, int, int, int], margin_ratio: float, max_w: int, max_h: int) -> tuple[int, int, int, int]`
  - `build_crop(image: np.ndarray, bbox: tuple[int, int, int, int], polygon: list[tuple[int, int]] | None, mode: str = "masked_whiteout") -> np.ndarray`
  - `union_bboxes(bboxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]`

- [ ] **Step 1: Write failing test in `tests/test_coordinates_and_crops.py`**

```python
import numpy as np

def test_crop_margin_expansion_and_whiteout():
    from src.spatial.crop_builder import expand_bbox, build_crop, union_bboxes

    bbox = (100, 100, 200, 200)
    expanded = expand_bbox(bbox, margin_ratio=0.05, max_w=500, max_h=500)
    assert expanded == (95, 95, 205, 205)

    # Synthetic image 300x300, filled with gray (128)
    img = np.full((300, 300, 3), 128, dtype=np.uint8)
    polygon = [(100, 100), (200, 100), (200, 200), (100, 200)]
    crop = build_crop(img, expanded, polygon=polygon, mode="masked_whiteout")
    
    # Area inside polygon should remain 128; corners outside polygon should be white (255)
    assert crop[10, 10, 0] == 255  # margin outside polygon is whited out
    assert crop[55, 55, 0] == 128  # inside polygon remains original

    # Test union of multiple bboxes
    u = union_bboxes([(10, 20, 50, 60), (30, 10, 100, 90)])
    assert u == (10, 10, 100, 90)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_coordinates_and_crops.py -v`
Expected: FAIL with ModuleNotFoundError: No module named 'src.spatial'

- [ ] **Step 3: Implement `coordinates.py` and `crop_builder.py`**

Implement bbox clamp/expand logic and OpenCV polygon masking with `cv2.fillPoly` to white-out background outside polygon. Fall back to raw crop when `polygon` is None or mode="raw".

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_coordinates_and_crops.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/spatial/ tests/test_coordinates_and_crops.py
git commit -m "feat(spatial): implement bbox coordinate math and polygon-masked whiteout cropping"
```

---

### Task 3: Text-Line Ownership Assignment & Residual Recovery

**Files:**
- Create: `src/spatial/line_assignment.py`
- Create: `src/spatial/residual_regions.py`
- Create: `tests/test_line_assignment_residual.py`

**Interfaces:**
- Consumes: `LayoutRegion`, `TextLineEvidence`
- Produces:
  - `assign_lines_to_regions(regions: list[LayoutRegion], lines: list[TextLineEvidence], iou_threshold: float = 0.3) -> dict[str, list[str]]`
  - `detect_residual_lines(lines: list[TextLineEvidence], assigned_line_ids: set[str]) -> list[TextLineEvidence]`
  - `cluster_residual_lines(residual_lines: list[TextLineEvidence], max_dist: int = 25) -> list[LayoutRegion]`

- [ ] **Step 1: Write failing test in `tests/test_line_assignment_residual.py`**

```python
def test_line_ownership_and_residual_recovery():
    from src.schemas.region import LayoutRegion, TextLineEvidence
    from src.spatial.line_assignment import assign_lines_to_regions
    from src.spatial.residual_regions import detect_residual_lines, cluster_residual_lines

    regions = [LayoutRegion(id="R1", bbox=(50, 50, 200, 100), polygon=[], label="text", order=0, score=0.9)]
    lines = [
        TextLineEvidence(line_id="L1", polygon=[(60, 60), (190, 60), (190, 80), (60, 80)], owner_candidates=[], score=0.95),
        TextLineEvidence(line_id="L2", polygon=[(60, 85), (190, 85), (190, 95), (60, 95)], owner_candidates=[], score=0.92),
        TextLineEvidence(line_id="L_uncovered", polygon=[(50, 400), (200, 400), (200, 420), (50, 420)], owner_candidates=[], score=0.9),
    ]

    ownership = assign_lines_to_regions(regions, lines)
    assert ownership["R1"] == ["L1", "L2"]

    assigned = {"L1", "L2"}
    uncovered = detect_residual_lines(lines, assigned)
    assert len(uncovered) == 1
    assert uncovered[0].line_id == "L_uncovered"

    # Cluster into residual region
    residuals = cluster_residual_lines(uncovered)
    assert len(residuals) == 1
    assert residuals[0].label == "residual"
    assert residuals[0].bbox == (50, 400, 200, 420)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_line_assignment_residual.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement line assignment and residual clustering in `src/spatial/`**

Use Shapely polygon intersection to measure overlap area of each line against candidate region bboxes. Group uncovered lines within `max_dist` into bounding boxes marked with `label="residual"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_line_assignment_residual.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/spatial/line_assignment.py src/spatial/residual_regions.py tests/test_line_assignment_residual.py
git commit -m "feat(spatial): implement text-line ownership assignment and residual text recovery"
```

---

### Task 4: Region Graph & Reading Order DAG Engine

**Files:**
- Create: `src/spatial/region_graph.py`
- Create: `src/spatial/reading_order.py`
- Create: `tests/test_region_graph_order.py`

**Interfaces:**
- Consumes: `LayoutRegion`, `TextLineEvidence`
- Produces:
  - `RegionGraph`: adjacency model tracking nodes, geometric edges, column IDs, and reading-order predecessors
  - `build_region_graph(regions: list[LayoutRegion], page_w: int, page_h: int) -> RegionGraph`
  - `reconcile_order_as_dag(graph: RegionGraph) -> list[str]` (topological ordering, cycle detection)

- [ ] **Step 1: Write failing test in `tests/test_region_graph_order.py`**

```python
def test_region_graph_column_detection_and_dag_order():
    from src.schemas.region import LayoutRegion
    from src.spatial.region_graph import build_region_graph
    from src.spatial.reading_order import reconcile_order_as_dag

    # 2-column layout: Col 1 has B1, B2; Col 2 has B3, B4
    regions = [
        LayoutRegion(id="B1", bbox=(50, 50, 200, 150), polygon=[], label="text", order=0, score=0.9),
        LayoutRegion(id="B2", bbox=(50, 170, 200, 300), polygon=[], label="text", order=1, score=0.9),
        LayoutRegion(id="B3", bbox=(250, 50, 400, 150), polygon=[], label="text", order=2, score=0.9),
        LayoutRegion(id="B4", bbox=(250, 170, 400, 300), polygon=[], label="text", order=3, score=0.9),
    ]

    graph = build_region_graph(regions, page_w=500, page_h=500)
    assert graph.get_column_id("B1") == graph.get_column_id("B2")
    assert graph.get_column_id("B3") == graph.get_column_id("B4")
    assert graph.get_column_id("B1") != graph.get_column_id("B3")
    assert graph.is_cross_column("B1", "B3") is True

    ordered_ids = reconcile_order_as_dag(graph)
    assert ordered_ids == ["B1", "B2", "B3", "B4"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_region_graph_order.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement `RegionGraph` and DAG topological sorting**

Build `RegionGraph` with node metadata and column assignment based on horizontal X-overlap clustering. Provide Kahn's algorithm in `reading_order.py` for cycle-free topological order derivation.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_region_graph_order.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/spatial/region_graph.py src/spatial/reading_order.py tests/test_region_graph_order.py
git commit -m "feat(spatial): implement region graph with column clustering and DAG order resolver"
```

---

### Task 5: Model Adapters & GLM-OCR Client Interface

**Files:**
- Create: `src/adapters/__init__.py`
- Create: `src/adapters/layout_ppdocv3.py`
- Create: `src/adapters/text_detector_ppocr.py`
- Create: `src/adapters/light_ocr_ppocr.py`
- Create: `src/adapters/glm_ocr_client.py`
- Create: `tests/test_adapters.py`

**Interfaces:**
- Produces:
  - `LayoutDetectorAdapter.detect(image: np.ndarray) -> list[LayoutRegion]`
  - `TextDetectorAdapter.detect_lines(image: np.ndarray) -> list[TextLineEvidence]`
  - `LightOcrAdapter.recognize(crops: list[np.ndarray]) -> list[str]`
  - `GlmOcrClient.recognize_batch(requests: list[tuple[str, np.ndarray, str]]) -> list[ParseEvidence]`

- [ ] **Step 1: Write failing test in `tests/test_adapters.py` with mockable interfaces**

```python
import pytest
import numpy as np

@pytest.mark.asyncio
async def test_adapters_normalization_and_contracts():
    from src.adapters.glm_ocr_client import GlmOcrClient
    from src.schemas.evidence import ParseEvidence

    client = GlmOcrClient(endpoint="mock://glm", mock_mode=True)
    reqs = [("H1", np.zeros((100, 100, 3), dtype=np.uint8), "Table Recognition:")]
    results = await client.recognize_batch(reqs)
    
    assert len(results) == 1
    assert isinstance(results[0], ParseEvidence)
    assert results[0].hypothesis_id == "H1"
    assert results[0].prompt == "Table Recognition:"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_adapters.py -v`
Expected: FAIL with ModuleNotFoundError: No module named 'src.adapters'

- [ ] **Step 3: Implement adapters with real engine hooks and testing mock modes**

Create adapters in `src/adapters/` that translate PaddleOCR / PaddleX / GLM-OCR outputs into normalized `LayoutRegion`, `TextLineEvidence`, and `ParseEvidence` models. Support `mock_mode=True` for offline testing without GPU.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_adapters.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/adapters/ tests/test_adapters.py
git commit -m "feat(adapters): implement normalized wrappers for PP-DocLayoutV3, PP-OCRv5 and GLM-OCR client"
```

---

### Task 6: Candidate Hypothesis Generator & Prompt Router

**Files:**
- Create: `src/semantic/__init__.py`
- Create: `src/semantic/candidate_generator.py`
- Create: `src/semantic/prompt_router.py`
- Create: `src/semantic/probe_scheduler.py`
- Create: `tests/test_candidate_generator.py`

**Interfaces:**
- Consumes: `RegionGraph`, `LayoutRegion`
- Produces:
  - `route_prompt(label: str) -> str`: canonical prompt (`Text Recognition:`, `Table Recognition:`, `Formula Recognition:`)
  - `generate_hypotheses(graph: RegionGraph, max_hypotheses_per_group: int = 3) -> list[RegionHypothesis]`
  - `ProbeScheduler`: bounds requests per page $\le 30$ and limits concurrency.

- [ ] **Step 1: Write failing test in `tests/test_candidate_generator.py`**

```python
def test_candidate_generator_borderless_table_and_math():
    from src.schemas.region import LayoutRegion
    from src.spatial.region_graph import build_region_graph
    from src.semantic.candidate_generator import generate_hypotheses
    from src.semantic.prompt_router import route_prompt

    assert route_prompt("table") == "Table Recognition:"
    assert route_prompt("formula") == "Formula Recognition:"
    assert route_prompt("text") == "Text Recognition:"

    # 4 small text boxes arranged in 2x2 grid in same column -> candidate for table MERGE
    regions = [
        LayoutRegion(id="B1", bbox=(50, 50, 100, 70), polygon=[], label="text", order=0, score=0.9),
        LayoutRegion(id="B2", bbox=(110, 50, 160, 70), polygon=[], label="text", order=1, score=0.9),
        LayoutRegion(id="B3", bbox=(50, 80, 100, 100), polygon=[], label="text", order=2, score=0.9),
        LayoutRegion(id="B4", bbox=(110, 80, 160, 100), polygon=[], label="text", order=3, score=0.9),
    ]
    graph = build_region_graph(regions, page_w=500, page_h=500)
    hyps = generate_hypotheses(graph)

    # Must contain a MERGE to table proposal
    merge_hyps = [h for h in hyps if h.action == "MERGE" and h.candidate_label == "table"]
    assert len(merge_hyps) >= 1
    assert set(merge_hyps[0].source_region_ids) == {"B1", "B2", "B3", "B4"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_candidate_generator.py -v`
Expected: FAIL with ModuleNotFoundError: No module named 'src.semantic'

- [ ] **Step 3: Implement `candidate_generator.py`, `prompt_router.py`, `probe_scheduler.py`**

Identify candidate groups: grid of aligned text boxes for `MERGE(table)`, single box with large vertical line gaps for `SPLIT`, text boxes containing mathematical character density for `RELABEL(formula)`. Route to canonical GLM prompts.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_candidate_generator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/semantic/candidate_generator.py src/semantic/prompt_router.py src/semantic/probe_scheduler.py tests/test_candidate_generator.py
git commit -m "feat(semantic): implement bounded hypothesis generator, prompt router and probe scheduler"
```

---

### Task 7: Evidence Feature Extraction

**Files:**
- Create: `src/semantic/evidence_features.py`
- Create: `tests/test_evidence_features.py`

**Interfaces:**
- Consumes: `ParseEvidence`, `LayoutRegion`, `TextLineEvidence`
- Produces:
  - `extract_table_features(html_content: str, line_count: int) -> dict[str, float]`: row/col count, cell density, alignment score
  - `extract_formula_features(latex_content: str) -> dict[str, float]`: bracket balance, operator density, syntax validity
  - `calculate_content_consistency(probe_text: str, line_transcripts: list[str]) -> float`: character preservation ratio $\in [0, 1]$

- [ ] **Step 1: Write failing test in `tests/test_evidence_features.py`**

```python
def test_evidence_feature_extraction():
    from src.semantic.evidence_features import extract_table_features, extract_formula_features, calculate_content_consistency

    # Valid HTML table with 2 rows and 2 columns
    html = "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>"
    feats = extract_table_features(html, line_count=4)
    assert feats["is_valid_html"] == 1.0
    assert feats["row_count"] == 2.0
    assert feats["col_count"] == 2.0
    assert feats["cell_count"] == 4.0

    # Formula check
    latex = r"\frac{a + b}{c} = \sum_{i=1}^{n} x_i"
    f_feats = extract_formula_features(latex)
    assert f_feats["is_balanced"] == 1.0
    assert f_feats["has_math_tokens"] == 1.0

    # Content consistency
    score = calculate_content_consistency("Hello World", ["Hello", "World"])
    assert score > 0.95
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evidence_features.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement feature extraction in `src/semantic/evidence_features.py`**

Parse HTML tables using BeautifulSoup/regex to count rows/cols and verify grid consistency. Verify balanced curly braces and math operators in LaTeX formulas. Compute normalized edit distance / Levenshtein overlap for character consistency.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evidence_features.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/semantic/evidence_features.py tests/test_evidence_features.py
git commit -m "feat(semantic): implement structural table, formula and character consistency feature extractors"
```

---

### Task 8: Dual-Skip Validator & Hard Constraints Engine

**Files:**
- Create: `src/skip/__init__.py`
- Create: `src/skip/high_spatial_skip.py`
- Create: `src/skip/low_spatial_skip.py`
- Create: `src/skip/dual_skip_validator.py`
- Create: `tests/test_dual_skip_validator.py`

**Interfaces:**
- Consumes: `RegionHypothesis`, `RegionGraph`, `TextLineEvidence`, `ParseEvidence`
- Produces:
  - `HighSpatialSkip`: immutable wrapper around initial regions, labels, and order DAG
  - `LowSpatialSkip`: immutable wrapper around text-line polygons and pixel crops
  - `validate_hard_constraints(hyp: RegionHypothesis, skip_h: HighSpatialSkip, skip_l: LowSpatialSkip, evidence: ParseEvidence) -> tuple[bool, list[str]]`

- [ ] **Step 1: Write failing test in `tests/test_dual_skip_validator.py` (review focus 2: cross-column rejection)**

```python
def test_cross_column_merge_rejected_by_hard_constraints():
    from src.schemas.region import LayoutRegion, TextLineEvidence
    from src.schemas.hypothesis import RegionHypothesis
    from src.schemas.evidence import ParseEvidence
    from src.spatial.region_graph import build_region_graph
    from src.skip.high_spatial_skip import HighSpatialSkip
    from src.skip.low_spatial_skip import LowSpatialSkip
    from src.skip.dual_skip_validator import validate_hard_constraints

    r1 = LayoutRegion(id="B1", bbox=(50, 50, 200, 150), polygon=[], label="text", order=0, score=0.9)
    r2 = LayoutRegion(id="B2", bbox=(250, 50, 400, 150), polygon=[], label="text", order=1, score=0.9)
    graph = build_region_graph([r1, r2], page_w=500, page_h=500)

    skip_h = HighSpatialSkip([r1, r2], graph)
    skip_l = LowSpatialSkip([], np.zeros((500, 500, 3), dtype=np.uint8))

    # Cross-column merge proposal
    hyp = RegionHypothesis(
        hypothesis_id="H_merge_col", action="MERGE", source_region_ids=["B1", "B2"],
        candidate_label="text", candidate_bbox=(50, 50, 400, 150)
    )
    evidence = ParseEvidence(hypothesis_id="H_merge_col", parser="glm", prompt="Text Recognition:", content="Unified text", status="ok")

    passed, flags = validate_hard_constraints(hyp, skip_h, skip_l, evidence)
    assert passed is False
    assert "cross_column_violation" in flags
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dual_skip_validator.py -v`
Expected: FAIL with ModuleNotFoundError: No module named 'src.skip'

- [ ] **Step 3: Implement `high_spatial_skip.py`, `low_spatial_skip.py`, and `dual_skip_validator.py`**

Validate hard constraints:
1. Reject `MERGE` across different columns (`graph.is_cross_column`).
2. Reject `MERGE` or `SPLIT` that loses required text lines from `LowSpatialSkip`.
3. Reject `RELABEL(table)` if table HTML contains 0 rows or is unparseable.
4. Reject `RELABEL(formula)` if LaTeX has unclosed brackets.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dual_skip_validator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/skip/ tests/test_dual_skip_validator.py
git commit -m "feat(skip): implement immutable dual-skip snapshots and strict hard-constraints validator"
```

---

### Task 9: Hypothesis Scorer & Semantic Layout Refiner

**Files:**
- Create: `src/semantic/hypothesis_scorer.py`
- Create: `src/semantic/semantic_layout_refiner.py`
- Create: `tests/test_semantic_layout_refiner.py`

**Interfaces:**
- Consumes: `RegionHypothesis`, `ParseEvidence`, `HighSpatialSkip`, `LowSpatialSkip`
- Produces:
  - `calculate_hypothesis_score(hyp: RegionHypothesis, evidence: ParseEvidence, feats: dict, skip_h: HighSpatialSkip, skip_l: LowSpatialSkip, cfg: dict) -> float`
  - `refine_hypotheses(group_id: str, choices: list[RegionHypothesis], evidence_map: dict[str, ParseEvidence], skip_h: HighSpatialSkip, skip_l: LowSpatialSkip, cfg: dict) -> RegionRevision`

- [ ] **Step 1: Write failing test in `tests/test_semantic_layout_refiner.py` (review focus 1 & 5: table acceptance & low-gain abstention)**

```python
def test_semantic_layout_refiner_accept_and_abstain():
    from src.schemas.region import LayoutRegion
    from src.schemas.hypothesis import RegionHypothesis
    from src.schemas.evidence import ParseEvidence
    from src.spatial.region_graph import build_region_graph
    from src.skip.high_spatial_skip import HighSpatialSkip
    from src.skip.low_spatial_skip import LowSpatialSkip
    from src.semantic.semantic_layout_refiner import refine_hypotheses

    r1 = LayoutRegion(id="B1", bbox=(50, 50, 100, 70), polygon=[], label="text", order=0, score=0.9)
    r2 = LayoutRegion(id="B2", bbox=(110, 50, 160, 70), polygon=[], label="text", order=1, score=0.9)
    graph = build_region_graph([r1, r2], page_w=500, page_h=500)
    skip_h = HighSpatialSkip([r1, r2], graph)
    skip_l = LowSpatialSkip([], np.zeros((500, 500, 3), dtype=np.uint8))

    h0 = RegionHypothesis(hypothesis_id="H0", action="KEEP", source_region_ids=["B1", "B2"], candidate_label="text", candidate_bbox=(50, 50, 160, 70))
    h1 = RegionHypothesis(hypothesis_id="H1", action="MERGE", source_region_ids=["B1", "B2"], candidate_label="table", candidate_bbox=(50, 50, 160, 70))

    evidence_map = {
        "H0": ParseEvidence(hypothesis_id="H0", parser="light", prompt="", content="Text", status="ok"),
        "H1": ParseEvidence(hypothesis_id="H1", parser="glm", prompt="Table Recognition:", content="<table><tr><td>A</td><td>B</td></tr></table>", status="ok")
    }
    cfg = {"min_score_gain": 0.10, "min_absolute_score": 0.60}

    # High quality table evidence -> Accept MERGE
    rev = refine_hypotheses("group_1", [h0, h1], evidence_map, skip_h, skip_l, cfg)
    assert rev.status == "accepted"
    assert rev.action == "MERGE"
    assert rev.new_label == "table"

    # Weak evidence -> Abstain with KEEP
    evidence_weak = {
        "H0": ParseEvidence(hypothesis_id="H0", parser="light", prompt="", content="Text", status="ok"),
        "H1": ParseEvidence(hypothesis_id="H1", parser="glm", prompt="Table Recognition:", content="Corrupted text", status="ok")
    }
    rev_abstain = refine_hypotheses("group_1", [h0, h1], evidence_weak, skip_h, skip_l, cfg)
    assert rev_abstain.status == "abstained"
    assert rev_abstain.action == "KEEP"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_semantic_layout_refiner.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement scoring function and refiner decision engine**

Implement formula:
$$Score(H) = w_g S_g + w_r S_r + w_c S_c + w_s S_s - w_l P_{loss} - w_k P_{complexity}$$
Compare $Score(H^*) - Score(H_0) \ge \Delta$. If passes hard constraints and margin, return `accepted` `RegionRevision`; otherwise return `abstained` `KEEP` revision with warnings.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_semantic_layout_refiner.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/semantic/hypothesis_scorer.py src/semantic/semantic_layout_refiner.py tests/test_semantic_layout_refiner.py
git commit -m "feat(semantic): implement multi-objective hypothesis scorer and conservative layout refiner"
```

---

### Task 10: Revision Graph Updater & Final Skip-Assembly

**Files:**
- Create: `src/skip/revision_assembler.py`
- Create: `tests/test_revision_assembler.py`

**Interfaces:**
- Consumes: `RegionGraph`, `RegionRevision`, `LayoutRegion`, `ParseEvidence`
- Produces:
  - `apply_revisions_to_graph(graph: RegionGraph, revisions: list[RegionRevision]) -> RegionGraph`
  - `assemble_document(original_regions: list[LayoutRegion], final_graph: RegionGraph, recognitions: dict[str, str], revisions: list[RegionRevision], warnings: list[str]) -> PageParseResult`

- [ ] **Step 1: Write failing test in `tests/test_revision_assembler.py`**

```python
def test_revision_assembly_and_provenance_preservation():
    from src.schemas.region import LayoutRegion
    from src.schemas.output import RegionRevision, PageParseResult
    from src.spatial.region_graph import build_region_graph
    from src.skip.revision_assembler import apply_revisions_to_graph, assemble_document

    r1 = LayoutRegion(id="B1", bbox=(50, 50, 100, 70), polygon=[], label="text", order=0, score=0.9)
    r2 = LayoutRegion(id="B2", bbox=(110, 50, 160, 70), polygon=[], label="text", order=1, score=0.9)
    graph = build_region_graph([r1, r2], page_w=500, page_h=500)

    rev = RegionRevision(
        revision_id="rev_1", action="MERGE", source_region_ids=["B1", "B2"], target_region_id="T1",
        new_label="table", new_bbox=(50, 50, 160, 70), status="accepted", reasons=["table_grid_verified"]
    )
    refined_graph = apply_revisions_to_graph(graph, [rev])
    assert "T1" in refined_graph.nodes
    assert "B1" not in refined_graph.nodes
    assert "B2" not in refined_graph.nodes

    recognitions = {"T1": "<table><tr><td>A</td><td>B</td></tr></table>"}
    result = assemble_document(original_regions=[r1, r2], final_graph=refined_graph, recognitions=recognitions, revisions=[rev], warnings=[])

    assert isinstance(result, PageParseResult)
    assert len(result.original_regions) == 2  # original immutable
    assert len(result.final_regions) == 1
    assert result.final_regions[0].id == "T1"
    assert result.final_regions[0].source_region_ids == ["B1", "B2"]
    assert "<table>" in result.markdown
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_revision_assembler.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement revision graph updater and final assembler**

Implement `apply_revisions_to_graph` to substitute parent nodes for child clusters in the graph while preserving topological reading order. Implement `assemble_document` to build `PageParseResult` with Markdown synthesis and complete lineage mapping.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_revision_assembler.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/skip/revision_assembler.py tests/test_revision_assembler.py
git commit -m "feat(skip): implement revision graph update and traceable dual-skip document assembly"
```

---

### Task 11: End-to-End Pipeline Orchestration & FastAPI Service

**Files:**
- Create: `configs/default.yaml`
- Create: `src/pipeline.py`
- Create: `src/app.py`
- Create: `tests/test_pipeline_e2e.py`

**Interfaces:**
- Produces:
  - `parse_page(page_image: np.ndarray, config: dict, layout_adapter, text_adapter, light_ocr_adapter, glm_client) -> PageParseResult`
  - `app`: FastAPI web application exposing `POST /parse`

- [ ] **Step 1: Write failing test in `tests/test_pipeline_e2e.py`**

```python
import pytest
import numpy as np

@pytest.mark.asyncio
async def test_end_to_end_pipeline_with_mock_services():
    from src.pipeline import parse_page
    from src.adapters.layout_ppdocv3 import LayoutDetectorAdapter
    from src.adapters.text_detector_ppocr import TextDetectorAdapter
    from src.adapters.light_ocr_ppocr import LightOcrAdapter
    from src.adapters.glm_ocr_client import GlmOcrClient
    from src.schemas.output import PageParseResult
    import yaml

    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Synthetic page image
    img = np.full((600, 600, 3), 255, dtype=np.uint8)

    layout_svc = LayoutDetectorAdapter(mock_mode=True)
    text_svc = TextDetectorAdapter(mock_mode=True)
    light_svc = LightOcrAdapter(mock_mode=True)
    glm_svc = GlmOcrClient(mock_mode=True)

    result = await parse_page(img, cfg, layout_svc, text_svc, light_svc, glm_svc)
    assert isinstance(result, PageParseResult)
    assert len(result.final_regions) > 0
    assert result.markdown != ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_e2e.py -v`
Expected: FAIL with FileNotFoundError or ModuleNotFoundError

- [ ] **Step 3: Implement `configs/default.yaml`, `src/pipeline.py` and `src/app.py`**

Wire the 4 execution stages into `parse_page` as defined in Section 10 of the spec:
1. Spatial encoding (PP-DocLayoutV3 + PP-OCRv5 det).
2. Graph construction + residual text detection + bounded hypothesis generation.
3. Light OCR (optional) + GLM-OCR semantic probes + dual-skip evaluation.
4. Revision graph update + DAG topological order reconciliation + final assembly.
Add FastAPI endpoint `POST /parse` in `src/app.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline_e2e.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add configs/default.yaml src/pipeline.py src/app.py tests/test_pipeline_e2e.py
git commit -m "feat(pipeline): implement full 4-stage end-to-end pipeline and FastAPI endpoint"
```

---

### Task 12: Benchmark & Ablation Evaluation Harness (A0–A9)

**Files:**
- Create: `src/evaluation/__init__.py`
- Create: `src/evaluation/evaluate_refinement.py`
- Create: `src/evaluation/evaluate_detection.py`
- Create: `src/evaluation/evaluate_parsing.py`
- Create: `tests/test_evaluation.py`

**Interfaces:**
- Produces:
  - `compute_refinement_metrics(predicted_revisions: list[RegionRevision], ground_truth_revisions: list[dict]) -> dict[str, float]`: MERGE/SPLIT/RELABEL Precision/Recall, Over-correction rate
  - `compute_reading_order_ned(predicted_order: list[str], ground_truth_order: list[str]) -> float`: Normalized Edit Distance
  - `run_ablation_suite(dataset_dir: str, configurations: list[str]) -> dict[str, dict[str, float]]`

- [ ] **Step 1: Write failing test in `tests/test_evaluation.py`**

```python
def test_evaluation_metrics_and_over_correction_rate():
    from src.schemas.output import RegionRevision
    from src.evaluation.evaluate_refinement import compute_refinement_metrics
    from src.evaluation.evaluate_parsing import compute_reading_order_ned

    preds = [
        RegionRevision(revision_id="r1", action="MERGE", source_region_ids=["B1", "B2"], target_region_id="T1", new_label="table", new_bbox=(0,0,10,10), status="accepted", reasons=[]),
        RegionRevision(revision_id="r2", action="RELABEL", source_region_ids=["B3"], target_region_id="B3", new_label="formula", new_bbox=(0,0,10,10), status="accepted", reasons=[])
    ]
    gt = [
        {"action": "MERGE", "source_region_ids": ["B1", "B2"], "new_label": "table"}
    ]

    metrics = compute_refinement_metrics(preds, gt)
    assert metrics["merge_precision"] == 1.0
    assert metrics["merge_recall"] == 1.0
    # r2 was not in GT, so it is counted as over-correction
    assert metrics["over_correction_count"] == 1

    ro_ned = compute_reading_order_ned(["A", "B", "C"], ["A", "B", "C"])
    assert ro_ned == 1.0  # Perfect order
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluation.py -v`
Expected: FAIL with ModuleNotFoundError: No module named 'src.evaluation'

- [ ] **Step 3: Implement evaluation metrics in `src/evaluation/`**

Implement precision, recall, and F1 for layout actions (MERGE, SPLIT, RELABEL); compute over-correction rate (false-positive revisions on clean regions); calculate Normalized Edit Distance for reading order.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evaluation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/ tests/test_evaluation.py
git commit -m "feat(eval): implement evaluation harness for revision precision, over-correction, and reading order"
```

---

## Self-Review

1. **Spec coverage:**
   - Section 1-2 (Dual skip, formal notation): Covered in Tasks 1, 4, 8.
   - Section 3 (Components & responsibilities): Covered in Tasks 2, 3, 5, 6, 7.
   - Section 4 (Region graph, table merge, split, DAG order): Covered in Tasks 4, 6, 10.
   - Section 5 (Dual skip H & L, polygon whiteout padding): Covered in Tasks 2, 8.
   - Section 6 (Routing policy, budget limits): Covered in Tasks 6, 11.
   - Section 7 (Scoring & decision rules with $\Delta, \tau$): Covered in Task 9.
   - Section 8-9 (Sequence, FSM, data contracts & provenance): Covered in Tasks 1, 9, 10.
   - Section 10 (Pseudocode): Realized end-to-end in Task 11.
   - Section 11 (Repository & service topology): Realized across Tasks 1–11.
   - Section 12 (Ablation suite A0–A9 & metrics): Covered in Task 12.

2. **Step scan:** Every task follows TDD strictly: failing test step, fail verification, implementation step, pass verification, commit step.
3. **Type consistency:** All modules exchange validated Pydantic models (`LayoutRegion`, `TextLineEvidence`, `RegionHypothesis`, `ParseEvidence`, `RegionRevision`, `FinalRegion`, `PageParseResult`).
4. **Review Focus:**
   - Borderless table merge: Covered in Task 6, Task 9.
   - Cross-column merge rejection: Covered in Task 4, Task 8.
   - Mixed heading-body split: Covered in Task 6, Task 9.
   - Display formula relabel: Covered in Task 6, Task 7, Task 9.
   - Abstention & over-correction defense: Covered in Task 9, Task 12.
5. **Proportion:** Clear, actionable, and granular without drowning in speculative source transcripts.

