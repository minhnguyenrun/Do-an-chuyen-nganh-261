# Cascaded Semantic V3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and measure DOCX Step3 using real V3 order/polygon evidence and a fitted frozen-Model2Vec+LightGBM CPU arbitrator.

**Architecture:** Optional measured V3 fields drive page-local spatial safety and recovery. A separately persisted learned head resolves bounded textual conflicts and supplies semantic quality evidence without weakening geometry. Separate flags isolate every ablation and preserve legacy defaults.

**Tech Stack:** Existing Python3.12/PaddleOCR3.7.0/PaddleX3.7.2/Shapely; isolated model2vec base+LightGBM, frozen potion8M256D, sklearn calibration utilities; immutable JSON/native classifier artifacts.

**Spec:** docs/superpowers/specs/2026-11-17-cascaded-semantic-v3-design.md

## Global Constraints

- New flags `enable_v3_polygons`, `enable_v3_reading_order`, `enable_semantic_arbitration`, `enable_semantic_quality` default False; both Phase1 flags remain False.
- Proposal adds optional rank/polygon; serialize omits only new None fields; legacy dictionaries/detections/targets must reproduce baseline.
- No staging/commit, no discard of existing untracked work, no overwrite of old cache/data; use fresh signature namespaces and snapshots.
- GPT-6.2 Sol production implementers run sequentially. Independent GPT-6.2 Sol testers own test writing/execution, training, probes and evaluation. Controller coordinates only.
- Proper escalation for ACL/network; isolated workspace dependencies only; no global Transformer change, distillation, huge shard or dataset download.
- At most64 true-train pages/100MB raw row payload, optional64 images/100MB; model fetch pinned full commitSHA; API row content hashes and observed repo SHA are provenance, not invented revision pinning.
- Document-group fit/calibration, fixed valid20 plus 30 separate valid pages for extended evaluation, and no test use. Missing data, class support or dependencies cause explicit unavailable/abstain.
- Preserve all geometry/ownership/completeness guards, original Q weights and replacement margin>=.10.

## Review Focus

- One-based public order may be None/bool/duplicate: Task1 contract tests and Task2 graph ambiguity tests must abstain, never fabricate rank.
- Geometry edits invalidate measured polygon: Task2 stale-mask test requires clear polygon and retained old-line coverage.
- API rows are not revision-pinned: Task3 acquisition manifest checks before/after repo SHA and payload hashes; Task4 rejects changed/mismatched provenance.
- Rare/absent classes and pdf-cell/OCR shift: Task3 support/calibration gates and Task4 class coverage report prohibit universal confidence claims.
- Changed targets/crops invalidate cached MinerU coordinates: Task4 counters and counterfactual labels prohibit invented live speedup/F1 evidence.

---

### Task 1: Preserve public V3 evidence and old JSON contracts

**Files:** Modify method2/src/cascaded/{types,adapters,runner}.py and serialization-only Proposal asdict sites in fusion.py/attribution.py; create v3_features.py; tester creates method2/tests/test_cascaded_v3_contract.py. Full brief: scratch/cascaded_phase2/task1_v3_contract.md.

**Interfaces:** `serialize_proposal(p)->dict`; `normalize_paddle_box(raw,*,width,height,variant,proposal_id)->(Proposal|None,audit)`; Proposal rank/polygon fields and four Config flags plus semantic_artifact_path. Public box mapping/capability output consumed by Task2 and actual-feature probe.

- [ ] Tester writes RED assertions for old positional/JSON compatibility, order2=>rank1, None skipped order, malformed/bool order, polygon pairing/clipping/invalid geometry, S fallback and actual worker normalization path.
- [ ] Tester runs `$PY -m pytest tests/test_cascaded_v3_contract.py -q -p no:cacheprovider` from method2; captures RED failure tied to missing behavior. Include numpy/nonfinite raw audit allow_nan=False serialization.
- [ ] Implementer adds pure normalizer/default fields/serializer, real child adapter integration and capability metadata regardless of usage flags. Update Proposal serialization sites to preserve legacy featureless dictionaries and cache isolation. Fresh real-feature outputs retain evidence; compare baseline canonical geometry/labels/lineage/ownership rather than deleting evidence.
- [ ] Tester reruns focused file; expected all pass and no model downloads/imports. Independent reviewer checks public API fidelity/feature pairing/legacy equality; controller records status.

### Task 2: Use polygons and same-column reading graph conservatively

**Files:** Create method2/src/cascaded/{v3_geometry,reading_order}.py; modify {core,fusion,attribution,geometry_refine}.py; tester creates method2/tests/test_cascaded_v3_spatial.py. Full brief: scratch/cascaded_phase2/task2_v3_spatial.md.

**Interfaces:** `proposal_shape(p,*,use_v3_polygon=False)`, `proposal_line_coverage(line,p,*,use_v3_polygon=False)`, `protected_overlap(candidate_bbox,p,*,use_v3_polygon=False)`; `build_reading_graph(proposals,lines,cfg)->dict`; `graph_neighbor(region,orphan_lines,graph,lines,confirmed)->dict` returning supported,neighbor_id,before_id,after_id,reasons,evidence. Consume Task1 fields; no semantic dependency yet.

- [ ] Tester RED cases: false bbox/nontext collision resolved only by valid measured polygon; true overlap/foreign owner remain blocked; missing/S/invalid features exact fallback; stale mask cleared after WBF/expansion/ABSORB; valid same-column indented first line recovered; cross-column/page-wide bridge/duplicate rank/ambiguous recipient abstains; one-edit and line accounting retained.
- [ ] Tester runs focused spatial file, records RED.
- [ ] Implementer shares shape policy across assignment/protection and combined geometry refinement; adds pure graph evidence to active fusion boundary recovery/orphan grouping and attribution ABSORB. Relax only first-line margin with unique same-column proof. Preserve output boxes, thresholds and frozen ownership.
- [ ] Tester runs spatial+Task1+existing cascaded safety subset; expected all pass. Reviewer verifies no rank-derived cross-column joins and no confidence/shape bypass of ownership guards.

### Task 3: Real frozen embedding, audited corpus, native head and optional resolver

**Files:** Create method2/src/cascaded/{semantic_features,semantic}.py and method2/scripts/{prepare_cascaded_semantic,train_cascaded_semantic}.py; modify {types,runner,fusion,attribution}.py and requirements-cascaded.txt; tester creates method2/tests/test_cascaded_semantic_v3.py. Full brief: scratch/cascaded_phase2/task3_semantic.md.

**Interfaces:** preparation CLI `--output ROOT --max-pages 64 --prepare-only`; training CLI `--corpus ROOT --output ARTIFACT_ROOT --seed 0`; `SemanticArbitrator.from_artifact(path)` and `.predict(label,lines,page_lines,bbox,width,height)->dict`; `semantic_features.region_features(...)` exact artifact schema. Optional `semantic=None` threads prepare_layout, quality and attribution helpers with old calls unchanged.

Preparation may start alongside Task1 using worker-owned scratch acquire_semantic_assets.py, without waiting for production CLI. Stable handoff prepared_assets/{assets_manifest.json,packages.json,embedding/manifest.json,train/manifest.json,train/raw_rows/block_<offset>.json} is defined in Task3 brief. Production CLI accepts `--assets-root`, verifies hashes/provenance and builds corpus without silently redownloading/reselecting rows. Setup worker never fits or tests.

- [ ] Tester writes RED for artifact/schema/hash invalidation, unavailable/no-import legacy path, finite calibrated class probabilities, low OCR/unsupported class abstain, independent flags, conflict joint=.55det+.45P / thresholds.65/.15, weak gate.60, unchanged Q/margin and forbidden geometry bypass.
- [ ] Tester runs focused semantic file, records RED.
- [ ] Implementer writes bounded preparation/fitting CLIs and manifest verification first. Preparation resolves package/model full revisions and actual train row identity; immutable corpus has original_filename/page_hash; no train fitting is executed by implementer. Native head/calibration loader rejects stale schema/hashes. Keep model2vec no distill/train extras.
- [ ] After script review, preparation worker executes dependency install into workspace target/venv and reviewed prepare-only CLI via proper escalation. Capture exact package/model source revisions/hashes and dataset SHA before/after+API revision status. Failure stops dependent learned runs, not other V3 work.
- [ ] Implementer adds featurizer/resolver and explicit threading: arbitration flag for one-to-one textual conflict/recovery; quality flag for attribution/Q. Preserve PP geometry/lineage/score and all hard gates. Structured unavailable falls back to legacy routing. No eager learned imports/model load when flagsFalse.
- [ ] Tester runs focused file and safety subset; reviewer inspects actual Model2Vec API/artifact flow, train provenance and dependency isolation before Task4 fitting.

### Task 4: Independent fitting, actual V3 capability and valid ablations

**Files:** Tester creates scratch/cascaded_phase2/{evaluate_phase2,probe_v3_features}.py, scoped tests as above, RESULTS.md, immutable logs/manifests/metric outputs. Production corrections return to implementer. Full brief: scratch/cascaded_phase2/task4_evaluation.md.

**Interfaces:** consumes Task1–3 functions/CLIs/native artifact; fixed valid IDs0,1,2,3,4,5,6,8,9,10,11,13,14,17,18,20,21,22,23,24; source caches S75ac93d5d258 / V3b4aa156a1a43; shared evaluator used by Phase1 replay.

- [ ] Verify reviewed preparation provenance; tester executes training CLI once with seed0 and document-group split; record actual fit/calibration IDs/documents/class support and reliability. If insufficient calibration/groups, report unsupported/incomplete instead of leaking valid into fit.
- [ ] Run one fixed valid V3 page layout+OCR-only probe, separate cache with raw box mapping/version/hash metadata. If real polygon/rank capability passes, capture only fixed20 valid V3 pages; no test pages or broad model rerun. Preserve original image dimensions and OCR/feature provenance.
- [ ] Prepare 30 additional unique valid images and annotations, disjoint by page and document from fixed20 and train. Keep pilot replay evidence separate. Use only matched inputs for a 50-page comparison and record any expanded sections with partial input coverage.
- [ ] Run relevant tests once with JUnit/log/exit status; after fixes repeat only affected tests plus necessary integration. Default-off replay must match all40 saved variant-page observations and unchanged input hashes.
- [ ] Run baseline, polygon-only, order-only, V3 both, semantic arbitration-only, Q-only, semantic both, combined available-feature ablations. S fallback control. Report strictF1@.50/.75, per-class TP/FP/FN, ownership checks, target changes, feature/support coverage and CPU time.
- [ ] Distinguish planned/avoided/actually invoked/replayed calls. Only reuse raw MinerU output for identical crop/target coordinate frames; changed frames unavailable or explicitly fixed-response counterfactual. No actual end-to-end latency/speedup claim from CPU replay. A bounded live trial requires controller scope and real valid geometry checks.
- [ ] Independent reviewer checks source/artifact/evaluation evidence; write RESULTS.md with measured gains/regressions and blockers. Leave all flagsFalse and report Step3 incomplete if real training or actual feature capability did not complete. No staging/commit or further user direction question is needed.

## Execution record

Use scratch/cascaded_phase2/progress.md for assigned owner/status, exact source fingerprints, RED/GREEN evidence, preparation provenance, fit/eval exits and reviewer conclusions. Do not mark checkboxes complete from package presence or synthetic fixtures alone. Planner self-check: spec sections map to Tasks1–4; shared interfaces/names match; five Review Focus failures each have owned tests; fitting grouped and valid external; no approval loop is inserted beyond tool escalation/reviewed execution.
