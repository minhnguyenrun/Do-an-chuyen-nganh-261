# Fast Cascaded v3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Thử cải thiện F1 và chi phí Cascaded baseline theo DOCX, thực thi Phase 1 gồm audit Step 1 và Step 2 có kiểm chứng, không thay mục tiêu thành runtime-only.

**Architecture:** Giữ PP fallback và frozen OCR ownership v2.1; thêm geometry refinement trước candidate gates và adaptive inference request có flag độc lập. Replay raw valid đánh giá geometry/F1; inference request tests và real smoke riêng chứng minh policy/cost, không tái diễn giải cache dưới crop mới. Giữ Q legacy, đo blocker trước mọi hypothesis scoring riêng.

**Tech Stack:** Python 3.12, dataclasses, Shapely, numpy, pytest, existing Paddle/YOLO/MinerU adapters; không dependency/model mới trong Phase 1.

**Spec:** docs/superpowers/specs/2026-11-17-fast-cascaded-v3-design.md

## Global Constraints

- Config enable_geometry_refinement=False, enable_adaptive_inference=False; legacy callers/default tương thích.
- Agent chính chỉ điều phối; tester-only viết/chạy tests, production implementer-only sửa src; planner chỉ docs/briefs.
- In-place scoped snapshots vì source baseline untracked; không stage/commit toàn repo hoặc ghi đè cache/user changes.
- Valid dùng thử/tune; test chỉ final. Cùng image IDs/GT/literal taxonomy để so F1.
- Q legacy và strict margin .10 giữ nguyên; no ownership loss/duplicates, frozen owners và atomic fallback.
- Không áp unsupported MinerULogitsProcessor grammar/Y-stop; verify primary installed API trước adaptive integration.
- Các số 7–12x và 68–72% F1 từ DOCX là projections, không phải success criteria đã đo.

## Review Focus

- Grazing foreign line phía mép với core đầy đủ: rescue chỉ nếu không mất core; Task 1 tests.
- No-core/rotated/invalid polygon hay true line-bisection: reject giữ baseline; Task 1 tests.
- Caption sát nontext, multi-column và duplicate ownership sau refit: guard chạy lại; Task 1 tests.
- Thin/edge crop, client exception và settings leak: clamp/grid/token cap/restore; Task 2 tests.
- Cache crop frame mismatch, Q saturation và metric confounding: legacy 40/40 replay và geometry/Q breakdown; Task 3 tests/report.

---

### Task 1: Safe candidate geometry and unchanged Q diagnostics

**Files:** Create method2/src/cascaded/geometry_refine.py; Modify attribution.py:candidate_boxes/finish_layout, types.py:Config; Test method2/tests/test_cascaded_geometry_refine.py (tester owner).
**Interfaces:** Consumes Proposal, Line, Config; Produces refine_candidate_bbox(bbox, label, target_lines, lines, owners, confirmed, cfg)->dict with bbox/core_line_ids/excluded_line_ids/codes/changed/valid; exact algorithm and fields in Spec. No input mutation.

- [ ] Tester writes tests: flag off same bbox; orphan body candidate core coverage 1 + grazed line .04 rescues and can ADD under Q legacy; no-core two fragments remains rejected; inner bisection/rotated/invalid polygon reject; core shift rollback; nontext not refit; caption carve preserves core/rechecks foreign; multiple owners/column reject; complete fallback and atomic rollback; current Q high KEEP remains KEEP, low semantic KEEP vs valid partition can win only through existing quality difference.
- [ ] Tester runs python -m pytest tests/test_cascaded_geometry_refine.py -q from method2, records intended RED due missing module/behavior. Existing tests unchanged.
- [ ] Implementer builds helper and integrates geometry only behind flag; normalizes label before geometry, logs raw/refined coordinates; retains all post gates/Q/completion/atomic guard. No Q v3 or partial commit unless proof and independent review authorize bounded follow-up.
- [ ] Tester runs focused + existing cascaded tests; expects PASS and untouched input assertions. Reviewer audits no foreign line skip hidden by shrink and safety after every transform.
- [ ] Record scoped diff/snapshot in scratch/fast_cascaded_v3; do not commit untracked baseline.

### Task 2: Adaptive token and aspect preserving inference policy

**Files:** Create method2/src/cascaded/inference_policy.py; Modify types.py:Config, runner.py: target loop, adapters.py:MinerUAdapter/MockAdapter; Test method2/tests/test_cascaded_inference_policy_v3.py (tester owner).
**Interfaces:** build_inference_request(target, lines, width, height, cfg)->dict crop_bbox, layout_size, max_new_tokens, reason. MinerUAdapter.predict(crop, *, max_new_tokens=None, layout_image_size=None)->dict; existing positional calls valid. Task 1 only shares Config; perform shared file edits sequentially.

- [ ] Tester writes legacy request/optional kwargs tests; adaptive formulas 96/116/732/768 token cases; 28 grid/aspect/page edge/thin crop; no empty crop; fake official client verifies effective override and restoration on exception; unsupported capability report; completion/token instrumentation not lost; cache policy field/signature change.
- [ ] Tester runs python -m pytest tests/test_cascaded_inference_policy_v3.py -q; records RED before implementation.
- [ ] Implementer first reads installed mineru-vl-utils client/sampling/image resize source and records supported primary API in scratch/fast_cascaded_v3/api_verification.md. Implements pure policy per Spec and safe override only if supported, then runner integration/effective request logging. Keeps budget 8/order; full-page/batch/grammar/Y-stop deferred pending independent verification, not silently claimed implemented.
- [ ] Tester runs focused regression with fake adapters. If real runtime available, run one valid image adaptive smoke and record actual pixels/cap/remaining completion; no GPU parallel runs or downloads.
- [ ] Record coverage/limitations; no commit whole untracked tree. Adaptive flag stays False until paired measurement.

### Task 3: Cached valid ablation and honest improvement decision

**Files:** Create scratch/fast_cascaded_v3/replay_evaluation.py, ablation_results.json, RESULTS.md; tests harness owned exclusively by tester. Reuse existing report/evaluation modules and baseline harness as reference, never overwrite baseline artifacts.
**Interfaces:** read baseline cached responses/targets/crop frames and Config; evaluate baseline(false, false), geometry(true, false). Adaptive config(true/false, true) is request/cost-real trial only, not simulated accuracy with old crop responses. JSON includes variants/imageids/source hash/config/GT sha/metrics/geometry codes/Q reasons/calls/tokens/historical seconds/measured replay wall/limitations.

- [ ] Tester pins evaluator meaningful fixtures and no ownership loss/duplicate/unchanged baseline replay assertions; run intended RED if harness absent. Capture before hashes/git status/74 test baseline in ledger.
- [ ] Tester implements CPU replay using saved raw model responses, keeps crop_bbox coordinates and original cache read-only; compare baseline 40/40 saved final, then geometry variant against same GT. Production implementer fixes only exposed defects, tester reruns relevant checks.
- [ ] Report strict/loose F1@.50/.75, per class TP/FP/FN; candidate pool pre/post geometry, partition feasibility, Q keep/Q new/reject counts; geometry accepted new TP/new FP/FN removed. Replay time is CPU work, skipped historical crop seconds are projections, not optimized latency.
- [ ] If geometry all blocked by Q, preserve quality and produce bounded next proposal for separate enable_geometry_quality=False experiment plus baseline/geometry/geometry+quality ablation; no automatic .20 fit bonus or lowering margin. If partial commit not implemented, report exact incomplete coverage / parent overlap proof barrier. If geometry regresses, keep default off and record actual deltas.
- [ ] Independent review, focused + existing tests, and real paired valid subset if ready; final delivery links plan/spec/results and says Phase 1 implemented/tried vs Step 3/4 pending. Only enable profile by default if both variant F1 thresholds nonregress and measured benefit; otherwise retain explicit experimental flags.

## Follow on DOCX phases not completed by Phase 1

Step 3: task plan for adapter actual order/mask serialization (S fallback), graph ABSORB indentation, semantic train/calibration using train/valid only. Step 4: task plan verify llama.cpp official API/GGUF+mmproj/Windows CUDA and paired quality/VRAM benchmarks. Record these as pending in the ledger; user request for trial does not justify claiming full architecture finished.

## Execution handoff

User chose execution through subagents; continue after linking spec/plan, no renewed approval pause. Read scratch/fast_cascaded_v3/task1_geometry.md, task2_inference.md, task3_replay.md; independent tester writes RED tests before production implementation, reviewer checks each task. No planner/controller production/test editing. Self-review: all Phase 1 spec sections mapped to tasks 1–3; Step 3/4 explicitly pending, interface keys/signatures consistent, each Review Focus has owner tests.
