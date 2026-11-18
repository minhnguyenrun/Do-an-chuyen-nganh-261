# Fast Cascaded v3 design

2026-11-17. Trạng thái: thực thi theo yêu cầu người dùng; đây là Phase 1 thử nghiệm có kiểm chứng, không tuyên bố hoàn tất toàn bộ Fast Cascaded v3. Agent chính chỉ điều phối; planner chỉ sửa tài liệu, tester viết/chạy test, implementer chỉ sửa production.

## Mục tiêu và nguồn

Bám docs/superpowers/plans/Tối Ưu Hóa Cascaded Layout.docx, gồm checklist 4 bước ở đoạn 195-199 trong scratch/research_cascaded/docx_reread/full_extraction.txt; báo cáo đầy đủ READING_REPORT.md cùng thư mục. Mục tiêu đồng thời cải thiện layout F1 và giảm chi phí MinerU trên Windows GPU 4GB. Các con số >68–72% F1 và 5.4–9.2s/trang trong DOCX là dự phóng; không dùng làm kết quả hoặc cam kết.

Baseline active v2.1 ở method2/src/cascaded: fusion.prepare_layout, attribution.finish_layout, runner.Experiment. S/V3 cached 20 valid mỗi nhánh khớp source cc54410437545d8fc7e24d632f64a37c11184b64d2b5ed53624777fb82e5be75. Baseline F1@.50 final S 52.3929%, V3 59.2255%; source/report và provenance trong scratch/baseline_testing/BASELINE_REPORT.md. Repo hiện nhiều source/tests cascaded chưa tracked từ trước; chỉnh in-place có scoped snapshot, không thể tái tạo baseline bằng worktree từ HEAD. Không stage/commit toàn repo hoặc ghi đè cache/user changes.

## Lộ trình bám DOCX

Step 1 audit/reconcile: đã có PP fallback, không mutual drop, review-only unknown, nhiều aliases, title-compatible, PP dominance, WBF optional. Phải ghi delta còn thiếu (weighted calibrated score/coarse groups đầy đủ/conflict .30) nhưng không áp score weights theo dự phóng vào mã active chưa calibration.

Step 2 Phase 1 executable: Active Shrink, text Envelope Refit, 1D Carving có kiểm chứng core/ownership; adaptive crop 28px và dynamic max_new_tokens; replay CPU trên raw cache valid, ablation tách hình học và Q diagnosis, kế tiếp fake-adapter inference policy tests. Chỉ chạy real model sau khi phần mềm và replay đạt invariant, nếu runtime sẵn có thì dùng subset valid; không dùng test để tune.

Step 3 follow-on bắt buộc ghi pending: V3 order/mask thật cần adapter capability; Model2Vec-LightGBM cần train data/weights/calibration. Không xưng hoàn tất semantic learned nếu chỉ heuristic. Step 4 pending: verify official llama.cpp/backend+mmproj/GGUF và Windows CUDA/VRAM trước đổi weights/backend; không cài model mới ngầm. Những phase này chưa thuộc implementation nhỏ hiện tại.

## Ranh giới và interfaces

Tạo method2/src/cascaded/geometry_refine.py với refine_candidate_bbox(bbox:list[float], label:str, target_lines:list[Line], lines:list[Line], owners:dict[str, str], confirmed:list[Proposal], cfg:Config)->dict. Kết quả luôn có bbox, core_line_ids, excluded_line_ids, codes, changed, valid. bbox là page XYXY; valid=False nghĩa giữ rejection cũ, không đưa bbox chưa chứng minh vào partition. Không mutation input. Codes gồm ACTIVE_SHRINK_APPLIED, OCR_ENVELOPE_REFIT_APPLIED, NON_TEXT_CARVING_APPLIED, GEOMETRY_REFINEMENT_REJECTED. Return nguyên bbox nếu flag enable_geometry_refinement=False. Chỉ target core ban đầu coverage>=.80 là evidence chắc; no-core/line chia giữa thân không được biến thành valid bằng bỏ dòng.

Trong attribution.candidate_boxes chỉ gọi refine sau mapping crop→page và label normalization, trước hard bisection gate; giữ raw/page/refined bbox trong history. Sau geometry phải chạy lại toàn bộ page-line intersections, coverage, foreign ink, protected nontext, cluster, semantic, partition và final atomic validation. Không suy ra đúng nhãn từ việc có OCR.

Tạo method2/src/cascaded/inference_policy.py với build_inference_request(target:dict, lines:list[Line], width:int, height:int, cfg:Config)->dict trả crop_bbox:list[int], layout_size:tuple[int, int], max_new_tokens:int, reason:str. Legacy mode giữ context_box, 768 square, max 512. Adaptive mode: padding 10% each dimension, capped 1.5 median line height nhưng ít nhất 1px, clamp page; area budget 448² cho <=3 target lines và <=1 fallback, 560² otherwise; maintain aspect ratio, round down multiples 28, min 28, không upsample quá area budget; tokens=min(768, max(96, 32+28*max(N_target_lines, N_target_source_proposals))) với source count=len(unique source_ids), tối thiểu 1. Không đổi max_crops/target ordering ở Phase 1. Config flags enable_geometry_refinement=False, enable_adaptive_inference=False; profile mới phải explicit.

MinerUAdapter.predict(crop, *, max_new_tokens=None, layout_image_size=None)->dict giữ positional callers. Runner truyền kwargs chỉ khi adaptive bật; MockAdapter và fake tests theo contract. Adapter phải inspect official installed mineru-vl-utils 2.0.8 source/API trước chọn mechanism per-call sampling/image size; snapshot/restore state nếu official sequential client buộc đổi settings. Unsupported override phải báo rõ và giữ legacy, không silent claim adaptive applied. Không monkey patch token grammar hoặc early-stop unsupported. Hiện measured_generate instrument giữ completion/token handling và finally restore.

Runner lưu request policy/effective size/token cap trong evidence để đối chiếu. Request config/source hash mới tạo output riêng, raw baseline chỉ đọc. Offline replay chỉ đánh giá geometry trên đúng response/crop gốc; không đem raw responses của square 768 gán cho crop/aspect mới.

## Thuật toán hình học và guard

Text roles: text, title, doc_title, paragraph_title, caption, footnote, list_item; formula_number không tự đổi taxonomy. Active shrink chỉ loại partial peripheral line khi 0<CovY<.45 hoặc 0<CovX<.30, core còn >=.80; không shrink nếu core bị mất, line nằm giữa 2 core theo trục, polygon xoay/invalid không chứng minh được hoặc không còn core. Kiểm tra mọi lines, owned/orphan, không chỉ target. Owned non-ABSORB vẫn phải loại hoàn toàn hoặc rejection cũ; không trộm ownership.

Refit chỉ text roles, lấy envelope các dòng target hợp lệ sau shrink; có thể snap Y khi CovX>=.75, CovY>=.45, dòng near-horizontal <=10deg; X expansion<=15% width và không sweep thêm foreign lines. Giữ core, không overmerge cột. Nontext image/table/formula bbox không refit bằng OCR envelope. Carving chỉ text candidate sát nontext boundary khi original overlap>.15 nhưng union OCR polygon ink overlap<=.05, cắt trục lấn nông nhất giữ core >=.80, rồi toàn bộ guard chạy lại. Không bật blanket protected threshold .35 trong Phase 1: DOCX exemption là hypothesis follow-on vì có thể che label errors. Nếu carve không đưa overlap về <=.15 thì reject như cũ.

Giữ frozen ownership, no duplicate line IDs, complete fallback line coverage, final atomic rollback. Partial commit trong DOCX chỉ được đưa vào khi chứng minh không drop/duplicate baseline lines và không giữ parent overlap với child; Phase 1 giữ atomic baseline nếu chưa có proof, report measured blocker cụ thể. Invalid/zero/nonfinite polygons/bboxes và geometry exceptions không rescue; log reason rồi xử lý cũ.

## Q và quyết định điều phối

Giữ Q legacy = .45 coverage - .30 split + .25 semantic, strict Q new>Q keep+.10 và semantic>=.60. Phải đo candidate count trước/sau geometry, Q keep/Q new, reject reason, partitions có khả thi không. Q upper bound .70 làm REPLACE khi KEEP Q>=.60 có thể bất khả thi; không tuyên bố shrink giúp accuracy nếu chỉ tăng candidate pool rồi tất cả bị Q chặn. Có thể ADD orphan vì KEEP Q 0 và cải thiện semantic split case mà không đổi Q.

Ruling: không gộp Q v3 hoặc giảm margin dưới flag geometry. Nếu replay cho thấy geometry inert do Q, dừng để lập thí nghiệm scoring riêng flag enable_geometry_quality=False, ablation baseline/geometry/geometry+quality. Fit bonus .20 chỉ là hypothesis mới chưa được quyết định triển khai. Giá của quyết định bảo thủ: có thể bỏ lỡ REPLACE gains đợt 1; lợi ích là tránh regression/confounding do thay geometry và scoring cùng lúc. Q-pruning runtime-only từ design cũ không thay thế các mục tiêu này.

## Đo và tiêu chí hoàn tất Phase 1

Tester phải snapshot/hash source trước; chạy existing 74 tests, red tests của geometry/inference policy trước production, rồi green focused+regression. Replay cả 40 cached valid với baseline config và geometry bật, same GT/literal taxonomy; baseline replay phải khớp 40/40 saved final. Optimized final ĐƯỢC thay đổi; yêu cầu no lost/duplicate owned lines, semantic/nontext/atomic safety, F1 strict/loose@.50/.75/perclass, TP/FP/FN và raw geometry/Q checkpoint deltas đầy đủ. Candidate geometry cùng crop cached có thể đo accuracy nhưng không đo adaptive inference accuracy. Report crop calls/tokens thực hoặc cached historical estimate đúng nhãn, không biến CPU replay walltime thành inference latency.

Accept default chỉ sau không regression strict F1@.50/.75 cả 2 variants và có lợi ích đo được; nếu chưa có accuracy gain hoặc tradeoff, giữ default False, đưa kết quả thật và artifacts, không ép tăng F1 bằng nới luật. Phase 1 trial có thể kết thúc với kết luận không cải thiện; Step 3/4 vẫn pending. Không gọi Phase 1 hoàn tất toàn bộ Fast V3.
