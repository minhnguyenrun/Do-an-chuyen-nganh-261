# Kiểm chứng v2.1 ngày 07/11/2026

Phạm vi theo yêu cầu: sửa mã, kiểm thử và chạy **một ảnh thật**; không chạy lại tập20 ảnh.

## Kiểm thử

92 kiểm thử đạt, gồm29 kiểm thử attribution mới. Hai notebook S/V3 chạy từ kernel mới
với adapter MOCK, mỗi notebook một ảnh. Các bài kiểm thử v1/v2, cache, dataset,
model failure/OOM/hết ngân sách và không có orphan vẫn đạt.

Đã kiểm tra bbox rộng nhưng không nhiễm chữ, context-only, snap dọc, dòng xoay,
chia ngang chưa chứng minh, partition heading/body, orphan80%, bảo toàn mọi dòng
fallback nhiều vùng, ABSORB nguyên khối/hoàn tác, chặn nhiều hàng xóm/cột khác,
nhiễm sau snap, ownership sau snap, beam giới hạn và log không tính Q khi rỗng.

## Một ảnh thật

Chạy `run_cascaded.py --variant s --split valid --count 1`, CUDA trên RTX3050Ti4GB,
đọc weights có sẵn trong cache, không dùng mock hoặc chuyển CPU.
Ảnh validation đầu tiên `image_0`, kích thước1025x1025, 87 dòng OCR, 5 bbox GT.

| Pha | Thời gian nạp (s) | Xử lý ảnh, chưa tính nạp (s) | Wall pha (s) |
|---|---:|---:|---:|
| PP-DocLayout-S + PP-OCRv5 mobile det/rec | 53.075 | 4.402 | 59.291 |
| DocLayout-YOLO | 22.384 | 1.960 | 24.847 |
| MinerU2.5-2509-1.2B Transformers | 33.747 | 63.205 | 97.439 |

Tổng wall ba pha181.578s; phần xử lý không tính nạp69.567s. Đây là một lượt smoke,
không phải benchmark ổn định. MinerU gọi3 crop, tổng600 token:66,22,512.
Hai output hợp lệ, một output chạm512 token và bị đánh dấu incomplete.
PyTorch peak reserved của pha MinerU2598MiB; nvidia-smi ghi đỉnh2733MiB toàn GPU
với chu kỳ lấy mẫu200ms, có thể gồm process khác và bỏ lỡ đỉnh ngắn. Không gặp OOM.

Không có đề xuất MinerU được nhận trên ảnh này:

- Target0: hai bbox `LINE_BISECTION`, một `PURE_CONTEXT_BOX` -> `NO_VALID_PARTITION`.
- Target1: `PROTECTED_NONTEXT_OVERLAP` -> `NO_VALID_PARTITION`.
- Target2: `MODEL_INCOMPLETE`, dừng trước hậu xử lý/Q.

Vì không có partition khả thi nên không ghi lỗi giả `insufficient_quality_gain`.
Final giữ3 vùng baseline; 85/87 dòng có chủ, 2 dòng còn lại vào review.
Strict F1@0.50 và0.75 đều0.25, TP1/FP2/FN4, giống baseline trên **riêng ảnh này**.
Không có bằng chứng cải thiện chất lượng từ lượt smoke. Chưa chạy thật biến thểV3
trong lần sửa này; chưa đánh giá lại20 ảnh của hai notebook.

## Kết quả lưu

Thư mục: `method2/outputs/cascaded_v2_1_smoke/s/real/v2.1/181f929d3fd6/`.

- `manifest.json`: config, phiên bản, tải model/wall/VRAM.
- `image_0/result.json`, `paddle.json`, `yolo.json`, `mineru.json`, các crop.
- `image_0/preview_0.png` tới `preview_4.png`: layout/GT/crop và checkpoint.
- `report_literal/metrics.json`, `summary.csv`, `per_class.csv`, `mineru_boxes.csv`,
  `decisions.csv`, snapshot đã đánh giá.
- `latest_source_verification.json`: chạy lại **chỉ hậu xử lý** trên đúng output thô
  của ảnh này sau chỉnh ownership sau snap cuối cùng. Không thêm lượt suy luận model.
  Final, decisions, evidence và review khớp hoàn toàn; lưu hash mã nguồn cuối để đối chiếu.

Có cảnh báo cuDNN của Paddle (bản biên dịch9.9, runtime9.5), nhưng suy luận thực tế
vẫn hoàn tất. Lượt smoke không chứng minh tương thích với mọi dữ liệu hoặc workload.

Khi chạy20 ảnh: Restart Kernel, Run All từng notebook, giữ cùng validation/taxonomy/config.
Không dùng kết quả một ảnh này để chọn ngưỡng hoặc kết luận tốt hơn Dual-Skip.
