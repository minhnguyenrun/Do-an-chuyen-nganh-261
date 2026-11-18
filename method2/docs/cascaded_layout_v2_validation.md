# Kiểm chứng Cascaded Layout v2 — 07/11/2026

123 kiểm thử đã qua; cả hai notebook đã chạy trong kernel mới bằng MOCK.
Đã chạy smoke một ảnh thật cho S và V3, sau đó chạy mỗi bản trên cùng 20 ảnh validation gốc (226 bbox GT).
Ngưỡng cố định theo kế hoạch, WBF tắt; không điều chỉnh theo kết quả lần đo này. Đây là mẫu nhỏ theo thứ tự, không đại diện toàn bộ benchmark.

| Bản | Stage | TP | FP | FN | Precision | Recall | F1 @0.50 | F1 @0.75 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| S | paddle | 104 | 64 | 122 | 61.90% | 46.02% | 52.79% | 42.13% |
| S | ensemble | 104 | 66 | 122 | 61.18% | 46.02% | 52.53% | 41.92% |
| S | recovery | 104 | 66 | 122 | 61.18% | 46.02% | 52.53% | 40.91% |
| S | final | 104 | 66 | 122 | 61.18% | 46.02% | 52.53% | 40.91% |
| V3 | paddle | 118 | 77 | 108 | 60.51% | 52.21% | 56.06% | 45.13% |
| V3 | ensemble | 128 | 79 | 98 | 61.84% | 56.64% | 59.12% | 48.50% |
| V3 | recovery | 130 | 82 | 96 | 61.32% | 57.52% | 59.36% | 47.95% |
| V3 | final | 130 | 82 | 96 | 61.32% | 57.52% | 59.36% | 47.95% |

## Giới hạn và chi phí thực đo

Review không được tính là detection; GT chưa được phát hiện vẫn là FN. Tỷ lệ dòng được phủ không tương đương độ đúng layout.
Không có confidence giả cho MinerU. AP final là N/A; xem báo cáo từng run để đọc AP baseline.

| Bản | Median giây/trang | P95 giây/trang | Crop MinerU | Token MinerU | Dòng chưa giải quyết |
|---|---:|---:|---:|---:|---:|
| S | 15.84 | 32.96 | 71 | 7774 | 7.03% |
| V3 | 21.88 | 74.81 | 81 | 7868 | 18.06% |

Thời gian bảng trên loại thời gian khởi tạo model; manifest ghi riêng load/cache/wall time. Chưa đạt mục tiêu 30–50ms/trang.

## Quyết định và nguồn lỗi

### S

- Vùng MinerU được chấp nhận: 0.
- Trạng thái evidence: {"ok": 35, "incomplete": 10, "no_target_lines": 6, "invalid_output": 26, "budget_exhausted": 7}.
- Lý do từ chối: {"missing_target_lines": 77, "insufficient_quality_gain": 83, "model_unavailable_review_orphan": 39, "model_unavailable_keep_baseline": 10, "overlaps_locked_region": 1}.

### V3

- Vùng MinerU được chấp nhận: 0.
- Trạng thái evidence: {"no_target_lines": 9, "ok": 52, "incomplete": 9, "invalid_output": 20, "budget_exhausted": 22}.
- Lý do từ chối: {"model_unavailable_keep_baseline": 16, "insufficient_quality_gain": 111, "missing_target_lines": 103, "model_unavailable_review_orphan": 44, "semantic_gate_failed": 2, "overlaps_locked_region": 1}.

## Đọc kết quả

Bản S giữ TP ở IoU0.50 nhưng thêm hai FP từ bổ sung YOLO; final chưa vượt baseline. Đọc bảng V3 ở trên để đánh giá mức lợi ích thực tế của biến thể đó.
Các cổng bảo toàn baseline không bảo đảm tăng F1. MinerU vẫn có output sai định dạng và bị cắt; trường hợp không đủ bằng chứng được KEEP/REVIEW thay vì ép nhãn.
WBF chỉ kiểm thử chức năng và mặc định tắt, chưa có kết luận thực nghiệm WBF trên tập này.

## File kết quả

- S: `method2/outputs/cascaded_v2_validation/s/real/v2/48fab27a4ab8/report_literal/metrics.json`; snapshot/ và source_errors giữ chi tiết để phân tích.
- V3: `method2/outputs/cascaded_v2_validation/v3/real/v2/0aba6457c29c/report_literal/metrics.json`; snapshot/ và source_errors giữ chi tiết để phân tích.
- Bảng ghép: `method2/outputs/cascaded_v2_validation/comparison_v2/comparison.csv`.
