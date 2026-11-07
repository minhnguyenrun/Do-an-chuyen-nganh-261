# Kế hoạch Hiện thực Chi tiết (Implementation Plan)
## Kiến trúc Semantic-Guided Dual-Skip Document Parsing (v4.0)

> Bản sao thuận tiện lưu tại `method2/Ke_hoach_hien_thuc_Dual_Skip_v4.md`.  
> File gốc quy chuẩn theo chuẩn Superpowers lưu tại: [`docs/superpowers/plans/2026-10-24-semantic-dual-skip-parser.md`](../docs/superpowers/plans/2026-10-24-semantic-dual-skip-parser.md).

Vui lòng xem chi tiết 12 nhiệm vụ (tasks) TDD và kiểm thử độc lập tại liên kết trên.

## Mã hiện thực

- Mã nguồn: [`src/`](src/), cấu hình: [`configs/default.yaml`](configs/default.yaml).
- Notebook chạy chính: [`Dual_Skip_v4.ipynb`](Dual_Skip_v4.ipynb), nhận ảnh/PDF và hiển thị trực tiếp.
- Hướng dẫn cài đặt, chạy notebook/CLI, đánh giá và giới hạn: [`README.md`](README.md).
- Kiểm thử: [`tests/`](tests/), chạy `python -m pytest tests -q` từ `method2`.

Đã có các nhóm chức năng của 12 task: schemas/provenance, crop/coordinates, line ownership/residuals, graph/DAG, adapters, candidates/prompts/budget, evidence features, dual-skip validation, scoring/refinement, revision/rollback/assembly, pipeline/notebook và evaluation A0–A9. Phần API trong kế hoạch gốc đã được thay bằng notebook và suy luận GLM trực tiếp theo yêu cầu cập nhật.

Các test dùng mô hình giả lập hoặc model/processor tiêm vào. Chạy mô hình thật và benchmark cần runtime PaddleOCR, PyTorch/Transformers, trọng số GLM-OCR và dữ liệu ground truth riêng; không cần server. Các ràng buộc v4 được ưu tiên hơn những ví dụ test đơn giản trong kế hoạch: không chấp nhận bảng chỉ nhờ HTML hợp lệ hoặc đổi nhãn chỉ nhờ prompt chuyên biệt.

