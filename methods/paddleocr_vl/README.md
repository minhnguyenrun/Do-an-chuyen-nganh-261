# PaddleOCR-VL / PP-DocLayout

Phương pháp phân tích layout tài liệu dùng [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) và bộ công cụ layout phân tích tài liệu tiên tiến **PP-DocLayout** (model mặc định: `PP-DocLayout_plus-L`).

## Cấu trúc

- `src/run_inference.py` — code chạy layout detection, xuất dự đoán chuẩn COCO ra `outputs/predictions/paddleocr_vl/<dataset>/*.json`
- `notebooks/paddleocr_vl_report.ipynb` — notebook kết hợp inference, tính metric (Strict/Loose), vẽ biểu đồ và trực quan hóa so sánh Ground Truth vs. Prediction
- `report/` — báo cáo PDF cho phương pháp này

## Cách chạy

1. **Cài đặt thư viện phụ thuộc:**
   - Cài đặt cho CPU:
     ```bash
     pip install -r methods/paddleocr_vl/requirements.txt
     ```
   - Cài đặt tăng tốc GPU (CUDA 12.x, khuyến nghị):
     ```bash
     pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
     pip install -U "paddleocr[doc-parser]" matplotlib opencv-python tqdm pillow
     ```

2. **Chạy inference theo batch (script):**
   ```bash
   # Chạy layout detection trên N ảnh đầu tiên của datasets/doclaynet/processed/images
   python methods/paddleocr_vl/src/run_inference.py --dataset doclaynet --limit 20

   # Bỏ qua ảnh đã có dự đoán (tiếp tục chạy sau khi gián đoạn hoặc chạy toàn bộ dataset)
   python methods/paddleocr_vl/src/run_inference.py --dataset doclaynet --limit 0 --skip-existing

   # Đánh giá metric Strict & Loose, xuất biểu đồ metrics_chart.png
   python scripts/evaluate_predictions.py --method paddleocr_vl --dataset doclaynet --loose --allow-incomplete --plot

   # Vẽ hình đối chiếu Ground Truth vs. Dự đoán
   python visualization/scripts/visualize_predictions.py --method paddleocr_vl --dataset doclaynet --limit 20 --output-dir outputs/reports/paddleocr_vl_doclaynet
   ```

3. **Chạy thử nghiệm tương tác trong Jupyter Notebook:**
   - [paddleocr_vl_report.ipynb](notebooks/paddleocr_vl_report.ipynb) — Notebook báo cáo tổng hợp đánh giá và biểu đồ.
   - [paddleocr_vl.ipynb](notebooks/paddleocr_vl.ipynb) — Notebook thử nghiệm tương tác và trực quan hóa song song (Side-by-side).

Kết quả JSON lưu tại `outputs/predictions/paddleocr_vl/doclaynet/<image_id>.json` (mỗi ảnh một file, gồm `image_id`, `file_name`, `width`, `height`, `detections`: list `{category_id, category_name, bbox, score}`).
Hình đối chiếu lưu tại `outputs/reports/paddleocr_vl_doclaynet/<image_id>.png`.

## Chiến lược Mapping nhãn (Taxonomy Mapping)

Mô hình PP-DocLayout được huấn luyện trên bộ nhãn chi tiết gồm 20 lớp, trong khi tập dữ liệu benchmark DocLayNet có 11 lớp chuẩn. Script `run_inference.py` áp dụng quy tắc ánh xạ và lọc nhãn có căn cứ thực nghiệm như sau:

### 1. Bảng ánh xạ sang DocLayNet (11 lớp)

| Nhãn PaddleOCR (PP-DocLayout) | Nhãn DocLayNet | ID | Ghi chú |
|---|---|---:|---|
| `table` | `Table` | 9 | Khớp trực tiếp bảng biểu |
| `formula` | `Formula` | 3 | Công thức toán học (inline & display) |
| `footnote` | `Footnote` | 2 | Ghi chú chân trang |
| `image`, `chart` | `Picture` | 7 | Hình ảnh, biểu đồ, sơ đồ |
| `paragraph_title` | `Section-header` | 8 | Tiêu đề mục, đề mục đoạn |
| `doc_title` | `Title` | 11 | Tiêu đề chính của tài liệu |
| `figure_title`, `table_title`, `chart_title` | `Caption` | 1 | Chú thích hình ảnh và bảng biểu |
| `header`, `header_image` | `Page-header` | 6 | Tiêu đề đầu trang (running header) |
| `footer`, `footer_image` | `Page-footer` | 5 | Tiêu đề chân trang (running footer) |
| `text`, `abstract`, `content`, `reference`, `reference_content`, `aside_text`, `algorithm` | `Text` | 10 | Các khối văn bản đoạn văn, tóm tắt, mục lục, trích dẫn tài liệu, ghi chú bên lề |

*(DocLayNet có nhãn `List-item` (4), tuy nhiên các model họ DocLayout thường nhận diện từng dòng list dưới dạng khối `text` hoặc `content`)*.

### 2. Nhãn loại bỏ (`DEFAULT_DROP_LABELS`)

Qua phân tích và thực nghiệm so khớp IoU trên dữ liệu thực tế, các nhãn sau được loại bỏ mặc định:
- **`seal`** (con dấu): Đặc thù tài liệu/hợp đồng tiếng Trung, không có trong taxonomy DocLayNet.
- **`formula_number`**: Nhãn đánh số phương trình (vd: `(1)`), trong DocLayNet được gộp chung vào box `Formula` hoặc `Text`. Tách riêng sinh box trùng lặp.
- **`number`**: Các số rời rạc (số trang, chỉ mục). Kiểm chứng thực nghiệm cho thấy **80% box `number` không có GT tương ứng** (false positive).
- **Hiệu quả thực nghiệm**: Khi loại bỏ 3 nhãn này, **Precision tăng từ 0.693 lên 0.740 (+4.7%)** và **F1 tăng từ 0.741 lên 0.767 (+2.6%)**, trong khi số True Positives giữ nguyên 100% (không làm giảm Recall).
- Người dùng có thể truyền flag `--keep-all` vào `run_inference.py` nếu muốn giữ lại toàn bộ nhãn với `category_id = None`.

### 3. Ánh xạ sang PubLayNet (5 lớp)

- `Text` (1): `text`, `abstract`, `content`, `reference`, `aside_text`, `footnote`, `header`, `footer`, `algorithm`
- `Title` (2): `doc_title`, `paragraph_title`
- `Table` (4): `table`
- `Figure` (5): `image`, `chart`, `figure_title`, `table_title`, `chart_title`

## Ghi chú kỹ thuật

- **Workaround tương thích Windows**: 
  1. Thư viện `torch` được nạp trước `paddleocr` để tránh lỗi xung đột DLL `[WinError 127] shm.dll` giữa runtime của Paddle và PyTorch khi `paddlex` gọi `modelscope`.
  2. Model được khởi tạo với `enable_mkldnn=False` để khắc phục lỗi PIR instruction của PaddlePaddle 3.3 trên Windows CPU (`ConvertPirAttribute2RuntimeAttribute not support pir::ArrayAttribute<pir::DoubleAttribute>`).
- **So sánh với Docling**:
  - PaddleOCR phát hiện tốt cả `Page-header` và `Page-footer` (vốn là điểm yếu của Docling do model RT-DETR của Docling ép kích thước trang về 640x640 làm mất các dải hẹp ở biên).
  - Tốc độ xử lý trên CPU: ~2-2.5s/ảnh.
