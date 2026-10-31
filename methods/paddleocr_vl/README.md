# PaddleOCR-VL (layout stage: PP-DocLayoutV2)

Phương pháp phân tích layout tài liệu **PaddleOCR-VL** của [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR).

PaddleOCR-VL gồm hai tầng: **PP-DocLayoutV2** (phát hiện layout + thứ tự đọc) và **PaddleOCR-VL-0.9B** (nhận dạng nội dung từng khối). Bounding box chỉ do tầng layout tạo ra, nên `run_inference.py` chạy đúng tầng này và lấy **toàn bộ cấu hình từ pipeline chính thức** `PaddleOCR-VL.yaml` của PaddleX: model `PP-DocLayoutV2`, ngưỡng theo từng lớp, layout NMS, unclip ratio, chế độ merge box, và bước lọc box chồng lấn (`filter_overlap_boxes`) mà pipeline áp dụng ngay sau khi phát hiện. Tầng nhận dạng VLM không ảnh hưởng đến box nên được bỏ qua.

> Trước đây code dùng `PP-DocLayout_plus-L`, là một model layout độc lập, **không phải** model của pipeline PaddleOCR-VL. Prediction cũ (không có trường `layout_model`) tự được chạy lại khi dùng `--skip-existing`. Có thể chọn phiên bản mới hơn của pipeline bằng `--pipeline PaddleOCR-VL-1.5` hoặc `PaddleOCR-VL-1.6` (dùng PP-DocLayoutV3).

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

Mô hình PP-DocLayoutV2 được huấn luyện trên bộ nhãn chi tiết gồm 25 lớp, trong khi tập dữ liệu benchmark DocLayNet có 11 lớp chuẩn. Script `run_inference.py` áp dụng quy tắc ánh xạ và lọc nhãn có căn cứ thực nghiệm như sau:

### 1. Bảng ánh xạ sang DocLayNet (11 lớp)

| Nhãn PaddleOCR (PP-DocLayoutV2) | Nhãn DocLayNet | ID | Ghi chú |
|---|---|---:|---|
| `table` | `Table` | 9 | Khớp trực tiếp bảng biểu |
| `display_formula` | `Formula` | 3 | Công thức dạng khối (display). `inline_formula` bị loại vì thuộc khối Text |
| `footnote` | `Footnote` | 2 | Ghi chú chân trang |
| `image`, `chart` | `Picture` | 7 | Hình ảnh, biểu đồ, sơ đồ |
| `paragraph_title` | `Section-header` | 8 | Tiêu đề mục, đề mục đoạn |
| `doc_title` | `Title` | 11 | Tiêu đề chính của tài liệu |
| `figure_title`, `vision_footnote` | `Caption` | 1 | Chú thích hình ảnh, bảng biểu, biểu đồ. `vision_footnote` (văn bản gắn với hình/bảng) được DocLayNet gán là Caption |
| `number` | `Page-footer` (nửa dưới trang) / `Page-header` (nửa trên) | 5 / 6 | Số trang rời; DocLayNet gán số trang vào chân/đầu trang |
| `header`, `header_image` | `Page-header` | 6 | Tiêu đề đầu trang (running header) |
| `footer`, `footer_image` | `Page-footer` | 5 | Tiêu đề chân trang (running footer) |
| `text`, `vertical_text`, `abstract`, `content`, `reference`, `reference_content`, `aside_text`, `algorithm` | `Text` | 10 | Các khối văn bản đoạn văn, tóm tắt, mục lục, trích dẫn tài liệu, ghi chú bên lề |

*(DocLayNet có nhãn `List-item` (4), tuy nhiên các model họ DocLayout thường nhận diện từng dòng list dưới dạng khối `text` hoặc `content`)*.

### 2. Hai quyết định ánh xạ chọn trên tập validation

PP-DocLayoutV2 có hai nhãn không có tương ứng 1-1 trong DocLayNet. Hai cách ánh xạ dưới đây được chọn bằng cách so 6 phương án trên **400 trang đầu của DocLayNet validation** (không dùng tập test để chọn), rồi xác nhận lại trên 300 trang đầu của test; kết quả cùng chiều:

| Phương án (Strict, IoU ≥ 0.5) | F1 validation | F1 test (300 trang) |
|---|---:|---:|
| Loại bỏ `number`, `vision_footnote` → Footnote (cấu hình cũ) | 0.670 | 0.684 |
| `number` → Page-footer/Page-header theo vị trí y | 0.684 | 0.690 |
| `vision_footnote` → Caption | 0.681 | 0.697 |
| **Cả hai (đang dùng)** | **0.694** | **0.702** |

Page-footer F1 trên validation tăng 0.29 → 0.60 và Caption F1 0.01 → 0.29. (Kết luận cũ "80% box `number` là false positive" đo trên model `PP-DocLayout_plus-L`, không còn đúng với PP-DocLayoutV2.) Dùng `--no-page-number-mapping` để quay lại cách loại bỏ `number`. Prediction có trường `mapping_version` khác phiên bản hiện tại sẽ được chạy lại.

### 3. Nhãn loại bỏ (`DEFAULT_DROP_LABELS`)

Các nhãn sau được loại bỏ mặc định:
- **`seal`** (con dấu): Đặc thù tài liệu/hợp đồng tiếng Trung, không có trong taxonomy DocLayNet.
- **`formula_number`**: Nhãn đánh số phương trình (vd: `(1)`), trong DocLayNet được gộp chung vào box `Formula` hoặc `Text`. Tách riêng sinh box trùng lặp.
- **`inline_formula`**: Công thức nằm trong dòng chữ, thuộc khối `Text` trong DocLayNet.
- Người dùng có thể truyền flag `--keep-all` vào `run_inference.py` nếu muốn giữ lại toàn bộ nhãn với `category_id = None`.

### 4. Ánh xạ sang PubLayNet (5 lớp)

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
