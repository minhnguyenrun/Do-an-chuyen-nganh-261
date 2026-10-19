# Docling

Phương pháp phân tích layout tài liệu dùng [Docling](https://github.com/docling-project/docling).

## Cấu trúc

- `src/` — code chạy inference, xuất dự đoán ra `outputs/predictions/docling/<dataset>/*.json`
- `notebooks/docling_report.ipynb` — chạy inference, xem metric và hiển thị Ground Truth so với prediction
- `report/` — báo cáo PDF cho phương pháp này

## Cách chạy

```bash
pip install -r methods/docling/requirements.txt

# chay layout detection tren N anh dau tien cua datasets/doclaynet/processed/images
python methods/docling/src/run_inference.py --dataset doclaynet --limit 765

# ve hinh so sanh Ground Truth vs. du doan
python visualization/scripts/visualize_predictions.py --method docling --dataset doclaynet --limit 20 --output-dir outputs/reports/docling_doclaynet
```

Kết quả JSON lưu tại `outputs/predictions/docling/doclaynet/<image_id>.json`
(mỗi ảnh một file, gồm `detections`: list `{category_id, category_name, bbox, score}`
và `inference_time_s`: thời gian xử lý trang đó, không tính thời gian load model).
Thống kê thời gian/trang, peak RAM/VRAM của mỗi lần chạy được ghi thêm vào
`outputs/runtime/docling/doclaynet.json`; `scripts/evaluate_predictions.py` in
các số này ở mục `=== Runtime ===`. Với `--skip-existing`, file dự đoán định dạng
cũ (không có `inference_time_s`) sẽ được chạy lại.
Hình so sánh lưu tại `outputs/reports/docling_doclaynet/<image_id>.png`.

Có thể mở và chạy lần lượt các cell trong
[`notebooks/docling_report.ipynb`](notebooks/docling_report.ipynb) để kết hợp
inference, đánh giá strict/loose và trực quan hóa trong một report. Notebook
mặc định chạy inference trên toàn bộ ảnh có trong ground truth local (1,928 ảnh
hiện tại); nếu lần chạy bị gián đoạn, `--skip-existing` cho phép tiếp tục. Để
report gọn, notebook chỉ tạo hình minh họa cho 20 ảnh.

## Ghi chú kỹ thuật

- Docling dùng model layout mặc định `docling-project/docling-layout-heron`
  (tải tự động từ HuggingFace Hub lần chạy đầu). Model này được huấn luyện
  trên chính DocLayNet nên nhãn đầu ra (`DocItemLabel`) khớp trực tiếp với
  11 nhãn của DocLayNet (xem mapping trong `src/run_inference.py`).
- Script chỉ bật stage layout, tắt OCR/table-structure/các enrichment khác
  để chạy nhanh (~1-1.5s/ảnh trên CPU sau khi model đã load).
- Vì tắt OCR nên các cluster text "rỗng" (không có nội dung) sẽ bị Docling
  loại bỏ theo mặc định — phải set `layout_options.keep_empty_clusters =
  True` để giữ lại đầy đủ box layout.
- Box được lấy từ `ConversionResult.pages[0].assembled.elements` (mỗi element
  giữ `cluster` gốc của model, có `label`, `bbox`, `confidence`), **không** lấy
  từ `document.iterate_items()`: hàm này mặc định bỏ qua content layer
  "furniture", tức là làm mất toàn bộ `page_header`/`page_footer`.

## Kết quả đánh giá (765 ảnh đầu, doclaynet/train, IoU >= 0.5)

Chạy bằng `python scripts/evaluate_predictions.py --method docling --dataset doclaynet --loose`
(763 ảnh có Ground Truth được dùng để đánh giá).

### Precision / Recall / F1 (micro-average, IoU >= 0.5)

| | Precision | Recall | F1 |
|---|---|---|---|
| Overall (so khớp cả vị trí lẫn nhãn) | 0.888 | 0.882 | 0.885 |
| Overall (chỉ so vị trí, bỏ qua nhãn) | 0.928 | 0.923 | 0.925 |

Mean IoU của các box khớp: 0.931.

### mAP kiểu COCO (pycocotools, maxDets=100)

| mAP@[.5:.95] | mAP@.5 | mAP@.75 | AR@100 |
|---|---|---|---|
| 0.572 | 0.684 | 0.621 | 0.621 |

| Nhãn | AP | AP50 | | Nhãn | AP | AP50 |
|---|---|---|---|---|---|---|
| Table | 0.882 | 0.926 | | Page-footer | 0.602 | 0.894 |
| Text | 0.851 | 0.904 | | Page-header | 0.541 | 0.781 |
| Picture | 0.763 | 0.841 | | Caption | 0.453 | 0.540 |
| List-item | 0.670 | 0.740 | | Footnote | 0.304 | 0.380 |
| Section-header | 0.652 | 0.834 | | Title | 0.000 | 0.000 |

(`Formula` không có trong 763 ảnh này nên không có AP.)

Lưu ý khi đọc mAP:
- mAP là **trung bình đều giữa các nhãn** (macro), nên nhãn hiếm và yếu (Title,
  Footnote) kéo điểm xuống mạnh; còn P/R/F1 ở trên là micro, bị chi phối bởi `Text`.
  Vì vậy mAP@.5 (0.684) thấp hơn F1 (0.885) dù cùng ngưỡng IoU.
- Docling chỉ trả về box có confidence >= `score_threshold` (mặc định 0.3), nên
  đường Precision-Recall bị cắt ở vùng recall cao. mAP ở đây có thể **thấp hơn**
  so với khi đánh giá model với toàn bộ box dự đoán như trong paper.

### Nhận xét theo nhãn

- `Table`, `Text`, `Picture` tốt nhất (F1 0.85-0.94, AP 0.76-0.88).
- `Section-header` precision thấp hơn recall: Docling tách tên người/chức danh
  thành nhiều `section_header` nhỏ trong khi Ground Truth gộp vào 1 khối `Text`.
- `List-item` precision cao (0.96) nhưng recall thấp (0.74); khi bỏ qua nhãn thì
  recall lên 0.875 → một phần list-item bị gán nhầm thành `text`.
- `Title` = 0 khi so cả nhãn nhưng khớp 18/21 khi bỏ qua nhãn: Docling phát hiện
  đúng vị trí nhưng gán thành `section_header` (11), `text` (6), `page_header` (1).
- `Page-header` / `Page-footer` đạt F1 0.84 / 0.92. Phiên bản trước lấy box từ
  `document.iterate_items()` nên mất toàn bộ hai nhãn này (recall = 0), và từng
  bị diễn giải nhầm là do model RT-DETR không phát hiện được dải mỏng ở mép ảnh.
  Thực tế model vẫn phát hiện được; box chỉ bị bỏ ở bước dựng `DoclingDocument`.
