# Phương pháp: Chandra (Document Intelligence & OCR Layout by Datalab)

Thư mục này chứa toàn bộ pipeline thực nghiệm mô hình **Chandra** phát triển bởi **Datalab** (`datalab-to`).

## 1. Giới thiệu tổng quan
`Chandra` là mô hình OCR và Document Understanding thế hệ mới với khả năng giải mã toàn trang (full-page decoding), bảo toàn chính xác cấu trúc không gian và bố cục văn bản, bảng biểu phức tạp (nested tables, colspans), công thức toán học và biểu đồ.

- **Hugging Face Model ID**:
  - [`datalab-to/chandra-ocr-2`](https://huggingface.co/datalab-to/chandra-ocr-2) (~5B parameters, phiên bản tối ưu tốc độ và đa ngữ)
  - [`datalab-to/chandra`](https://huggingface.co/datalab-to/chandra) (Phiên bản gốc 9B)
- **Tập nhãn hỗ trợ**: Tự động chuẩn hóa hơn 18 block types của Chandra về 11 lớp chuẩn của **DocLayNet**:
  `Caption`, `Footnote`, `Formula`, `List-item`, `Page-footer`, `Page-header`, `Picture`, `Section-header`, `Table`, `Text`, `Title`.
- **Định dạng tọa độ**: Tự động bóc tách bounding box từ cấu trúc JSON / HTML tags và chuẩn hóa sang format chuẩn `[x, y, w, h]`.

---

## 2. Cấu trúc thư mục

```
methods/chandra/
├── src/
│   ├── __init__.py
│   ├── parser.py          # Bộ phân tích khối cấu trúc, chuẩn hóa nhãn & bbox
│   └── run_inference.py   # CLI suy luận tự động tải trọng số
├── weights/               # Chứa trọng số (tự động tải, được ignore bởi git)
│   ├── .gitkeep
│   └── README.md
├── notebooks/
│   └── chandra_report.ipynb    # Notebook thực nghiệm & báo cáo trực quan
├── report/                # Báo cáo kết quả và biểu đồ metrics
│   └── .gitkeep
├── .gitignore             # Bỏ qua các file checkpoint > 100MB
├── requirements.txt       # Danh sách thư viện phụ thuộc
└── README.md              # Hướng dẫn chi tiết này
```

---

## 3. Cài đặt môi trường

Cài đặt các gói phụ thuộc:
```bash
pip install -r methods/chandra/requirements.txt
```

> **Lưu ý môi trường**: `datalab-to/chandra-ocr-2` có kiến trúc `Qwen3_5ForConditionalGeneration` nên cần **`transformers>=5.2`**, trong khi các phương pháp dots.ocr / dots.mocr dùng `transformers==4.49`. Chandra phải chạy trong một môi trường riêng (trên máy này: `D:\DLA\chandra_env`, kernel Jupyter `chandra_env`). Checkpoint 10,6 GB (bf16) không vừa GPU 8 GB nên chạy với `--load-in-4bit` (bitsandbytes NF4).

---

## 4. Chạy suy luận (Inference)

Mô hình sẽ tự động kiểm tra và tải trọng số từ Hugging Face Hub nếu máy tính chưa có:

### Chạy trên DocLayNet (20 ảnh mẫu):
```bash
python methods/chandra/src/run_inference.py --dataset doclaynet --limit 20
```

### Sử dụng mô hình Chandra v1 (9B):
```bash
python methods/chandra/src/run_inference.py --dataset doclaynet --model-id datalab-to/chandra --limit 20
```

### Chạy với GPU (CUDA):
```bash
python methods/chandra/src/run_inference.py --dataset doclaynet --device cuda --limit 20
```

Kết quả dự đoán sẽ được lưu tại:
`outputs/predictions/chandra/doclaynet/<image_id>.json`

---

## 5. Đánh giá và Trực quan hóa

Sau khi chạy suy luận, đánh giá độ chính xác (mAP, Precision, Recall, F1) bằng script chung:
```bash
python scripts/evaluate_predictions.py --method chandra --dataset doclaynet --allow-incomplete --loose --plot outputs/reports/chandra_doclaynet/metrics_chart.png
```

Trực quan hóa nhãn dự đoán kèm bounding box và ground truth:
```bash
python visualization/scripts/visualize_predictions.py --method chandra --dataset doclaynet --limit 20 --output-dir outputs/reports/chandra_doclaynet
```

Người dùng cũng có thể mở trực tiếp notebook:
`methods/chandra/notebooks/chandra_report.ipynb` và bấm **Run All**.
