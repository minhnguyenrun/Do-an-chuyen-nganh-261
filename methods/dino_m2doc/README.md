# DINO + M2Doc: Multi-Modal Document Layout Analysis

Thư mục này triển khai và đánh giá phương pháp **DINO + M2Doc** nguyên bản (công bố tại **AAAI 2024** bởi SCUT HCIILAB & Alibaba Group), đóng vai trò là mô hình cơ sở độc lập (standalone baseline) trước khi tích hợp vào **Method 3 (ContextFusion Pipeline)**.

---

## 1. Giới thiệu & Điểm nổi bật

M2Doc giải quyết bài toán phân tích bố cục tài liệu (Document Layout Analysis - DLA) phức tạp thông qua cơ chế **Early & Late Multi-modal Fusion**:
- **Early-Fusion (Pixel-level)**: Sử dụng Multilingual BERT (`bert-base-multilingual-cased`) để nhúng các cell văn bản và bounding box thành biểu diễn lưới (`chargrid_map` 64 chiều), hòa trộn vào các tầng đặc trưng sớm của backbone ResNet-50.
- **Late-Fusion (Block-level)**: Khai thác ma trận tương quan vị trí giữa các bounding box đề xuất và khối văn bản ngữ nghĩa để tinh chỉnh đặc trưng từng vùng trước khi dự đoán.
- **DINO DETR Backbone**: So với biến thể Cascade Mask R-CNN, biến thể **DINO-4scale w/ M2Doc** sử dụng cơ chế Deformable Attention 4 scale với 900 queries, kết hợp DeNoising anchor boxes và Contrastive Denoising Training.
- **Hiệu năng SOTA**: Đạt **89.3% mAP** trên tập dữ liệu chuẩn **DocLayNet** (11 classes), vượt trội so với phiên bản Cascade Mask R-CNN (87.2% mAP).

---

## 2. Cấu trúc Thư mục

```text
methods/dino_m2doc/
├── README.md                     # Tài liệu hướng dẫn chi tiết
├── requirements.txt              # Danh sách thư viện phụ thuộc
├── weights/
│   ├── README.md                 # Hướng dẫn trọng số & cơ chế tái sử dụng từ Method 3
│   └── dino_m2doc_r50.pth        # Checkpoint ResNet-50 12 epochs (~1.56 GB)
├── src/
│   ├── __init__.py
│   ├── mmcv_shim.py              # Pure PyTorch fallback cho Deformable Attention & NMS trên Windows
│   ├── ocr_provider.py           # Trích xuất OCR từ DocLayNet parquet pdf_cells hoặc RapidOCR
│   └── run_inference.py          # Script suy luận sinh dự đoán JSON chuẩn COCO
├── notebooks/
│   └── dino_m2doc_report.ipynb   # Báo cáo đánh giá hoàn chỉnh (Strict/Loose, COCO mAP, Confusion Matrix, So sánh)
└── report/                       # Nơi lưu biểu đồ và báo cáo xuất ra
```

---

## 3. Cài đặt Môi trường

```bash
pip install -r methods/dino_m2doc/requirements.txt
```

> **Lưu ý Windows / Python 3.12**: Module `src/mmcv_shim.py` đã cung cấp sẵn bản thuần PyTorch của `MultiScaleDeformableAttention` (thông qua `F.grid_sample`) và `batched_nms` (thông qua `torchvision.ops`), cho phép chạy mượt mà trên Windows mà **không cần cài đặt MSVC C++ Compiler** hoặc compile `mmcv._ext`.

---

## 4. Trọng số Mô hình (Pre-trained Weights)

- **Tên file**: `dino_m2doc_r50.pth` (~1.56 GB)
- **Google Drive ID**: `11ZwmLLqz-SaehUIsuZg5incXOTgqeIwF`
- **Tối ưu dung lượng**: Nếu file trọng số đã có sẵn tại `methods/method3/weights/dino_m2doc_r50.pth`, pipeline sẽ tự động nhận diện và tái sử dụng, không tải lại hoặc sao chép trùng lặp.
- **Tải thủ công bằng gdown**:
  ```bash
  python -m gdown 11ZwmLLqz-SaehUIsuZg5incXOTgqeIwF -O methods/dino_m2doc/weights/dino_m2doc_r50.pth
  ```

---

## 5. Hướng dẫn Chạy Suy luận (Inference)

Chạy trên tập test DocLayNet:
```bash
python methods/dino_m2doc/src/run_inference.py --dataset doclaynet_test --split test --skip-existing
```

Các tham số tùy chọn:
- `--limit N`: Giới hạn số trang xử lý (ví dụ: `--limit 20` để test nhanh).
- `--device [cuda|cpu]`: Chọn GPU hoặc CPU (mặc định tự động nhận GPU CUDA).
- `--threshold 0.3`: Ngưỡng confidence score lọc box (mặc định: 0.3).
- `--force`: Ghi đè file dự đoán đã có sẵn.
- `--no-ocr`: Tắt module OCR (dùng dummy boxes).

Kết quả dự đoán được ghi vào: `outputs/predictions/dino_m2doc/<dataset>/<image_id>.json`.

---

## 6. Đánh giá & Báo cáo (Evaluation)

### 6.1 Đánh giá bằng Script
```bash
python scripts/evaluate_predictions.py \
    --method dino_m2doc \
    --dataset doclaynet_test \
    --split test \
    --loose \
    --plot outputs/reports/dino_m2doc_doclaynet_test/metrics_chart.png
```

### 6.2 Đánh giá qua Jupyter Notebook
Mở notebook:
```text
methods/dino_m2doc/notebooks/dino_m2doc_report.ipynb
```
Notebook cung cấp:
1. Setup kiểm tra môi trường, GPU, cấu hình và tự động liên kết trọng số.
2. Thực thi suy luận (hỗ trợ `skip_existing` hoặc `force_rerun`).
3. Bảng metric chi tiết 11 lớp DocLayNet theo cả 2 chế độ:
   - **Strict Metric**: Khớp cả vị trí ($\text{IoU} \ge 0.5$) và đúng nhãn layout.
   - **Loose Metric**: Đánh giá khả năng định vị khối tài liệu (bỏ qua nhãn).
   - **COCO mAP**: mAP@[0.5:0.95] và mAP@0.5 theo chuẩn COCO.
4. Trực quan hóa đối chiếu ảnh Ground Truth và dự đoán của DINO + M2Doc.
5. Bảng tổng hợp so sánh toàn diện với các phương pháp khác trên DocLayNet (Docling, PaddleOCR-VL, M2Doc Cascade Mask R-CNN, DINO_w_M2Doc, dots.ocr, dots.mocr, Method 3).
