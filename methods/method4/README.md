# Book-Layout RT-DETR: Cross-Page Layout Semantic Fusion

Hệ thống nhận diện cấu trúc tài liệu nhiều trang dựa trên **RT-DETRv2** kết hợp cơ chế **Layout Semantic Fusion** (mô phỏng theo nguyên lý Multimodal Fusion OCR).

## 1. Cấu trúc Thư mục

```text
methods/method_rtdetr_fusion/
├── models/
│   ├── book_layout_encoder.py   # Module rút trích & nén ngữ nghĩa bố cục sách (sinh E_book)
│   ├── rtdetr_fusion.py         # Cổng Cross-Attention Fusion hòa trộn E_book vào 300 Queries
│   └── rtdetr_network.py        # Kiến trúc hoàn chỉnh tích hợp RT-DETRv2 + Fusion + Residual Heads
├── data/
│   └── doclaynet_book_dataset.py# Dataloader gom nhóm trang theo từng cuốn sách DocLayNet
├── weights/
│   ├── book_rtdetr_best.pth     # Trọng số Best Checkpoint đạt kết quả cao nhất
│   └── book_rtdetr_latest.pth   # Checkpoint epoch mới nhất
├── report/
│   ├── training_history.json    # Lịch sử huấn luyện chi tiết từng epoch
│   ├── training_history.csv     # Bảng dữ liệu Loss, Val Accuracy, Learning Rate
│   ├── training_report.png      # Đồ thị trực quan hóa quá trình train/val
│   └── benchmark_comparison.json# Bảng đối chuẩn so sánh Base RT-DETR vs Book-Layout RT-DETR
├── train.py                     # Script huấn luyện với logging & checkpoint tracking
├── evaluate.py                  # Script đánh giá và so sánh định lượng trên từng class
└── notebooks/
    └── book_rtdetr_pipeline.ipynb # Notebook trình diễn trực quan kết quả
```

## 2. Cách Chạy Huấn luyện

```bash
# Huấn luyện mô hình:
py -3.12 -u methods/method_rtdetr_fusion/train.py --epochs 10 --batch-size 2 --lr 2e-4
```

## 3. Cách Đánh giá So sánh với Base Model

```bash
# Đánh giá đối chuẩn trên tập test:
py -3.12 -u methods/method_rtdetr_fusion/evaluate.py --limit-pages 100 --weights methods/method_rtdetr_fusion/weights/book_rtdetr_best.pth
```
