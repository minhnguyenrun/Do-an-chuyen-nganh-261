# Phương pháp: dots.mocr (3B Multimodal VLM Document Parser)

Thư mục này chứa toàn bộ pipeline thực nghiệm mô hình **dots.mocr** phát triển bởi **Dots Studio / RedNote HI Lab**.

## 1. Giới thiệu tổng quan
`dots.mocr` là mô hình thị giác - ngôn ngữ đa phương thức (Vision-Language Model - 3B parameters) tối ưu hóa cho nhận diện tài liệu, bóc tách cấu trúc bố cục văn bản, biểu đồ (chart), công thức toán học và chuyển đổi đồ họa sang SVG/Markdown/JSON.

- **Hugging Face Model ID**: [`dots-studio/dots.mocr`](https://huggingface.co/dots-studio/dots.mocr)
- **Kiến trúc**: 3B Multimodal Transformer (Qwen2-VL backbone tối ưu hóa)
- **Tập nhãn hỗ trợ trực tiếp**: 11 lớp chuẩn của **DocLayNet**:
  `Caption`, `Footnote`, `Formula`, `List-item`, `Page-footer`, `Page-header`, `Picture`, `Section-header`, `Table`, `Text`, `Title`.
- **Định dạng tọa độ**: Tự động chuẩn hóa từ pixel `[x1, y1, x2, y2]` sang format chuẩn của benchmark `[x, y, w, h]`.

### Cấu hình suy luận (khớp cấu hình chính thức của dots.mocr)

- **Prompt**: `prompt_layout_only_en` (chỉ xuất bbox + nhãn, không xuất nội dung), danh sách nhãn là 11 lớp DocLayNet.
- **Độ phân giải**: dùng mặc định của processor chính thức (`min_pixels=3136`, `max_pixels=11289600`, bội số 28), tức là ảnh DocLayNet 1025×1025 được đưa vào ở 1036×1036 mà không bị thu nhỏ. Tọa độ model sinh ra nằm trong không gian ảnh đã resize này và được quy đổi về ảnh gốc trong `parser.py`.
- **EOS**: `[151643, 151645, 151672, 151673]` (hợp của `generation_config.json` của checkpoint và `<|im_end|>`), `max_new_tokens=2048` để trang dày (nhiều box) không bị cắt cụt.
- **Batch size mặc định = 1**: trên GPU 8 GB, batch 4 chậm hơn ~3,5 lần (33,9s so với 9,6s/trang) do tràn VRAM; tăng `--batch-size` nếu dùng GPU lớn hơn.
- Checkpoint `dots-studio/dots.mocr` dùng kiến trúc `DotsOCRForCausalLM` (`model_type: dots_ocr`), nên kiểm tra trọng số dựa trên `model_type` này.
- Mô hình không trả về confidence nên mọi box có `score = 0.95` (mAP kiểu COCO không có ý nghĩa với phương pháp này).

---

## 2. Cấu trúc thư mục

```
methods/dots_mocr/
├── src/
│   ├── __init__.py
│   ├── parser.py          # Bộ phân tích JSON, chuẩn hóa nhãn & bounding box
│   └── run_inference.py   # CLI suy luận tự động tải trọng số
├── weights/               # Chứa trọng số (tự động tải, được ignore bởi git)
│   ├── .gitkeep
│   └── README.md
├── notebooks/
│   └── dots_mocr_report.ipynb  # Notebook thực nghiệm & báo cáo trực quan
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
pip install -r methods/dots_mocr/requirements.txt
```

---

## 4. Chạy suy luận (Inference)

Mô hình sẽ tự động kiểm tra và tải trọng số từ Hugging Face Hub nếu máy tính chưa có:

### Chạy trên DocLayNet (20 ảnh mẫu):
```bash
python methods/dots_mocr/src/run_inference.py --dataset doclaynet --limit 20
```

### Chạy với GPU (CUDA):
```bash
python methods/dots_mocr/src/run_inference.py --dataset doclaynet --device cuda --limit 20
```

### Tùy chỉnh danh sách ảnh cụ thể:
```bash
python methods/dots_mocr/src/run_inference.py --dataset doclaynet --image-ids 0a3a4532cdd82eb171a3c3d345f01b66b08a7443dc252d7e8aede7d3493629ef,0a4aaef44d29e23274842991f40e1e75c9d20820759cf7953a60f5bbe7cfe8f6
```

Kết quả dự đoán sẽ được lưu tại:
`outputs/predictions/dots_mocr/doclaynet/<image_id>.json`

---

## 5. Đánh giá và Trực quan hóa

Sau khi chạy suy luận, đánh giá độ chính xác (mAP, Precision, Recall, F1) bằng script chung:
```bash
python scripts/evaluate_predictions.py --method dots_mocr --dataset doclaynet --allow-incomplete --loose --plot outputs/reports/dots_mocr_doclaynet/metrics_chart.png
```

Trực quan hóa nhãn dự đoán kèm bounding box và ground truth:
```bash
python visualization/scripts/visualize_predictions.py --method dots_mocr --dataset doclaynet --limit 20 --output-dir outputs/reports/dots_mocr_doclaynet
```

Người dùng cũng có thể mở trực tiếp notebook:
`methods/dots_mocr/notebooks/dots_mocr_report.ipynb` và bấm **Run All**.
