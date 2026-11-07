# dots.ocr: Multilingual Vision-Language Model for Document Layout Analysis

Phương pháp phân tích layout tài liệu bằng mô hình thị giác - ngôn ngữ **dots.ocr** (1.7B tham số, phát triển bởi nhóm nghiên cứu **RedNote HI Lab / Dots Studio**).

Mô hình kết hợp mạng thị giác độ phân giải cao (High-resolution Vision Encoder) và mô hình ngôn ngữ nền tảng Qwen2-VL 1.7B, cho phép xử lý văn bản đa ngôn ngữ, công thức toán học LaTeX, bảng biểu HTML và bóc tách cấu trúc layout tài liệu dạng bounding box `[x1, y1, x2, y2]` thông qua cơ chế prompt linh hoạt.

---

## Cấu trúc thư mục

```
methods/dots_ocr/
├── README.md                      # Tài liệu hướng dẫn phương pháp
├── requirements.txt               # Các thư viện phụ thuộc
├── .gitignore                     # Bỏ qua weights lớn (>100MB) và cache
├── weights/
│   ├── README.md                  # Hướng dẫn tải trọng số từ Hugging Face
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── parser.py                  # Module parse output JSON từ VLM, chuẩn hóa tọa độ pixel sang [x, y, w, h]
│   └── run_inference.py           # Script chạy inference xuất kết quả chuẩn JSON
├── notebooks/
│   └── dots_ocr_report.ipynb      # Notebook báo cáo: tải model, inference, đánh giá metric, trực quan hóa
└── report/
    └── .gitkeep                   # Thư mục lưu biểu đồ và báo cáo
```

---

## Thông tin Mô hình (Model Zoo)

| Mô hình | Số tham số | Kiểu dữ liệu | Dung lượng | Nguồn tải chính thức |
|---|:---:|:---:|:---:|---|
| **dots.ocr** | **1.7B** | BF16 / FP16 | **5.67 GB** | [Hugging Face: dots-studio/dots.ocr](https://huggingface.co/dots-studio/dots.ocr) |

> **Cơ chế tải tự động**: Khi chạy script `run_inference.py` hoặc notebook `dots_ocr_report.ipynb`, nếu chưa có file weights trong thư mục `weights/`, hệ thống sẽ tự động tải trực tiếp từ Hugging Face qua `huggingface_hub`.

---

## Bảng ánh xạ nhãn (Taxonomy Mapping)

`dots.ocr` hỗ trợ trực tiếp 11 danh mục của tập dữ liệu **DocLayNet** thông qua prompt `prompt_layout_only_en`:

| STT | Nhãn dots.ocr | DocLayNet Category ID | Mô tả thành phần |
|:---:|---|:---:|---|
| 1 | `Caption` | 1 | Chú thích hình ảnh/bảng biểu |
| 2 | `Footnote` | 2 | Ghi chú chân trang |
| 3 | `Formula` | 3 | Công thức toán học |
| 4 | `List-item` | 4 | Mục danh sách liệt kê |
| 5 | `Page-footer` | 5 | Chân trang (số trang, ngày tháng) |
| 6 | `Page-header` | 6 | Tiêu đề đầu trang |
| 7 | `Picture` | 7 | Hình ảnh, đồ thị, sơ đồ |
| 8 | `Section-header` | 8 | Tiêu đề mục / phân đoạn |
| 9 | `Table` | 9 | Bảng biểu dữ liệu |
| 10 | `Text` | 10 | Đoạn văn bản nội dung chính |
| 11 | `Title` | 11 | Tiêu đề tài liệu / báo cáo |

---

## Hướng dẫn cài đặt và sử dụng

### 1. Cài đặt môi trường

```bash
pip install -r methods/dots_ocr/requirements.txt
```

### 2. Chạy Inference

```bash
# Chạy inference trên 20 ảnh đầu tiên của tập doclaynet
python methods/dots_ocr/src/run_inference.py --dataset doclaynet --limit 20

# Bỏ qua các ảnh đã có kết quả (chạy tiếp tục)
python methods/dots_ocr/src/run_inference.py --dataset doclaynet --limit 0 --skip-existing

# Sử dụng GPU (nếu có CUDA)
python methods/dots_ocr/src/run_inference.py --dataset doclaynet --limit 20 --device cuda
```

Kết quả dự đoán được ghi vào: `outputs/predictions/dots_ocr/<dataset>/<image_id>.json` theo định dạng chuẩn:
```json
{
  "image_id": 0,
  "file_name": "0.png",
  "width": 1025,
  "height": 1025,
  "detections": [
    {
      "category_id": 10,
      "category_name": "Text",
      "bbox": [99.66, 118.11, 789.61, 42.79],
      "score": 0.95
    }
  ]
}
```

### 3. Đánh giá Metric

```bash
python scripts/evaluate_predictions.py --method dots_ocr --dataset doclaynet --allow-incomplete --loose --plot
```

### 4. Trực quan hóa đối chiếu Ground Truth vs Prediction

```bash
python visualization/scripts/visualize_predictions.py --method dots_ocr --dataset doclaynet --limit 20 --output-dir outputs/reports/dots_ocr_doclaynet
```
