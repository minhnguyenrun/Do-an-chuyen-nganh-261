# M2Doc: A Multi-Modal Fusion Approach for Document Layout Analysis

Phương pháp phân tích layout tài liệu đa phương thức **M2Doc** (công bố tại **AAAI 2024** bởi nhóm nghiên cứu SCUT HCIILAB & Alibaba Group).

M2Doc kết hợp đồng thời đặc trưng thị giác (visual features từ ảnh tài liệu) và đặc trưng văn bản ngữ nghĩa (textual features từ OCR/PDF cells) thông qua hai cơ chế fusion cốt lõi:
1. **Early-Fusion (Pixel-level)**: Mã hóa tọa độ và nội dung các từ/dòng văn bản thông qua mô hình ngôn ngữ (Multilingual BERT) thành biểu diễn lưới (text grid), sau đó căn chỉnh và hòa trộn trực tiếp vào các tầng đặc trưng đầu của mạng nơ-ron tích chập (ResNet Backbone).
2. **Late-Fusion (Block-level)**: Tại tầng RoI Head (Cascade Mask R-CNN), tính tương quan IoU giữa các vùng đề xuất (proposals) và các khối văn bản để chiếu đặc trưng ngữ nghĩa vào vector biểu diễn layout của từng block trước khi phân lớp và hồi quy bounding box.

---

## Cấu trúc thư mục

```
methods/m2doc/
├── README.md                      # Tài liệu hướng dẫn phương pháp
├── requirements.txt               # Các thư viện phụ thuộc
├── weights/
│   └── cascade_mask_rcnn_m2doc_r50.pth  # Trọng số pre-trained trên DocLayNet (12 epochs)
├── mmdetection/                   # Cấu trúc mã nguồn MMDetection tùy biến của M2Doc
├── src/
│   ├── __init__.py
│   ├── mmcv_shim.py               # Shim chuyển đổi toán tử C++ của mmcv.ops sang torchvision.ops (chạy thuần trên Windows/CPU)
│   ├── ocr_provider.py            # Trích xuất văn bản & bounding box từ DocLayNet parquet (pdf_cells) hoặc RapidOCR
│   └── run_inference.py           # Script chạy inference xuất kết quả JSON chuẩn COCO
├── notebooks/
│   └── m2doc_report.ipynb         # Notebook chạy pipeline hoàn chỉnh: inference, đánh giá metric, vẽ biểu đồ và trực quan hóa
└── report/                        # Thư mục lưu biểu đồ và báo cáo
```

---

## Trọng số Pre-trained (Model Zoo)

M2Doc cung cấp các trọng số chính thức huấn luyện trên tập **DocLayNet**:

| Mô hình | Backbone | Epoch | mAP (COCO) | Nguồn tải |
|---|---|---|---|---|
| **Cascade Mask R-CNN** | ResNet-50 | 12 | **84.6** | [Google Drive (1.48GB)](https://drive.google.com/file/d/14cf3cmVMWTG8g6w0qMd_lkKYfyR0EDX-/view?usp=sharing) |
| **Cascade Mask R-CNN** | ResNet-101| 36 | **85.9** | [Google Drive](https://drive.google.com/file/d/1VB7077QEsPYOczrPP7Q5OVmZs941wdHI/view?usp=sharing) |
| **DINO** | ResNet-50 | 12 | **89.3** | [Google Drive](https://drive.google.com/file/d/11ZwmLLqz-SaehUIsuZg5incXOTgqeIwF/view?usp=sharing) |
| **DINO** | ResNet-101| 36 | **89.5** | [Google Drive](https://drive.google.com/file/d/1X8N1TYB1vsQ1O1skvbq9JDCP0Kk2LBkx/view?usp=sharing) |

> Trọng số mặc định của phương pháp này là **Cascade Mask R-CNN ResNet-50** được lưu tại: `methods/m2doc/weights/cascade_mask_rcnn_m2doc_r50.pth`.

---

## Hướng dẫn cài đặt và sử dụng

### 1. Cài đặt môi trường

```bash
pip install -r methods/m2doc/requirements.txt
```

### 2. Chạy Inference

```bash
# Chạy inference trên 20 ảnh đầu tiên của tập doclaynet
python methods/m2doc/src/run_inference.py --dataset doclaynet --limit 20

# Bỏ qua các ảnh đã có kết quả (chạy tiếp tục)
python methods/m2doc/src/run_inference.py --dataset doclaynet --limit 0 --skip-existing

# Chạy inference thuần visual (tắt tính năng OCR text fusion)
python methods/m2doc/src/run_inference.py --dataset doclaynet --limit 20 --no-ocr
```

Kết quả dự đoán được ghi vào: `outputs/predictions/m2doc/<dataset>/<image_id>.json` theo chuẩn:
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
      "score": 0.9984
    }
  ]
}
```

### 3. Đánh giá Metric

```bash
# Đánh giá Strict, Loose và COCO mAP@[.5:.95], mAP@.5
python scripts/evaluate_predictions.py --method m2doc --dataset doclaynet --allow-incomplete --loose --plot
```

### 4. Trực quan hóa hình ảnh

```bash
# Xuất ảnh đối chiếu Ground Truth vs M2Doc Prediction cạnh nhau
python visualization/scripts/visualize_predictions.py --method m2doc --dataset doclaynet --limit 20 --output-dir outputs/reports/m2doc_doclaynet
```

---

## Bảng ánh xạ nhãn (Taxonomy Mapping)

Do M2Doc được huấn luyện trực tiếp trên tập dữ liệu DocLayNet chuẩn, 11 lớp đầu ra của mô hình khớp hoàn toàn 1-1 với 11 danh mục của DocLayNet:

| Chỉ số M2Doc | Tên nhãn | DocLayNet Category ID |
|:---:|---|:---:|
| 0 | Caption | 1 |
| 1 | Footnote | 2 |
| 2 | Formula | 3 |
| 3 | List-item | 4 |
| 4 | Page-footer | 5 |
| 5 | Page-header | 6 |
| 6 | Picture | 7 |
| 7 | Section-header | 8 |
| 8 | Table | 9 |
| 9 | Text | 10 |
| 10 | Title | 11 |

---

## Điểm nổi bật kỹ thuật và lưu ý khi triển khai

1. **Kho lưu trữ chính thức của mã nguồn M2Doc**:
   - Kho lưu trữ ban đầu `https://github.com/SCUT-HCIILab/M2Doc.git` đã đổi URL / không còn truy cập trực tiếp được (gây lỗi `exit status 128` khi git clone).
   - Mã nguồn chính thức hiện tại được duy trì tại: `https://github.com/johnning2333/M2Doc.git`. Script và notebook đã được cập nhật đường dẫn mới.

2. **Khả năng tương thích môi trường Windows & Python 3.12**:
   - MMDetection và MMCV nguyên bản yêu cầu biên dịch C++/CUDA extensions (`mmcv._ext`), vốn thường gặp lỗi thiếu compiler toolchain trên Windows.
   - Module `methods/m2doc/src/mmcv_shim.py` triển khai cơ chế dynamic module finding và shim các toán tử quan trọng (`batched_nms`, `nms`, `roi_align`, `RoIAlign`) bằng `torchvision.ops`, giúp M2Doc chạy ổn định trên CPU/GPU Windows mà không cần cài đặt Visual Studio C++ build tools.

3. **Xử lý đường dẫn Unicode tiếng Việt trên Windows**:
   - Khi thư mục chứa ký tự tiếng Việt có dấu (ví dụ: `Đồ án chuyên ngành`), hàm `cv2.imread()` của OpenCV trên Windows sẽ trả về `None`.
   - Toàn bộ pipeline M2Doc sử dụng `cv2.imdecode(np.fromfile(...))` để đọc file ảnh an toàn.

4. **Cơ chế Multi-Modal OCR Provider & RapidOCR**:
   - `methods/m2doc/src/ocr_provider.py` tự động đọc thông tin `pdf_cells` có sẵn trong tập dữ liệu parquet của DocLayNet (`datasets/doclaynet/raw/data/*.parquet`), cung cấp bounding box và nội dung từ vựng chuẩn xác cho mô hình BERT.
   - Khi chạy trên ảnh bên ngoài hoặc tài liệu quét tổng quát, module tự động chuyển sang cơ chế fallback qua thư viện `RapidOCR` để bóc tách từ ngữ on-the-fly. Hỗ trợ tương thích cả chuẩn `RapidOCROutput` (bản mới) và `tuple` (bản cũ).

5. **Tự động chọn phần cứng tăng tốc (CUDA)**:
   - Script `methods/m2doc/src/run_inference.py` mặc định kiểm tra và sử dụng `cuda` nếu phát hiện GPU rời, giúp tốc độ xử lý đạt khoảng ~15-25s/ảnh.
