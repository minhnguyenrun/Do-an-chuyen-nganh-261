# Method 9: YOLOv26 Document Layout Analysis trên DocLayNet

## 1. Giới thiệu tổng quan
Method 9 thay thế detector RT-DETR truyền thống (trong Method 5/7/8) bằng **YOLOv26** — thế hệ phát hiện đối tượng End-to-End NMS-Free với các khối chú ý không gian C2PSA.

Trong Method 9, chúng ta đã tiến hành hai giai đoạn khảo sát:
1. **Giai đoạn 1 (YOLOv26x - COCO Pretrained)**: Khởi tạo từ trọng số COCO 80 lớp, kiểm tra khả năng hội tụ khi ngẫu nhiên hóa lại classification head sang 11 lớp DocLayNet.
2. **Giai đoạn 2 (YOLOv26 - DocLayNet Pretrained)**: Nạp trọng số đã được tiền huấn luyện trực tiếp trên toàn miền DocLayNet 11 lớp, tiến hành fine-tune trên tập con Pilot.

---

## 2. So Sánh Hai Giai Đoạn Thử Nghiệm

| Tiêu chí | Giai đoạn 1: YOLOv26x (COCO) | Giai đoạn 2: YOLOv26 (DocLayNet Pretrained) | Bước nhảy vọt |
| :--- | :---: | :---: | :---: |
| **Nguồn trọng số** | `yolo26x.pt` (MS COCO 80 classes) | `yolo26l-doclaynet.pt` (11 classes) | 100% khớp miền văn bản |
| **Trọng số chuyển giao** | 1080/1092 items (bị reset 12 tensor) | **1092/1092 items (100% giữ nguyên)** | Zero random reset |
| **Số tham số** | 58.83M params | **26.19M params** | Nhẹ hơn 55% |
| **Thời gian train** | 9.26 phút (5 epochs) | **3.36 phút (3 epochs)** | Nhanh hơn gần 3 lần |
| **Độ trễ suy luận** | 25.3 ms / ảnh (~40 FPS) | **14.6 ms / ảnh (~68.5 FPS)** | Nhanh hơn 73% |
| **mAP@50** | 40.93% | **`90.66%`** | **Tăng +49.73%** |
| **mAP@[.50:.95]** | 25.42% | **`78.91%`** | **Tăng +53.49%** |
| **mAP@75** | 25.75% | **`83.18%`** | **Tăng +57.43%** |

---

## 3. Hiệu năng chi tiết trên từng lớp văn bản (Giai đoạn 2)

| Lớp văn bản (Class) | mAP@[.50:.95] | mAP@50 | Recall | Precision | Đánh giá |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Text** | **91.11%** | 96.5% | 93.1% | 92.5% | Gần như hoàn hảo |
| **Title** | **89.23%** | 95.7% | 93.8% | 92.6% | Bắt trúng toàn bộ tiêu đề lớn |
| **Page-header** | **89.35%** | 98.8% | 93.3% | 97.6% | Định vị viền đầu trang chuẩn xác |
| **List-item** | **88.55%** | 94.1% | 93.7% | 92.4% | Phân biệt xuất sắc các dòng danh sách |
| **Section-header** | **82.45%** | 96.2% | 91.3% | 96.1% | Tách biệt chuẩn với text thường |
| **Table** | **80.84%** | 87.6% | 75.2% | 85.7% | Bao phủ trọn vẹn cấu trúc bảng |
| **Caption** | **79.97%** | 88.4% | 76.8% | 92.0% | Nhận diện tốt chú thích |
| **Picture** | **72.86%** | 85.7% | 75.8% | 82.1% | Tách biệt hình ảnh / đồ họa |
| **Formula** | **70.74%** | 90.4% | 91.1% | 80.8% | Nhận diện tốt các công thức toán |
| **Page-footer** | **65.04%** | 95.5% | 93.2% | 97.2% | Bao quát chân trang và số trang |
| **Footnote** | **57.84%** | 68.5% | 60.9% | 78.6% | Bắt được lớp văn bản nhỏ nhất |

---

## 4. Tài nguyên đã xuất bản trong Method 9
1. **Notebooks**:
   - [01_yolo26x_pilot_report.ipynb](notebooks/01_yolo26x_pilot_report.ipynb): Khảo sát baseline YOLOv26x (COCO).
   - [02_yolo26_doclaynet_pretrained_report.ipynb](notebooks/02_yolo26_doclaynet_pretrained_report.ipynb): Báo cáo đột phá của YOLOv26 Pretrained DocLayNet (**90.66% mAP50**).
   - [03_train_yolo26_sidecar_full_doclaynet.ipynb](notebooks/03_train_yolo26_sidecar_full_doclaynet.ipynb): Pipeline huấn luyện quy mô lớn toàn bộ DocLayNet với **ngữ cảnh liên trang 64 trang (Global Book Context - WINDOW_SIZE=64)** kết hợp YOLOv26 + Zero-Gated Multi-Scale Spatial Sidecar CNN.
2. **Checkpoints**:
   - `weights/yolo26_doclaynet_pretrained.pt`: Trọng số gốc từ DocLayNet (Hugging Face).
   - `weights/yolo26_doclaynet_best.pt`: Trọng số tốt nhất sau khi tinh chỉnh trên pipeline của đồ án.
   - `weights/yolo26_sidecar_pilot_test.pt`: Trọng số thử nghiệm kiến trúc Sidecar tích hợp YOLOv26.
3. **Báo cáo định lượng**:
   - `report/pilot_train_results_doclaynet.json`: Chi tiết số liệu baseline DocLayNet.
   - `report/sidecar_pilot_results.json`: Số liệu thực nghiệm tích hợp Sidecar CNN và sự dịch chuyển của các cổng Fusion Gates.
