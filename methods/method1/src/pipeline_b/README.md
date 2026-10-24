# Pipeline B: BBox-Level Multimodal Transformer

Pipeline B là module tiếp nhận tập hợp các ứng viên Bounding Box $[B_1, B_2, \dots, B_N]$ từ **Pipeline A (YOLOv8n-bbox-only)** và gán nhãn phân loại tài liệu (DocLayNet 11 classes) dựa trên 4 chiều thông tin đa phương thức.

---

## 1. Kiến trúc Tổng thể (Architecture Overview)

```
                            TRANG TÀI LIỆU (IMAGE)
                                       |
                   +-------------------+-------------------+
                   |                                       |
                   v                                       v
         [PIPELINE A: YOLO]                       [TRÍCH XUẤT NỘI DUNG]
         Phát hiện Bounding Boxes                 Text/OCR tokens trong BBox
                   |                                       |
                   +-------------------+-------------------+
                                       |
                         TẬP HỢP N BOUNDING BOXES: [B_1, B_2, ..., B_N]
                                       |
       +-------------------------------+-------------------------------+
       |                               |                               |
       v                               v                               v
[VISUAL CROP ENCODER]        [TEXT CONTENT ENCODER]         [2D SPATIAL ENCODER]
MobileNetV3-Small (128d)     MiniLM / MLP (128d)            MLP (x0,y0,x1,y1,w,h) (64d)
       |                               |                               |
       +-------------------------------+-------------------------------+
                                       |
                           (Concatenate & Linear Projection)
                                       v
                     Chuỗi Box Embeddings: E in R^{N x 256}
                                       |
                                       v
               +-----------------------------------------------+
               |     MULTI-HEAD SELF-ATTENTION TRANSFORMER     |
               |     (2-4 tầng Transformer Encoder, d=256)      |
               |                                               |
               | - Mỗi BBox "giao tiếp" với TẤT CẢ BBox khác    |
               | - Tự động học ngữ cảnh láng giềng (trên/dưới) |
               +-----------------------------------------------+
                                       |
                                       v
                       [LINEAR CLASSIFICATION HEAD]
                       11 Classes (DocLayNet Taxonomy)
```

---

## 2. Chi tiết 4 Nguồn Đặc trưng của từng BBox

| Đặc trưng | Đầu vào | Mạng trích xuất | Kích thước vector | Ý nghĩa |
| :--- | :--- | :--- | :---: | :--- |
| **1. Visual (Thị giác)** | Ảnh patch crop $I[y:y+h, x:x+w]$ | MobileNetV3-Small | $128$ | Nhận diện kết cấu hình ảnh (nét đậm, viền bảng, hình vẽ, ký hiệu công thức) |
| **2. Textual (Ngữ nghĩa)** | Chuỗi ký tự bên trong BBox | MiniLM / Text MLP | $128$ | Nhận diện ngữ nghĩa từ ngữ ("Table 1", "Introduction", "References", số liệu) |
| **3. Spatial (Hình học 2D)** | $[x_0, y_0, x_1, y_1, w, h, \text{aspect}, \text{area}]$ | 2-layer MLP + LayerNorm | $64$ | Tọa độ chuẩn hóa so với toàn trang (đầu trang, chân trang, lề trái, độ rộng) |
| **4. Relational Context** | Toàn bộ chuỗi $N$ boxes | Multi-Head Self-Attention | $256$ | **Cơ chế tương tác liên hộp**: Nhãn của một hộp được quyết định dựa trên các hộp lân cận |

---

## 3. Cấu trúc thư mục mã nguồn

```
methods/method1/src/pipeline_b/
├── __init__.py
├── README.md        # Tài liệu thiết kế chi tiết
├── model.py         # Kiến trúc mạng BBoxMultimodalTransformer
├── dataset.py       # Bộ nạp dữ liệu DocLayNetBBoxDataset và collate padding
├── pipeline.py      # Bộ ghép nối End-to-End: Pipeline A -> Pipeline B
└── train.py         # Script huấn luyện Pipeline B
```
