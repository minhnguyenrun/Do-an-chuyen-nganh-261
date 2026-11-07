# Method 3: Hierarchical Book-Context Document Layout Analysis (Docling Scout + DINO-M2Doc)

Phương pháp **Method 3** kết hợp hai giai đoạn xử lý nhằm giải quyết hạn chế cố hữu của các mô hình phân tích bố cục tài liệu đơn trang (Single-Page DLA) khi áp dụng lên tài liệu dài hoặc sách nhiều trang:

1. **Pipeline 1 — Docling Layout Scout (Đóng băng / Frozen):**
   - Sử dụng mô hình `docling-layout-heron` (RT-DETRv2) quét nhanh qua toàn bộ các trang $P_1, P_2, \dots, P_N$ của cuốn sách.
   - Không cần huấn luyện lại (Zero Training Overhead), đóng vai trò trích xuất bản đồ bố cục sơ bộ $\mathcal{R} = \{R_1, R_2, \dots, R_N\}$ gồm tọa độ chuẩn hóa, nhãn dự đoán và độ tự tin.
2. **Cầu nối — Hierarchical Book-Level Context Encoder:**
   - Mã hóa từng vùng $r_{ij} = \operatorname{MLP}([f_{ij}; e_{ij}; p_{ij}; s_{ij}])$ và cộng thêm thông tin vị trí trang $z_{ij} = r_{ij} + e_{\text{page}}(i)$ (tích hợp tính chẵn/lẻ Recto-Verso và tỷ lệ vị trí trang).
   - Nén phân cấp từ cấp độ vùng (**Page Encoder**) sang cấp độ toàn sách (**Book Encoder**) để sinh ngữ cảnh liên trang $C_{\text{book}}$ mà không bị bùng nổ bộ nhớ bậc hai $\mathcal{O}((N \times M)^2)$.
3. **Pipeline 2 — DINO + M2Doc + Context Fusion (Mô hình tinh chỉnh chính):**
   - Sử dụng kiến trúc **DINO + M2Doc** (`DINO_w_M2Doc`, ResNet-50 + Multilingual BERT Early/Late Fusion, đạt **89.3% mAP** trên DocLayNet).
   - Tích hợp ngữ cảnh toàn sách $C_{\text{book}}$ thông qua **Cross-Attention + Learnable Residual ($\widetilde{F}_i = F_i + \alpha F_i'$)** (Cách A: Feature/Query-level Fusion) và **Context-Aware Box Refiner** (Cách B: Box-level Refinement).

---

## Cấu trúc thư mục

```text
methods/method3/
├── README.md                          # Tài liệu hướng dẫn chi tiết Method 3
├── requirements.txt                   # Các thư viện phụ thuộc
├── weights/
│   ├── README.md                      # Thông tin trọng số DINO + M2Doc và Book-Context
│   ├── dino_m2doc_r50.pth             # Trọng số pre-trained DINO + M2Doc ResNet-50 (1.56 GB)
│   └── method3_book_context_best.pth  # Trọng số Book-Level Context & Fusion sau khi huấn luyện
├── src/
│   ├── __init__.py
│   ├── mmcv_shim.py                   # Shim PyTorch thuần cho MultiScaleDeformableAttention & NMS/RoIAlign trên Windows
│   ├── pipeline1_docling.py           # Pipeline 1: Docling Layout Scout (Frozen) trích xuất đặc trưng từng trang
│   ├── book_context.py                # Cầu nối: HierarchicalBookEncoder, ContextFusionModule & ContextAwareBoxRefiner
│   ├── pipeline2_dino_m2doc.py        # Pipeline 2: DINO + M2Doc tích hợp Book-Level Context Fusion
│   ├── dataset.py                     # Dataset cửa sổ sách liên trang (BookContextWindowDataset)
│   ├── train.py                       # Script huấn luyện Book-Level Context & Context Fusion (tối ưu cho 8GB VRAM)
│   └── run_inference.py               # Script chạy End-to-End Inference xuất kết quả JSON chuẩn COCO
├── notebooks/
│   └── method3_pipeline.ipynb         # Jupyter Notebook trình diễn toàn bộ kiến trúc và thực nghiệm Method 3
└── report/                            # Thư mục lưu biểu đồ và kết quả đánh giá
```

---

## Trọng số Pre-trained (DINO + M2Doc)

Method 3 sử dụng trọng số chính thức của **DINO + M2Doc** huấn luyện trên **DocLayNet**:

| Mô hình | Config | Backbone | Epoch | mAP (COCO) | Nguồn tải |
|---|---|---|---|---|---|
| **DINO + M2Doc** | `dino-4scale_w_m2doc_r50_doclaynet_12epoch.py` | ResNet-50 | 12 | **89.3** | [Google Drive (1.56 GB)](https://drive.google.com/file/d/11ZwmLLqz-SaehUIsuZg5incXOTgqeIwF/view?usp=sharing) |
| **DINO + M2Doc** | `dino-4scale_w_m2doc_r101_doclaynet_36epoch.py` | ResNet-101 | 36 | **89.5** | [Google Drive](https://drive.google.com/file/d/1X8N1TYB1vsQ1O1skvbq9JDCP0Kk2LBkx/view?usp=sharing) |

> Trọng số mặc định đã được tải sẵn tại: `methods/method3/weights/dino_m2doc_r50.pth`.

---

## Hướng dẫn sử dụng

### 1. Huấn luyện module Book-Level Context & Context Fusion

Huấn luyện trên toàn bộ tập dữ liệu DocLayNet (10,604 ảnh, 328 sách/tài liệu, 2,673 cửa sổ liên trang):

```bash
# Huấn luyện 12 epochs trên toàn bộ DocLayNet (tự động cập nhật checkpoint tốt nhất)
python methods/method3/src/train.py --dataset doclaynet --epochs 12 --limit-pages 0 --window-size 8 --lr 2e-4

# Nếu muốn chạy thử nghiệm nhanh không lưu checkpoint
python methods/method3/src/train.py --dataset doclaynet_test --epochs 3 --limit-pages 40 --no-save
```

### 2. Chạy End-to-End Inference (Pipeline 1 -> Book Context -> Pipeline 2)

```bash
# Chạy với Cách A (Feature/Query-level Cross-Attention Fusion)
python methods/method3/src/run_inference.py --dataset doclaynet_test --fusion-mode feature --window-size 8 --limit 20

# Chạy với Cách B (Box-level Context Refinement)
python methods/method3/src/run_inference.py --dataset doclaynet_test --fusion-mode box --window-size 8 --limit 20
```

### 3. Đánh giá Metric & Trực quan hóa

```bash
# Đánh giá metric chuẩn COCO (Strict, Loose, mAP@[.5:.95], mAP@.5)
python scripts/evaluate_predictions.py --method method3 --dataset doclaynet_test --allow-incomplete --loose --plot

# Trực quan hóa đối chiếu Ground Truth vs Method 3 trên một vài trang tập Val/Test
python visualization/scripts/visualize_predictions.py --method method3 --dataset doclaynet_test --split test --limit 5 --output-dir methods/method3/report/val_visualizations
```
