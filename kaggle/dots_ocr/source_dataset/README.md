# DO-AN-261 — So sánh các phương pháp Document Layout Analysis

Đồ án môn học: đánh giá và so sánh các phương pháp phân tích cấu trúc layout tài liệu
(Document Layout Analysis) trên các bộ dữ liệu chuẩn (DocLayNet, PubLayNet, DocBank).

## Cấu trúc project

```
DOAN261/
├── datasets/               # dữ liệu (chạy local, KHÔNG push, trừ README.md)
├── scripts/download_datasets/  # script tải dataset tự động
├── src/common/              # code dùng chung: io, label_mapping, metrics, vis_utils
├── methods/                 # mỗi phương pháp (docling, paddleocr_vl, ...) một folder con
├── outputs/                 # kết quả thô chạy local (KHÔNG push)
└── visualization/           # ghép PDF/ảnh + JSON thành hình trực quan để đưa vào báo cáo
```

Xem chi tiết nguồn và cách tải dữ liệu tại [datasets/README.md](datasets/README.md).

## Cách chạy

1. Tải dataset:

   ```bash
   python scripts/download_datasets/download_doclaynet.py --max-size-gb 1
   python scripts/download_datasets/download_publaynet.py --max-size-gb 1
   python scripts/download_datasets/download_docbank.py --max-size-gb 1
   ```

2. Chạy phương pháp tương ứng trong `methods/<method>/src/` hoặc notebook trong
   `methods/<method>/notebooks/`. Kết quả JSON dự đoán được lưu vào
   `outputs/predictions/<method>/<dataset>/`.

3. Dùng `visualization/scripts/` để ghép ảnh/PDF gốc với JSON dự đoán thành hình
   trực quan, lưu vào `visualization/figures/` để đưa vào báo cáo.

## Danh sách Phương pháp

| Phương pháp | Thư mục | Kiến trúc chính | Ghi chú |
|---|---|---|---|
| **Docling** | `methods/docling` | RT-DETR Layout Heron | Model tối ưu cho DocLayNet |
| **PaddleOCR-VL** | `methods/paddleocr_vl` | PP-DocLayout_plus-L | Layout Detection + VLM |
| **M2Doc** | `methods/m2doc` | Cascade Mask R-CNN + BERT | Multi-modal Early/Late Fusion (AAAI 2024) |
| **dots.ocr** | `methods/dots_ocr` | Qwen2-VL 1.7B VLM | RedNote HI Lab / Dots Studio |
| **dots.mocr** | `methods/dots_mocr` | 3B Multimodal Transformer | Dots Studio / RedNote HI Lab |
| **Chandra** | `methods/chandra` | Full-page Decoding VLM | Datalab OCR & Document Intelligence |

## Lưu ý quan trọng khi triển khai (Windows & Python 3.12)

1. **Quản lý file trọng số (Weights & Git)**:
   - Các file trọng số lớn (> 100MB) như `cascade_mask_rcnn_m2doc_r50.pth` (1.48 GB của M2Doc) hoặc các file `.safetensors` được lưu trong thư mục `weights/` của từng phương pháp và đã được cấu hình trong `.gitignore` để tránh commit lên Git.
   - Các file mã nguồn `.py` nằm trong thư mục `weights/` của `dots_ocr` và `dots_mocr` là kiến trúc mô hình tùy biến (custom modeling) được tải từ Hugging Face Hub, bắt buộc phải nằm cùng cấp với `config.json` để phục vụ cơ chế `trust_remote_code=True`.

2. **Khả năng tương thích trên môi trường Windows**:
   - **Xử lý đường dẫn tiếng Việt (OpenCV)**: Trên Windows, `cv2.imread()` không đọc được đường dẫn chứa ký tự tiếng Việt có dấu. Tất cả các phương pháp đã được chuyển sang dùng `cv2.imdecode(np.fromfile(...))` để đọc file an toàn.
   - **M2Doc**: Sử dụng `mmcv_shim.py` thay thế các toán tử C++ của MMDetection (`batched_nms`, `roi_align`) bằng `torchvision.ops` thuần PyTorch, không yêu cầu cài đặt C++ Build Tools. Bổ sung hỗ trợ trích xuất văn bản từ thư viện `RapidOCR 3.x+` (`RapidOCROutput`).
   - **dots.mocr**: Đã patch `modeling_dots_vision.py` để tự động fallback sang `torch.nn.functional.scaled_dot_product_attention` (`sdpa`) khi môi trường Windows không hỗ trợ cài đặt `flash_attn`.
   - **Chandra**: Sử dụng mô hình `datalab-to/chandra-ocr-2` (yêu cầu `transformers>=5.2.0`). Do Windows thiếu các native CUDA kernels của linear-attention (`causal_conv1d`, `flash-linear-attention`), mô hình sẽ fallback sang vòng lặp Python reference.

