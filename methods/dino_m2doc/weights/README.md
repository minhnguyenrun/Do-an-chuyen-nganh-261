# Trọng số Mô hình DINO + M2Doc (Standalone Method)

Thư mục này quản lý trọng số của mô hình **DINO + M2Doc** (công bố tại **AAAI 2024** bởi SCUT HCIILAB & Alibaba Group):

## 1. Thông tin Trọng số
- **Tên file**: `dino_m2doc_r50.pth`
- **Kích thước**: ~1.56 GB (1,559,355,140 bytes)
- **Kiến trúc**: DINO DETR với 4 scale feature maps, 900 queries, ResNet-50 backbone + Multilingual BERT Early & Late Multimodal Fusion.
- **Huấn luyện**: 12 epochs trên tập dữ liệu chuẩn **DocLayNet** (11 classes).
- **Hiệu năng công bố (Paper)**: **89.3% mAP** trên DocLayNet (vượt trội hơn Cascade Mask R-CNN M2Doc đạt 87.2% mAP).
- **Google Drive ID chính thức**: `11ZwmLLqz-SaehUIsuZg5incXOTgqeIwF`

## 2. Cơ chế Tái sử dụng Trọng số (Tối ưu Dung lượng)
Để tránh tốn thêm 1.56 GB bộ nhớ ổ đĩa:
- Pipeline tự động kiểm tra xem file `methods/method3/weights/dino_m2doc_r50.pth` đã tồn tại hay chưa.
- Nếu đã tồn tại ở `methods/method3/weights/dino_m2doc_r50.pth`, script `run_inference.py` và notebook `dino_m2doc_report.ipynb` sẽ tự động sử dụng trực tiếp trọng số này mà không cần tải lại hoặc copy trùng lặp.

## 3. Tải thủ công (nếu cần tải riêng vào thư mục này)
Nếu muốn tải riêng một bản về thư mục này:
```bash
python -m gdown 11ZwmLLqz-SaehUIsuZg5incXOTgqeIwF -O methods/dino_m2doc/weights/dino_m2doc_r50.pth
```
Hoặc tạo symbolic link / junction trên Windows:
```cmd
mklink methods\dino_m2doc\weights\dino_m2doc_r50.pth ..\..\method3\weights\dino_m2doc_r50.pth
```
