# Trọng số Mô hình Method 3 (DINO + M2Doc & Book-Level Context)

Thư mục này lưu trữ các file trọng số phục vụ **Method 3**:

1. **Trọng số cơ sở DINO + M2Doc (ResNet-50, 12 epochs, 89.3 mAP)**:
   - **Tên file**: `dino_m2doc_r50.pth` (~1.56 GB)
   - **Config tương ứng**: `methods/m2doc/mmdetection/m2doc_config/dino-4scale_w_m2doc_r50_doclaynet_12epoch.py`
   - **Google Drive ID**: `11ZwmLLqz-SaehUIsuZg5incXOTgqeIwF`
   - **Lệnh tải bằng gdown**:
     ```bash
     python -m gdown 11ZwmLLqz-SaehUIsuZg5incXOTgqeIwF -O methods/method3/weights/dino_m2doc_r50.pth
     ```

2. **Trọng số Book-Level Context Encoder & Context Fusion**:
   - **Tên file**: `method3_book_context_best.pth`
   - Được sinh ra sau khi chạy script huấn luyện `methods/method3/src/train.py` hoặc cell huấn luyện trong notebook `methods/method3/notebooks/method3_pipeline.ipynb`.
