"""Script to generate and pre-execute book_rtdetr_report.ipynb.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT_DIR = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = ROOT_DIR / "methods" / "method_rtdetr_fusion" / "notebooks" / "book_rtdetr_report.ipynb"

nb = nbf.v4.new_notebook()

cells = []

# Cell 0: Header
cells.append(nbf.v4.new_markdown_cell("""# BÁO CÁO TOÀN DIỆN VÀ ĐỐI CHUẨN: BOOK-LAYOUT RT-DETR
## Phân Tích Lịch Sử Huấn Luyện 10 Epochs, Khôi Phục Trọng Số Cuối Cùng & Đối Chuẩn 3 Mô Hình

Báo cáo này trình bày kết quả nghiên cứu và thực nghiệm kiến trúc **Book-Layout RT-DETR (Cross-Page Layout Semantic Fusion)**:
1. **Quá trình huấn luyện quy mô lớn**: Huấn luyện toàn bộ 10,604 ảnh tập train DocLayNet qua 10 epochs (26,510 iterations) sử dụng kỹ thuật Automatic Mixed Precision (AMP FP16).
2. **Khôi phục trọng số Epoch 10**: Khắc phục thiếu sót trong cơ chế tính Validation Accuracy (bị kẹt ở Epoch 3), khôi phục thành công toàn bộ ma trận trọng số hội tụ của Epoch cuối cùng (`book_rtdetr_epoch10.pth`).
3. **Đối chuẩn định lượng 3 chiều (3-Way Benchmark)** theo chuẩn `src.common.metrics`:
   - **Base RT-DETRv2** (Single-page pre-trained Docling Heron).
   - **Book-Layout RT-DETR (Epoch 3)** (Checkpoint lưu nhầm do metric val_acc hit-rate).
   - **Book-Layout RT-DETR (Epoch 10)** (Trọng số cuối cùng sau khi hội tụ toàn diện).
4. **Trực quan hóa suy luận (Visual Inference)** trên các trang tài liệu thực tế của tập test DocLayNet.
"""))

# Cell 1: Environment Setup
cells.append(nbf.v4.new_code_cell("""import sys
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Thiết lập đường dẫn workspace
ROOT_DIR = Path.cwd().resolve()
while not (ROOT_DIR / "datasets").exists() and ROOT_DIR.parent != ROOT_DIR:
    ROOT_DIR = ROOT_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

METHOD_DIR = ROOT_DIR / "methods" / "method_rtdetr_fusion"
REPORT_DIR = METHOD_DIR / "report"
WEIGHTS_DIR = METHOD_DIR / "weights"

print(f"✓ Root Workspace: {ROOT_DIR}")
print(f"✓ Report Directory: {REPORT_DIR}")
print(f"✓ Weights Directory: {WEIGHTS_DIR}")
"""))

# Cell 2: Training History Markdown
cells.append(nbf.v4.new_markdown_cell("""## 1. Lịch Sử Huấn Luyện 10 Epochs (Full Dataset 10,604 Images)
Theo dõi tiến trình hội tụ qua toàn bộ 10 epochs. Mỗi epoch duyệt qua **2,651 batches (batch size = 4)**, với Learning Rate suy giảm theo hàm Cosine Annealing từ $1.5 \\times 10^{-4} \\to 1.83 \\times 10^{-5}$.
"""))

# Cell 3: Training History Table
cells.append(nbf.v4.new_code_cell("""csv_path = REPORT_DIR / "training_history.csv"
if csv_path.exists():
    df_history = pd.read_csv(csv_path)
    print("=== TIẾN TRÌNH HUẤN LUYỆN 10 EPOCHS HOÀN CHỈNH ===")
    display(df_history.style.highlight_min(subset=["train_loss", "val_loss", "train_cls_loss", "train_box_loss"], color="#d4edda")\\
                           .highlight_max(subset=["val_acc"], color="#cce5ff"))
else:
    print("⚠️ Không tìm thấy training_history.csv")
"""))

# Cell 4: Training Loss Curve Plot
cells.append(nbf.v4.new_code_cell("""if csv_path.exists():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss curve
    ax1.plot(df_history["epoch"], df_history["train_loss"], "o-", label="Train Total Loss", color="#1f77b4", linewidth=2)
    ax1.plot(df_history["epoch"], df_history["val_loss"], "s--", label="Val Total Loss", color="#ff7f0e", linewidth=2)
    ax1.plot(df_history["epoch"], df_history["train_cls_loss"], "^-.", label="Train Cls Loss", color="#2ca02c", alpha=0.8)
    ax1.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Loss", fontsize=12, fontweight="bold")
    ax1.set_title("Hội tụ Loss qua 10 Epochs", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=11)

    # Learning rate & Time
    ax2.plot(df_history["epoch"], df_history["lr"] * 1e4, "d-", label="LR (x 1e-4)", color="#d62728", linewidth=2)
    ax2_time = ax2.twinx()
    ax2_time.bar(df_history["epoch"], df_history["epoch_time_sec"] / 60, alpha=0.25, color="#9467bd", label="Epoch Time (min)")
    ax2.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Learning Rate", fontsize=12, fontweight="bold", color="#d62728")
    ax2_time.set_ylabel("Thời gian chạy (phút)", fontsize=12, fontweight="bold", color="#9467bd")
    ax2.set_title("Lịch trình Learning Rate & Thời gian Huấn luyện", fontsize=13, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()
"""))

# Cell 5: Missing Metric Analysis Markdown
cells.append(nbf.v4.new_markdown_cell("""## 2. Phân Tích Thiếu Sót Trong Metric Validation & Khôi Phục Trọng Số Epoch 10

### 🔍 Nguyên nhân sai lệch:
1. **Thiếu sót của `val_acc` lúc train:**
   - Trong quá trình train, `val_acc` được tính bằng tỷ lệ query có xác suất cao nhất trùng nhãn với nhãn Ground Truth (Classification Hit-rate đơn thuần), **hoàn toàn bỏ qua tọa độ Bounding Box và chỉ số IoU** để validation diễn ra tức thì.
   - Do đó, ở Epoch 3, chỉ số này đạt đỉnh ảo **97.75%**, khiến code `train.py` lầm tưởng đây là checkpoint tốt nhất và lưu vào `book_rtdetr_best.pth`.
   - Các epoch sau (Epoch 4 đến 10), mặc dù mô hình tiếp tục giảm loss từ **$7.16 \\to 6.21$**, nhưng do `val_acc` đạt ~96% nên checkpoint tốt nhất không được ghi đè, dẫn đến file `book_rtdetr_best.pth` bị kẹt ở Epoch 3!
2. **Khôi phục trọng số ma trận cuối cùng:**
   - May mắn là file `book_rtdetr_latest.pth` đã ghi nhận đầy đủ trọng số của **Epoch 10 hoàn chỉnh**.
   - Chúng ta đã tiến hành trích xuất, tách biệt và lưu trữ thành 2 checkpoint rõ ràng:
     - `weights/book_rtdetr_epoch3.pth`: Trọng số Epoch 3 (lưu theo val_acc).
     - `weights/book_rtdetr_epoch10.pth` (đồng thời cập nhật `book_rtdetr_best.pth`): Trọng số Epoch 10 hội tụ cuối cùng.
"""))

# Cell 6: Verify Checkpoints Code
cells.append(nbf.v4.new_code_cell("""import torch

print("=== KIỂM TRA CÁC CHECKPOINTS TRỌNG SỐ ===")
ckpt_files = {
    "Epoch 3 (Old Best)": WEIGHTS_DIR / "book_rtdetr_epoch3.pth",
    "Epoch 10 (Restored Final)": WEIGHTS_DIR / "book_rtdetr_epoch10.pth",
    "Active Best Checkpoint": WEIGHTS_DIR / "book_rtdetr_best.pth",
}

for name, path in ckpt_files.items():
    if path.exists():
        c = torch.load(path, map_location="cpu")
        ep = c.get("epoch", "N/A")
        val_l = c.get("val_loss", "N/A")
        val_a = c.get("val_acc", "N/A")
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  ✓ {name:<26}: Epoch {ep} | Val Loss: {val_l:.4f} | Val Acc: {val_a:.4f} | Dung lượng: {size_mb:.1f} MB")
    else:
        print(f"  ❌ {name}: File không tồn tại!")
"""))

# Cell 7: 3-Way Benchmark Markdown
cells.append(nbf.v4.new_markdown_cell("""## 3. Đối Chuẩn Định Lượng 3 Mô Hình (Base vs Epoch 3 vs Epoch 10)
Đánh giá nghiêm ngặt theo tiêu chuẩn của bài toán Document Layout Analysis ($IoU \\ge 0.5$ + Đúng nhãn class) trên 150 trang kiểm thử DocLayNet.
"""))

# Cell 8: 3-Way Benchmark Table Code
cells.append(nbf.v4.new_code_cell("""bench_3way_file = REPORT_DIR / "benchmark_comparison_3way.json"
if bench_3way_file.exists():
    with open(bench_3way_file, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    df_3way = pd.DataFrame(bench_data["classes"])
    
    # Format hiển thị phần trăm
    view_df = pd.DataFrame()
    view_df["Class"] = df_3way["class"]
    view_df["Base F1"] = (df_3way["base_f1"] * 100).round(2).astype(str) + "%"
    view_df["Epoch 3 F1"] = (df_3way["ep3_f1"] * 100).round(2).astype(str) + "%"
    view_df["Epoch 10 F1"] = (df_3way["ep10_f1"] * 100).round(2).astype(str) + "%"
    view_df["Δ F1 (Ep10 vs Base)"] = df_3way["delta_f1_pct"].apply(lambda x: f"+{x:.2f}%" if x > 0 else f"{x:.2f}%")
    
    # Hàng Overall
    overall = bench_data["overall"]
    b_mf1 = overall["base_micro"]["f1"] * 100
    e3_mf1 = overall["ep3_micro"]["f1"] * 100
    e10_mf1 = overall["ep10_micro"]["f1"] * 100
    delta_m = overall["delta_micro_f1_pct"]
    sign_m = "+" if delta_m > 0 else ""
    
    overall_row = pd.DataFrame([{
        "Class": "OVERALL (Micro)",
        "Base F1": f"{b_mf1:.2f}%",
        "Epoch 3 F1": f"{e3_mf1:.2f}%",
        "Epoch 10 F1": f"{e10_mf1:.2f}%",
        "Δ F1 (Ep10 vs Base)": f"{sign_m}{delta_m:.2f}%",
    }])
    
    view_df = pd.concat([view_df, overall_row], ignore_index=True)
    print("=== BẢNG SO SÁNH ĐỐI CHUẨN ĐỊNH LƯỢNG 3 MÔ HÌNH ===")
    display(view_df.style.set_properties(**{"text-align": "center"}))
else:
    print("⚠️ Không tìm thấy benchmark_comparison_3way.json")
"""))

# Cell 9: Benchmark Bar Chart Markdown
cells.append(nbf.v4.new_markdown_cell("""## 4. Biểu Đồ So Sánh F1-Score Trực Quan (Base vs Epoch 3 vs Epoch 10)
"""))

# Cell 10: Benchmark Bar Chart Code
cells.append(nbf.v4.new_code_cell("""if bench_3way_file.exists():
    active_mask = (df_3way["base_f1"] > 0) | (df_3way["ep3_f1"] > 0) | (df_3way["ep10_f1"] > 0)
    df_plot = df_3way[active_mask]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(df_plot))
    width = 0.25
    
    rects1 = ax.bar(x - width, df_plot["base_f1"] * 100, width, label="Base RT-DETR (Single Page)", color="#2b5c8f", alpha=0.9)
    rects2 = ax.bar(x, df_plot["ep3_f1"] * 100, width, label="Book-Layout RT-DETR (Epoch 3)", color="#f39c12", alpha=0.9)
    rects3 = ax.bar(x + width, df_plot["ep10_f1"] * 100, width, label="Book-Layout RT-DETR (Epoch 10 Restored)", color="#27ae60", alpha=0.9)
    
    ax.set_ylabel("F1-Score (%)", fontsize=12, fontweight="bold")
    ax.set_title("Đối Chuẩn F1-Score: Base vs Epoch 3 vs Epoch 10 trên Tập Kiểm Thử DocLayNet", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["class"], rotation=30, ha="right", fontsize=11)
    ax.legend(fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.show()
"""))

# Cell 11: Visual Inference Markdown
cells.append(nbf.v4.new_markdown_cell("""## 5. Trực Quan Hóa Dự Đoán Thực Tế (Visual Inference on Test Samples)
Mỗi mẫu ảnh bên dưới thể hiện 3 góc nhìn song song:
1. **Bên trái:** Ground Truth (Nhãn chuẩn của DocLayNet).
2. **Ở giữa:** Dự đoán của Checkpoint Epoch 3.
3. **Bên phải:** Dự đoán của Checkpoint Epoch 10 (Khôi phục ma trận cuối cùng).
"""))

# Cell 12: Visual Inference Code
cells.append(nbf.v4.new_code_cell("""vis_samples = sorted(list((REPORT_DIR / "vis_samples").glob("compare_sample_*.png")))

if vis_samples:
    for idx, s_path in enumerate(vis_samples):
        img = Image.open(s_path)
        plt.figure(figsize=(18, 7))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Visual Inference Mẫu #{idx+1} [Ground Truth (Trái) | Epoch 3 (Giữa) | Epoch 10 (Phải)]", fontsize=13, fontweight="bold", pad=10)
        plt.show()
else:
    print("⚠️ Chưa có ảnh visual inference trong report/vis_samples/")
"""))

# Cell 13: Scientific Conclusion Markdown
cells.append(nbf.v4.new_markdown_cell("""## 6. Đúc Kết Khoa Học & Định Hướng Cải Tiến Pipeline

### 📌 Những phát hiện cốt lõi từ thực nghiệm:
1. **Bản chất của DETR Queries & Hungarian Matching:**
   - Trong kiến trúc DETR / RT-DETR, các Object Queries không có thứ tự cố định tương ứng với các bounding boxes. 
   - Việc fine-tuning Detection Head cần kết nối chặt chẽ thông qua **Hungarian Matcher (Bipartite Matching Loss)** chuẩn của DETR thay vì Cross-Entropy tuyến tính đơn thuần trên các query đầu.
2. **Khôi phục thành công trọng số Epoch 10:**
   - Toàn bộ tham số của Epoch 10 đã được lưu trữ an toàn và khôi phục hoàn chỉnh tại `methods/method_rtdetr_fusion/weights/book_rtdetr_epoch10.pth`.
3. **Định hướng mở rộng Pipeline tiếp theo:**
   - **Tích hợp Hungarian Criterion cho Refined Head**: Đảm bảo các gradient cập nhật cho Cross-Attention Fusion Module đi qua thuật toán ghép cặp bipartite chính xác 100%.
   - **Mở rộng Context Window (16 - 64 trang)**: Với cơ chế nén $K=8$ summary tokens, việc mở rộng từ 4 trang lên 64 trang hoàn toàn không làm tăng chi phí tính toán của Decoder RT-DETR.
   - **Hòa trộn đa tầng OCR + Bố cục (Layout-OCR Multimodal Fusion)**: Kết hợp nhịp nhàng giữa phân bố không gian hình học và chuỗi ký tự OCR toàn cuốn sách.
"""))

nb.cells = cells

# Save unexecuted notebook first
with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✓ Đã tạo notebook cấu trúc tại: {NOTEBOOK_PATH}")

# Execute notebook using nbclient
print("⏳ Đang thực thi và pre-render toàn bộ code cells của notebook...")
client = NotebookClient(nb, timeout=300, kernel_name="python3")
client.execute()

# Save executed notebook with all outputs
with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"🎉 HOÀN TẤT! Đã thực thi và lưu output thành công tại: {NOTEBOOK_PATH}")
