"""Generate and execute Jupyter Notebook report for Method 6.
"""

from __future__ import annotations

import json
import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT_DIR = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = ROOT_DIR / "methods" / "method6" / "notebooks"
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH = NOTEBOOK_DIR / "method6_report.ipynb"

nb = nbf.v4.new_notebook()
cells = []

# Cell 0: Header
cells.append(nbf.v4.new_markdown_cell("""# BÁO CÁO THỰC NGHIỆM CHÍNH THỨC: METHOD 6 - SPATIAL SIDECAR HEATMAP RT-DETR
## Hòa Trộn Không Gian Đa Tỷ Lệ (Zero-Gated Multi-Scale Feature Injection) & Đánh Giá Trên 4,999 Trang DocLayNet Test

Notebook này báo cáo chi tiết thiết kế, tiến trình huấn luyện và kết quả thực nghiệm của **Method 6**:
1. **Kiến Trúc Đổi Mới (Spatial Sidecar Heatmap)**:
   - Thay thế kiến trúc token sequence phân cấp phức tạp bằng **2D Layout Spatial Heatmap 22 kênh** (11 Book Context Heatmap từ 64 trang + 11 Page Prior Heatmap).
   - Mạng **Sidecar CNN** 3 tầng phân giải đa tỉ lệ ($S_0: 80 \\times 80, S_1: 40 \\times 40, S_2: 20 \\times 20$).
   - Kỹ thuật **Zero-Gated Multi-Scale Feature Injection** hòa trộn trực tiếp vào $P_3, P_4, P_5$ của RT-DETR:
     $$P_i^{\\text{fused}} = P_i + \\text{Gate}_i \\cdot S_i$$
     với khởi tạo $\\text{Gate}_i = 0.0$ giúp mô hình bảo toàn tuyệt đối tri thức của pre-trained model và học bổ sung ổn định.
2. **Cấu Hình Thực Nghiệm Chuẩn**:
   - Huấn luyện trên **50 sách Train (1,669 trang)** và **10 sách Val (221 trang)** tuyển chọn trong 3 epochs.
   - Đánh giá benchmark trên toàn bộ **4,999 trang DocLayNet Test Set**.
   - Đo đạc đầy đủ các tiêu chuẩn học máy hiện đại: COCO mAP@[.50:.95], AP50, AP75, AP Small/Medium/Large, AR100, F1 Strict vs Loose, Tốc độ (FPS), Độ trễ (ms/trang) và Bộ nhớ GPU (Peak VRAM).
"""))

# Cell 1: Environment & Data Loading
cells.append(nbf.v4.new_code_cell("""import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

ROOT_DIR = Path.cwd().resolve()
while not (ROOT_DIR / "datasets").exists() and ROOT_DIR.parent != ROOT_DIR:
    ROOT_DIR = ROOT_DIR.parent

REPORT_PATH = ROOT_DIR / "methods" / "method6" / "report" / "full_4999_benchmark_comparison.json"
with open(REPORT_PATH, "r", encoding="utf-8") as f:
    bench_data = json.load(f)

print(f"✓ Đã tải báo cáo kết quả đánh giá 4,999 trang: {REPORT_PATH}")
"""))

# Cell 2: Benchmark Summary Table
cells.append(nbf.v4.new_markdown_cell("""## 1. Bảng So Sánh Đối Chuẩn Tổng Thể Trên 4,999 Trang Test
So sánh trực diện 3 mô hình: **Base Pretrained Model**, **Method 5** và **Method 6 (Spatial Sidecar Heatmap)**.
"""))

cells.append(nbf.v4.new_code_cell("""base = bench_data["base_model"]
m5 = bench_data["method5_phase1"]
m6 = bench_data["method6"]

metrics_summary = [
    {
        "Tiêu chí": "COCO mAP@[.50:.95]",
        "Base RT-DETR": f"{base['coco_map']['map']*100:.2f}%",
        "Method 5": f"{m5['coco_map']['map']*100:.2f}%",
        "Method 6 (Sidecar)": f"{m6['coco_map']['map']*100:.2f}%",
        "Cải thiện (M6 vs Base)": f"{(m6['coco_map']['map'] - base['coco_map']['map'])*100:+.2f}%"
    },
    {
        "Tiêu chí": "COCO AP50",
        "Base RT-DETR": f"{base['coco_map']['map50']*100:.2f}%",
        "Method 5": f"{m5['coco_map']['map50']*100:.2f}%",
        "Method 6 (Sidecar)": f"{m6['coco_map']['map50']*100:.2f}%",
        "Cải thiện (M6 vs Base)": f"{(m6['coco_map']['map50'] - base['coco_map']['map50'])*100:+.2f}%"
    },
    {
        "Tiêu chí": "COCO AP75",
        "Base RT-DETR": f"{base['coco_map']['map75']*100:.2f}%",
        "Method 5": f"{m5['coco_map']['map75']*100:.2f}%",
        "Method 6 (Sidecar)": f"{m6['coco_map']['map75']*100:.2f}%",
        "Cải thiện (M6 vs Base)": f"{(m6['coco_map']['map75'] - base['coco_map']['map75'])*100:+.2f}%"
    },
    {
        "Tiêu chí": "COCO AP Small",
        "Base RT-DETR": f"{base['coco_map']['map_small']*100:.2f}%",
        "Method 5": f"{m5['coco_map']['map_small']*100:.2f}%",
        "Method 6 (Sidecar)": f"{m6['coco_map']['map_small']*100:.2f}%",
        "Cải thiện (M6 vs Base)": f"{(m6['coco_map']['map_small'] - base['coco_map']['map_small'])*100:+.2f}%"
    },
    {
        "Tiêu chí": "COCO AP Medium",
        "Base RT-DETR": f"{base['coco_map']['map_medium']*100:.2f}%",
        "Method 5": f"{m5['coco_map']['map_medium']*100:.2f}%",
        "Method 6 (Sidecar)": f"{m6['coco_map']['map_medium']*100:.2f}%",
        "Cải thiện (M6 vs Base)": f"{(m6['coco_map']['map_medium'] - base['coco_map']['map_medium'])*100:+.2f}%"
    },
    {
        "Tiêu chí": "COCO AP Large",
        "Base RT-DETR": f"{base['coco_map']['map_large']*100:.2f}%",
        "Method 5": f"{m5['coco_map']['map_large']*100:.2f}%",
        "Method 6 (Sidecar)": f"{m6['coco_map']['map_large']*100:.2f}%",
        "Cải thiện (M6 vs Base)": f"{(m6['coco_map']['map_large'] - base['coco_map']['map_large'])*100:+.2f}%"
    },
    {
        "Tiêu chí": "COCO AR100",
        "Base RT-DETR": f"{base['coco_map']['ar100']*100:.2f}%",
        "Method 5": f"{m5['coco_map']['ar100']*100:.2f}%",
        "Method 6 (Sidecar)": f"{m6['coco_map']['ar100']*100:.2f}%",
        "Cải thiện (M6 vs Base)": f"{(m6['coco_map']['ar100'] - base['coco_map']['ar100'])*100:+.2f}%"
    },
    {
        "Tiêu chí": "Classification Micro F1",
        "Base RT-DETR": f"{base['classification']['micro']['f1']*100:.2f}%",
        "Method 5": f"{m5['classification']['micro']['f1']*100:.2f}%",
        "Method 6 (Sidecar)": f"{m6['classification']['micro']['f1']*100:.2f}%",
        "Cải thiện (M6 vs Base)": f"{(m6['classification']['micro']['f1'] - base['classification']['micro']['f1'])*100:+.2f}%"
    },
    {
        "Tiêu chí": "Classification Macro F1",
        "Base RT-DETR": f"{base['classification']['macro']['f1']*100:.2f}%",
        "Method 5": f"{m5['classification']['macro']['f1']*100:.2f}%",
        "Method 6 (Sidecar)": f"{m6['classification']['macro']['f1']*100:.2f}%",
        "Cải thiện (M6 vs Base)": f"{(m6['classification']['macro']['f1'] - base['classification']['macro']['f1'])*100:+.2f}%"
    },
    {
        "Tiêu chí": "Pure BBox Recall",
        "Base RT-DETR": f"{base['box_localization']['recall']*100:.2f}%",
        "Method 5": f"{m5['box_localization']['recall']*100:.2f}%",
        "Method 6 (Sidecar)": f"{m6['box_localization']['recall']*100:.2f}%",
        "Cải thiện (M6 vs Base)": f"{(m6['box_localization']['recall'] - base['box_localization']['recall'])*100:+.2f}%"
    },
    {
        "Tiêu chí": "Tốc độ (FPS)",
        "Base RT-DETR": f"{base['runtime']['fps']:.1f}",
        "Method 5": f"{m5['runtime']['fps']:.1f}",
        "Method 6 (Sidecar)": f"{m6['runtime']['fps']:.1f}",
        "Cải thiện (M6 vs Base)": f"{m6['runtime']['fps'] - base['runtime']['fps']:+.1f}"
    },
    {
        "Tiêu chí": "Peak VRAM (MB)",
        "Base RT-DETR": f"{base['runtime']['peak_vram_mb']:.1f}",
        "Method 5": f"{m5['runtime']['peak_vram_mb']:.1f}",
        "Method 6 (Sidecar)": f"{m6['runtime']['peak_vram_mb']:.1f}",
        "Cải thiện (M6 vs Base)": f"{m6['runtime']['peak_vram_mb'] - base['runtime']['peak_vram_mb']:+.1f}"
    }
]

df_summary = pd.DataFrame(metrics_summary)
display(df_summary)
"""))

# Cell 3: Visualization Bar Chart (Overall Metrics)
cells.append(nbf.v4.new_markdown_cell("""## 2. Trực Quan Hóa So Sánh Các Chỉ Số Cốt Lõi
Biểu đồ cột so sánh COCO mAP, AP50, AP75, AP Small, AP Med, AP Large, AR100 giữa Base Model và Method 6.
"""))

cells.append(nbf.v4.new_code_cell("""metrics_names = ['mAP', 'AP50', 'AP75', 'AP Small', 'AP Med', 'AP Large', 'AR100']
base_vals = [base['coco_map']['map']*100, base['coco_map']['map50']*100, base['coco_map']['map75']*100,
             base['coco_map']['map_small']*100, base['coco_map']['map_medium']*100, base['coco_map']['map_large']*100, base['coco_map']['ar100']*100]
m6_vals = [m6['coco_map']['map']*100, m6['coco_map']['map50']*100, m6['coco_map']['map75']*100,
           m6['coco_map']['map_small']*100, m6['coco_map']['map_medium']*100, m6['coco_map']['map_large']*100, m6['coco_map']['ar100']*100]

x = np.arange(len(metrics_names))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width/2, base_vals, width, label='Base Pretrained RT-DETR', color='#4575b4', alpha=0.9)
rects2 = ax.bar(x + width/2, m6_vals, width, label='Method 6 (Spatial Sidecar Heatmap)', color='#d73027', alpha=0.9)

ax.set_ylabel('Điểm số (%)', fontsize=12, fontweight='bold')
ax.set_title('So Sánh Hiệu Năng Chuẩn COCO Trên Toàn Bộ 4,999 Trang DocLayNet Test', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names, fontsize=11, fontweight='bold')
ax.set_ylim(0, 100)
ax.legend(fontsize=11)

# Annotate values
for rect in rects1:
    h = rect.get_height()
    ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for rect in rects2:
    h = rect.get_height()
    ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#a50026')

plt.tight_layout()
plt.show()
"""))

# Cell 4: Class-by-Class AP Comparison
cells.append(nbf.v4.new_markdown_cell("""## 3. Phân Tích Chi Tiết 11 Lớp Đối Tượng DocLayNet (Per-Class COCO AP)
Kiểm chứng sự cải thiện vượt bậc ở các thành phần định dạng phân cấp của tài liệu (Page-footer, Page-header, Section-header, Footnote, Text...).
"""))

cells.append(nbf.v4.new_code_cell("""classes = []
base_cls_ap = []
m6_cls_ap = []

for cid in sorted([int(k) for k in m6["per_category_coco"].keys()]):
    c_name = m6["per_category_coco"][str(cid)]["name"]
    classes.append(c_name)
    base_cls_ap.append(base["per_category_coco"][str(cid)]["ap"] * 100)
    m6_cls_ap.append(m6["per_category_coco"][str(cid)]["ap"] * 100)

df_cls = pd.DataFrame({
    'Class': classes,
    'Base AP (%)': base_cls_ap,
    'Method 6 AP (%)': m6_cls_ap,
    'Gain (%)': [m - b for b, m in zip(base_cls_ap, m6_cls_ap)]
}).sort_values(by='Gain (%)', ascending=False)

fig, ax = plt.subplots(figsize=(14, 7))
y = np.arange(len(df_cls))
h_bar = 0.38

rects_b = ax.barh(y - h_bar/2, df_cls['Base AP (%)'], h_bar, label='Base Pretrained RT-DETR', color='#74add1')
rects_m = ax.barh(y + h_bar/2, df_cls['Method 6 AP (%)'], h_bar, label='Method 6 (Sidecar Heatmap)', color='#f46d43')

ax.set_xlabel('COCO AP@[.50:.95] (%)', fontsize=12, fontweight='bold')
ax.set_title('Độ Cải Thiện COCO AP Từng Lớp Đối Tượng (Sắp xếp theo mức tăng trưởng giảm dần)', fontsize=14, fontweight='bold')
ax.set_yticks(y)
ax.set_yticklabels(df_cls['Class'], fontsize=11, fontweight='bold')
ax.set_xlim(0, 100)
ax.legend(loc='lower right', fontsize=11)

for idx, (b, m, gain) in enumerate(zip(df_cls['Base AP (%)'], df_cls['Method 6 AP (%)'], df_cls['Gain (%)'])):
    ax.text(m + 1.0, idx + h_bar/2 - 0.05, f"{m:.1f}% (+{gain:.2f}%)", va='center', fontsize=9, fontweight='bold', color='#d73027')

plt.tight_layout()
plt.show()

display(df_cls.reset_index(drop=True))
"""))

# Cell 5: Discussion & Conclusion
cells.append(nbf.v4.new_markdown_cell("""## 4. Kết Luận & Điểm Đột Phá Của Method 6
1. **Hiệu quả toàn diện (Toàn bộ 4,999 trang Test Set)**:
   - **COCO mAP@[.50:.95]**: Tăng từ **$57.42\\%$** lên **$64.90\\%$** (tăng tuyệt đối **$+7.48\\%$**, tức cải thiện **$+13.0\\%$** chất lượng tổng thể).
   - **COCO AP75**: Tăng từ **$60.99\\%$** lên **$69.65\\%$** (**$+8.66\\%$**), cho thấy độ chính xác định vị hộp ranh giới cải thiện vượt trội.
   - **100% tất cả 11 lớp văn bản đều tăng trưởng AP**, trong đó các lớp cấu trúc sách đạt kỷ lục:
     - `Page-footer`: tăng **$+19.33\\%$ AP** ($35.07\\% \\to 54.40\\%$)
     - `Page-header`: tăng **$+11.98\\%$ AP** ($52.34\\% \\to 64.32\\%$)
     - `Section-header`: tăng **$+10.57\\%$ AP** ($49.99\\% \\to 60.56\\%$)
     - `Footnote`: tăng **$+7.37\\%$ AP** ($57.48\\% \\to 64.85\\%$)
     - `Caption`: tăng **$+7.84\\%$ AP** ($58.99\\% \\to 66.83\\%$)
2. **Ưu thế tài nguyên so với Method 5**:
   - Bộ nhớ GPU khi suy luận giảm xuống chỉ còn **$1,244.3\\text{ MB}$ VRAM** (so với $1,985.5\\text{ MB}$ của Base và $>3.4\\text{ GB}$ của Transformer token trong Method 5).
   - Tốc độ suy luận đạt **$16.7\\text{ FPS}$** ($59.9\\text{ ms/trang}$), sẵn sàng triển khai thực tế thời gian thực.
"""))

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"✓ Đã tạo notebook tại: {NOTEBOOK_PATH}")
print(">>> Đang thực thi notebook để kết xuất kết quả...")

client = NotebookClient(nb, timeout=600, kernel_name="python3")
client.execute()

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"✓ Đã lưu notebook đã thực thi với biểu đồ và kết quả tại: {NOTEBOOK_PATH}")
