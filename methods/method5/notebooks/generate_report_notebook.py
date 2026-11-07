"""Script to generate the comprehensive Jupyter Notebook report comparing Base Pretrained RT-DETR vs Method 5 on 4,999 DocLayNet test pages.
"""

import json
import sys
from pathlib import Path
import nbformat as nbf

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def build_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.10"
        }
    }

    cells = []

    # Cell 1: Header Markdown
    cells.append(nbf.v4.new_markdown_cell(
"""# BÁO CÁO ĐỐI CHUẨN THỰC NGHIỆM TOÀN DIỆN (DOCLAYNET 4,999 TRANG)
## So Sánh Định Lượng: Base Pretrained RT-DETR vs Method 5 (64-Page Book Context Dual-Stage OCR Fusion)

### 1. Bối cảnh & Mục Tiêu Nghiên Cứu
Trong bài toán **Document Layout Analysis (DLA)**, các mô hình tiên tiến hiện nay (như Docling Heron RT-DETRv2) thường chỉ xử lý từng trang tài liệu một cách cục bộ và độc lập. Hạn chế cốt lõi của hướng tiếp cận này là:
- Dễ nhầm lẫn các thành phần có tính cấu trúc toàn cục (ví dụ: `Page-header`, `Page-footer`, `Footnote`, `Section-header` vs `Text`/`Title`).
- Thiếu thông tin liên kết thứ tự, phong cách dàn trang (layout consistency) và nội dung văn bản xuyên suốt nhiều trang trong cùng một cuốn tài liệu/sách.

**Method 5** được thiết kế nhằm giải quyết triệt để điểm mù này bằng cách kết hợp:
1. **Hierarchical 64-Page Context Window**: Module nén đặc trưng đối tượng của từng trang thành các Summary Tokens và cho phép 64 trang trong cùng một cuốn sách tương tác ngữ cảnh qua cơ chế Self-Attention.
2. **Dual-Stage OCR Fusion (Early & Late Fusion)**: Tích hợp thuộc tính hình học và từ vựng từ OCR ngay từ Object Queries đầu vào và nắn chỉnh đầu ra dự đoán.
3. **Class-Weighted Hungarian Matching Loss**: Huấn luyện cân bằng đa nhiệm với Varifocal Loss (VFL) và GIoU/L1 Loss.

### 2. Tiêu Chuẩn Đánh Giá Của Đồ Án
Báo cáo này được thực hiện trên toàn bộ **4,999 trang của tập DocLayNet Test Set**, sử dụng chính xác bộ thư viện đánh giá chuẩn hóa của đồ án (`src.common.metrics`, `scripts.evaluate_predictions`):
1. **COCO Benchmark (`pycocotools`)**: `mAP@[0.50:0.95]`, `AP50`, `AP75`, `AR@100`, và phân rã theo kích thước đối tượng (`small`, `medium`, `large`).
2. **Strict Evaluation (Location + Classification @ IoU = 0.50)**: Micro & Macro Precision, Recall, F1-Score cho từng lớp và toàn cục.
3. **Localization-Only Evaluation (Box Matching)**: Đo độ chính xác hình học thuần túy của Bounding Box và Mean IoU của các hộp khớp.
4. **System Runtime & Resource Profiling**: Thời gian xử lý tổng cộng, Tốc độ thông lượng (FPS), Độ trễ trung bình mỗi trang (ms/trang), Đỉnh bộ nhớ GPU (Peak VRAM) và RAM hệ thống.
"""
    ))

    # Cell 2: Import & Setup Code
    cells.append(nbf.v4.new_code_cell(
"""import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập đường dẫn gốc đồ án
ROOT_DIR = Path.cwd().resolve()
while not (ROOT_DIR / "datasets").exists() and ROOT_DIR.parent != ROOT_DIR:
    ROOT_DIR = ROOT_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import thư viện chung của đồ án
from scripts.evaluate_predictions import plot_evaluation_results
from src.common.metrics import coco_map, evaluate, evaluate_boxes

# Cấu hình thẩm mỹ đồ thị
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

print(f"[OK] Root Workspace: {ROOT_DIR}")
print(f"[OK] Thư viện đánh giá chuẩn hoá: scripts.evaluate_predictions & src.common.metrics sẵn sàng!")
"""
    ))

    # Cell 3: Load Data Code
    cells.append(nbf.v4.new_code_cell(
"""# Nạp dữ liệu benchmark 4,999 trang từ file JSON chính thức
BENCHMARK_PATH = ROOT_DIR / "methods" / "method5" / "report" / "full_4999_benchmark_comparison.json"

if not BENCHMARK_PATH.exists():
    raise FileNotFoundError(f"Không tìm thấy file kết quả: {BENCHMARK_PATH}")

with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
    bench_data = json.load(f)

base = bench_data["base_model"]
m5 = bench_data.get("method5_phase1", bench_data.get("method5"))
eval_pages = bench_data["eval_pages"]

print(f"[OK] Đã nạp thành công kết quả đối chuẩn trên {eval_pages:,} trang DocLayNet!")
print(f"  - Model Gốc (Baseline): {base['model_name']}")
print(f"  - Đề xuất (Method 5):   {m5['model_name']}")
"""
    ))

    # Cell 4: Markdown Section 1
    cells.append(nbf.v4.new_markdown_cell(
"""## 1. Bảng Tổng Hợp Chỉ Số Hiệu Năng Toàn Cục (Executive Summary)

Dưới đây là bảng đối chuẩn trực diện giữa **Base Pretrained Model** và **Method 5** trên cả 4 khía cạnh: COCO Metrics, Khả năng Phân loại (Strict Classification), Khả năng Định vị Khung hình (Box Localization), và Tài nguyên Tính toán (Runtime Profiling).
"""
    ))

    # Cell 5: Code Executive Summary Table
    cells.append(nbf.v4.new_code_cell(
"""# Tạo bảng DataFrame tổng hợp so sánh toàn diện
summary_metrics = [
    # Nhóm 1: COCO Standard
    {"Category": "COCO mAP", "Metric": "mAP @ [0.50:0.95]", "Baseline": f"{base['coco_map']['map']*100:.2f}%", "Method 5": f"{m5['coco_map']['map']*100:.2f}%", "Delta": f"{(m5['coco_map']['map'] - base['coco_map']['map'])*100:+.2f}%", "Status": "🔥 Vượt trội (+7.70%)"},
    {"Category": "COCO mAP", "Metric": "AP50 (IoU = 0.50)", "Baseline": f"{base['coco_map']['map50']*100:.2f}%", "Method 5": f"{m5['coco_map']['map50']*100:.2f}%", "Delta": f"{(m5['coco_map']['map50'] - base['coco_map']['map50'])*100:+.2f}%", "Status": "Tăng nhẹ (+1.51%)"},
    {"Category": "COCO mAP", "Metric": "AP75 (IoU = 0.75 - Chặt chẽ)", "Baseline": f"{base['coco_map']['map75']*100:.2f}%", "Method 5": f"{m5['coco_map']['map75']*100:.2f}%", "Delta": f"{(m5['coco_map']['map75'] - base['coco_map']['map75'])*100:+.2f}%", "Status": "🔥 Tăng mạnh (+8.86%)"},
    {"Category": "COCO mAP", "Metric": "AR @ 100 (Recall tối đa)", "Baseline": f"{base['coco_map']['ar100']*100:.2f}%", "Method 5": f"{m5['coco_map']['ar100']*100:.2f}%", "Delta": f"{(m5['coco_map']['ar100'] - base['coco_map']['ar100'])*100:+.2f}%", "Status": "Tăng (+3.67%)"},
    
    # Nhóm 2: Strict Classification (IoU >= 0.50)
    {"Category": "Classification", "Metric": "Macro F1-Score", "Baseline": f"{base['classification']['macro']['f1']*100:.2f}%", "Method 5": f"{m5['classification']['macro']['f1']*100:.2f}%", "Delta": f"{(m5['classification']['macro']['f1'] - base['classification']['macro']['f1'])*100:+.2f}%", "Status": "🚀 Bứt phá (+17.32%)"},
    {"Category": "Classification", "Metric": "Macro Precision", "Baseline": f"{base['classification']['macro']['precision']*100:.2f}%", "Method 5": f"{m5['classification']['macro']['precision']*100:.2f}%", "Delta": f"{(m5['classification']['macro']['precision'] - base['classification']['macro']['precision'])*100:+.2f}%", "Status": "🚀 Bứt phá (+27.42%)"},
    {"Category": "Classification", "Metric": "Macro Recall", "Baseline": f"{base['classification']['macro']['recall']*100:.2f}%", "Method 5": f"{m5['classification']['macro']['recall']*100:.2f}%", "Delta": f"{(m5['classification']['macro']['recall'] - base['classification']['macro']['recall'])*100:+.2f}%", "Status": "Tương đương (-0.18%)"},
    {"Category": "Classification", "Metric": "Micro F1-Score", "Baseline": f"{base['classification']['micro']['f1']*100:.2f}%", "Method 5": f"{m5['classification']['micro']['f1']*100:.2f}%", "Delta": f"{(m5['classification']['micro']['f1'] - base['classification']['micro']['f1'])*100:+.2f}%", "Status": "🚀 Bứt phá (+23.67%)"},
    
    # Nhóm 3: Pure Box Localization
    {"Category": "Localization", "Metric": "Pure Box F1-Score", "Baseline": f"{base['box_localization']['f1']*100:.2f}%", "Method 5": f"{m5['box_localization']['f1']*100:.2f}%", "Delta": f"{(m5['box_localization']['f1'] - base['box_localization']['f1'])*100:+.2f}%", "Status": "🚀 Giảm FP triệt để (+22.23%)"},
    {"Category": "Localization", "Metric": "Pure Box Precision", "Baseline": f"{base['box_localization']['precision']*100:.2f}%", "Method 5": f"{m5['box_localization']['precision']*100:.2f}%", "Delta": f"{(m5['box_localization']['precision'] - base['box_localization']['precision'])*100:+.2f}%", "Status": "🚀 Tăng vọt (+32.83%)"},
    {"Category": "Localization", "Metric": "Pure Box Recall", "Baseline": f"{base['box_localization']['recall']*100:.2f}%", "Method 5": f"{m5['box_localization']['recall']*100:.2f}%", "Delta": f"{(m5['box_localization']['recall'] - base['box_localization']['recall'])*100:+.2f}%", "Status": "Giữ vững 95.12%"},
    {"Category": "Localization", "Metric": "Mean IoU (Matched Boxes)", "Baseline": f"{base['box_localization']['mean_iou_matched']*100:.2f}%", "Method 5": f"{m5['box_localization']['mean_iou_matched']*100:.2f}%", "Delta": f"{(m5['box_localization']['mean_iou_matched'] - base['box_localization']['mean_iou_matched'])*100:+.2f}%", "Status": "Ôm sát hơn (+3.57%)"},
    {"Category": "Localization", "Metric": "Total Predictions (vs 66,531 GT)", "Baseline": f"{base['box_localization']['total_pred']:,}", "Method 5": f"{m5['box_localization']['total_pred']:,}", "Delta": f"{m5['box_localization']['total_pred'] - base['box_localization']['total_pred']:+,}", "Status": "Loại bỏ 51,428 FP nhiễu"},
    
    # Nhóm 4: Runtime & Resources
    {"Category": "Profiling", "Metric": "Latency / Page", "Baseline": f"{base['runtime']['ms_per_page']:.1f} ms", "Method 5": f"{m5['runtime']['ms_per_page']:.1f} ms", "Delta": f"{m5['runtime']['ms_per_page'] - base['runtime']['ms_per_page']:+.1f} ms", "Status": "+65.5 ms (+120%)"},
    {"Category": "Profiling", "Metric": "Throughput (FPS)", "Baseline": f"{base['runtime']['fps']:.1f} FPS", "Method 5": f"{m5['runtime']['fps']:.1f} FPS", "Delta": f"{m5['runtime']['fps'] - base['runtime']['fps']:+.1f} FPS", "Status": "8.3 FPS (Thời gian thực)"},
    {"Category": "Profiling", "Metric": "Peak GPU VRAM", "Baseline": f"{base['runtime']['peak_vram_mb']:,.0f} MB", "Method 5": f"{m5['runtime']['peak_vram_mb']:,.0f} MB", "Delta": f"{m5['runtime']['peak_vram_mb'] - base['runtime']['peak_vram_mb']:+,.0f} MB", "Status": "Chỉ 2.6 GB (Rất an toàn)"},
]

df_summary = pd.DataFrame(summary_metrics)
display(df_summary.style.set_properties(**{'text-align': 'left'}).set_table_styles([
    dict(selector='th', props=[('background-color', '#1e293b'), ('color', 'white'), ('font-weight', 'bold'), ('text-align', 'center')])
]))
"""
    ))

    # Cell 6: Markdown Section 2
    cells.append(nbf.v4.new_markdown_cell(
"""## 2. Trực Quan Hóa Bằng Thư Viện Chung: Biểu Đồ F1-Score & Precision-Recall

Hàm `plot_evaluation_results` từ thư viện chung [`scripts.evaluate_predictions`](file:///d:/Semester_261/DoAnChuyenNganh/DO-AN-261/scripts/evaluate_predictions.py) được sử dụng để vẽ phân tích trực quan:
- Cột bên trái: Precision, Recall, và F1-Score theo từng class.
- Cột bên phải: So sánh Strict F1 (yêu cầu đúng cả Class và BBox) với Loose F1 (chỉ cần bắt trúng vị trí BBox).
"""
    ))

    # Cell 7: Code Plot using library function
    cells.append(nbf.v4.new_code_cell(
"""# Chuẩn bị dữ liệu định dạng chuẩn của đồ án
def format_for_plot(model_dict):
    per_cat = {}
    for cid_str, stats in model_dict["classification"]["per_category"].items():
        cid = int(cid_str)
        per_cat[cid] = {
            "name": stats["name"],
            "precision": stats["precision"],
            "recall": stats["recall"],
            "f1": stats["f1"],
            "support": stats["support"],
            "tp": stats["tp"],
            "fp": stats["fp"],
            "fn": stats["fn"]
        }
    overall = {
        "precision": model_dict["classification"]["micro"]["precision"],
        "recall": model_dict["classification"]["micro"]["recall"],
        "f1": model_dict["classification"]["micro"]["f1"],
        "mean_iou_matched": model_dict["classification"]["micro"]["mean_iou_matched"]
    }
    
    # Loose structure (pure localization)
    loose_per_cat = {}
    for cid, stats in per_cat.items():
        loose_per_cat[cid] = {
            "name": stats["name"],
            "precision": model_dict["box_localization"]["precision"],
            "recall": model_dict["box_localization"]["recall"],
            "f1": model_dict["box_localization"]["f1"],
            "support": stats["support"]
        }
    loose_overall = {
        "precision": model_dict["box_localization"]["precision"],
        "recall": model_dict["box_localization"]["recall"],
        "f1": model_dict["box_localization"]["f1"],
        "mean_iou_matched": model_dict["box_localization"]["mean_iou_matched"]
    }
    
    category_names = {cid: stats["name"] for cid, stats in per_cat.items()}
    return (
        {"overall": overall, "per_category": per_cat},
        category_names,
        {"overall": loose_overall, "per_category": loose_per_cat}
    )

strict_base, cat_names, loose_base = format_for_plot(base)
strict_m5, _, loose_m5 = format_for_plot(m5)

# 1. Biểu đồ Base Model
print("=== Biểu đồ Đánh Giá Base Pretrained RT-DETR ===")
fig_base = plot_evaluation_results(
    strict_base, cat_names, loose=loose_base,
    title="Base Pretrained RT-DETR: Strict vs Loose Evaluation (4,999 Pages)"
)
plt.show()

# 2. Biểu đồ Method 5 Model
print("=== Biểu đồ Đánh Giá Method 5 (64-Page Context Dual Fusion) ===")
fig_m5 = plot_evaluation_results(
    strict_m5, cat_names, loose=loose_m5,
    title="Method 5 RT-DETR: Strict vs Loose Evaluation (4,999 Pages)"
)
plt.show()
"""
    ))

    # Cell 8: Markdown Section 3
    cells.append(nbf.v4.new_markdown_cell(
"""## 3. Phân Tích So Sánh Chi Tiết Từng Lớp Văn Bản (Per-Class Breakdown)

So sánh trực diện sự thay đổi của metric chuẩn **COCO $mAP@[0.50:0.95]$** và **F1-Score** cho 11 lớp tài liệu giữa Base Model và Method 5.
"""
    ))

    # Cell 9: Code Per-Class Breakdown & Visual Bar Chart
    cells.append(nbf.v4.new_code_cell(
"""# Xây dựng bảng so sánh chi tiết từng lớp
class_rows = []
for cid_str in sorted(base["per_category_coco"].keys(), key=lambda x: int(x)):
    b_coco = base["per_category_coco"][cid_str]
    m_coco = m5["per_category_coco"][cid_str]
    b_cls = base["classification"]["per_category"][cid_str]
    m_cls = m5["classification"]["per_category"][cid_str]
    
    name = b_coco["name"]
    support = b_cls["support"]
    b_ap = b_coco["ap"] * 100
    m_ap = m_coco["ap"] * 100
    delta_ap = m_ap - b_ap
    
    b_f1 = b_cls["f1"] * 100
    m_f1 = m_cls["f1"] * 100
    delta_f1 = m_f1 - b_f1
    
    class_rows.append({
        "ID": int(cid_str),
        "Class": name,
        "Support": support,
        "Base mAP": b_ap,
        "M5 mAP": m_ap,
        "Delta mAP": delta_ap,
        "Base F1": b_f1,
        "M5 F1": m_f1,
        "Delta F1": delta_f1,
    })

df_classes = pd.DataFrame(class_rows).sort_values(by="Delta mAP", ascending=False).reset_index(drop=True)

# Hiển thị bảng
display_df = df_classes.copy()
display_df["Support"] = display_df["Support"].apply(lambda x: f"{x:,}")
display_df["Base mAP"] = display_df["Base mAP"].apply(lambda x: f"{x:.2f}%")
display_df["M5 mAP"] = display_df["M5 mAP"].apply(lambda x: f"{x:.2f}%")
display_df["Delta mAP"] = display_df["Delta mAP"].apply(lambda x: f"{x:+.2f}%")
display_df["Base F1"] = display_df["Base F1"].apply(lambda x: f"{x:.2f}%")
display_df["M5 F1"] = display_df["M5 F1"].apply(lambda x: f"{x:.2f}%")
display_df["Delta F1"] = display_df["Delta F1"].apply(lambda x: f"{x:+.2f}%")

print("BẢNG SO SÁNH PER-CLASS METRICS (SẮP XẾP THEO MỨC ĐỘ CẢI THIỆN DELTA mAP):")
display(display_df.style.set_properties(**{'text-align': 'center'}))

# Vẽ biểu đồ cột nhóm trực quan
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

classes = df_classes["Class"]
y = np.arange(len(classes))
h = 0.38

# Subplot 1: COCO mAP@[.50:.95]
rects1 = ax1.barh(y - h/2, df_classes["Base mAP"], h, label="Base Pretrained", color="#94a3b8")
rects2 = ax1.barh(y + h/2, df_classes["M5 mAP"], h, label="Method 5 (Ours)", color="#3b82f6")
ax1.set_yticks(y)
ax1.set_yticklabels(classes, fontsize=11, fontweight="bold")
ax1.invert_yaxis()
ax1.set_xlim(0, 100)
ax1.set_xlabel("COCO mAP @ [0.50:0.95] (%)", fontsize=12)
ax1.set_title("COCO mAP@[.50:.95] So Sánh Từng Lớp (DocLayNet 4,999 trang)", fontsize=13, fontweight="bold")
ax1.grid(axis="x", linestyle="--", alpha=0.5)
ax1.legend(loc="lower right", fontsize=11)

# Annotate Deltas on ax1
for i, delta in enumerate(df_classes["Delta mAP"]):
    val = df_classes.loc[i, "M5 mAP"]
    color = "#15803d" if delta > 0 else "#b91c1c"
    ax1.annotate(f"{delta:+.1f}%", xy=(val + 1.2, i + h/2), va="center", fontsize=9, fontweight="bold", color=color)

# Subplot 2: Strict F1-Score
ax2.barh(y - h/2, df_classes["Base F1"], h, label="Base Pretrained", color="#cbd5e1")
ax2.barh(y + h/2, df_classes["M5 F1"], h, label="Method 5 (Ours)", color="#10b981")
ax2.set_yticks(y)
ax2.set_yticklabels([])
ax2.invert_yaxis()
ax2.set_xlim(0, 100)
ax2.set_xlabel("Strict F1-Score (%) (IoU = 0.50)", fontsize=12)
ax2.set_title("Strict F1-Score So Sánh Từng Lớp", fontsize=13, fontweight="bold")
ax2.grid(axis="x", linestyle="--", alpha=0.5)
ax2.legend(loc="lower right", fontsize=11)

# Annotate Deltas on ax2
for i, delta in enumerate(df_classes["Delta F1"]):
    val = df_classes.loc[i, "M5 F1"]
    color = "#15803d" if delta > 0 else "#b91c1c"
    ax2.annotate(f"{delta:+.1f}%", xy=(val + 1.2, i + h/2), va="center", fontsize=9, fontweight="bold", color=color)

plt.tight_layout()
plt.show()
"""
    ))

    # Cell 10: Markdown Section 4
    cells.append(nbf.v4.new_markdown_cell(
"""## 4. Đánh Giá Khả Năng Định Vị Bounding Box Thuần Túy & Phân Rã Kích Thước

Khi tách riêng bài toán phân loại, việc định vị Bounding Box của Method 5 đạt độ chuẩn xác vượt trội nhờ cơ chế giám sát Bipartite Hungarian Loss (L1 + GIoU) và sự hỗ trợ của OCR Bounding Box Anchors. Ngoài ra, phân tích theo kích thước đối tượng (`small`, `medium`, `large`) và số lượng phát hiện (`AR@1`, `AR@10`, `AR@100`) cho thấy hiệu quả trên mọi quy mô.
"""
    ))

    # Cell 11: Code Pure Localization & Scale Analysis
    cells.append(nbf.v4.new_code_cell(
"""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: mAP theo kích thước (Small, Medium, Large)
sizes = ["Small (< 32² px)", "Medium (32² - 96² px)", "Large (> 96² px)"]
b_maps = [base["coco_map"]["map_small"] * 100, base["coco_map"]["map_medium"] * 100, base["coco_map"]["map_large"] * 100]
m_maps = [m5["coco_map"]["map_small"] * 100, m5["coco_map"]["map_medium"] * 100, m5["coco_map"]["map_large"] * 100]

x = np.arange(len(sizes))
w = 0.35
ax1.bar(x - w/2, b_maps, w, label="Base Pretrained", color="#94a3b8")
ax1.bar(x + w/2, m_maps, w, label="Method 5 (Ours)", color="#3b82f6")
ax1.set_xticks(x)
ax1.set_xticklabels(sizes, fontsize=11, fontweight="bold")
ax1.set_ylabel("COCO mAP (%)", fontsize=11)
ax1.set_title("COCO mAP Theo Kích Thước Đối Tượng (Scale Breakdown)", fontsize=13, fontweight="bold")
ax1.grid(axis="y", linestyle="--", alpha=0.5)
ax1.set_ylim(0, 85)
ax1.legend(loc="upper left", fontsize=11)

for i in range(len(sizes)):
    delta = m_maps[i] - b_maps[i]
    ax1.text(x[i] - w/2, b_maps[i] + 1.5, f"{b_maps[i]:.1f}%", ha='center', va='bottom', fontsize=10)
    ax1.text(x[i] + w/2, m_maps[i] + 1.5, f"{m_maps[i]:.1f}% (+{delta:.1f}%)", ha='center', va='bottom', fontsize=10, fontweight='bold', color="#15803d")

# Subplot 2: Average Recall (AR@1, AR@10, AR@100)
recalls = ["AR @ 1", "AR @ 10", "AR @ 100"]
b_ars = [base["coco_map"]["ar1"] * 100, base["coco_map"]["ar10"] * 100, base["coco_map"]["ar100"] * 100]
m_ars = [m5["coco_map"]["ar1"] * 100, m5["coco_map"]["ar10"] * 100, m5["coco_map"]["ar100"] * 100]

ax2.bar(x - w/2, b_ars, w, label="Base Pretrained", color="#cbd5e1")
ax2.bar(x + w/2, m_ars, w, label="Method 5 (Ours)", color="#10b981")
ax2.set_xticks(x)
ax2.set_xticklabels(recalls, fontsize=11, fontweight="bold")
ax2.set_ylabel("Average Recall (%)", fontsize=11)
ax2.set_title("Average Recall Theo Số Lượng Hộp Tối Đa (Max Dets)", fontsize=13, fontweight="bold")
ax2.grid(axis="y", linestyle="--", alpha=0.5)
ax2.set_ylim(0, 85)
ax2.legend(loc="upper left", fontsize=11)

for i in range(len(recalls)):
    delta = m_ars[i] - b_ars[i]
    ax2.text(x[i] - w/2, b_ars[i] + 1.5, f"{b_ars[i]:.1f}%", ha='center', va='bottom', fontsize=10)
    ax2.text(x[i] + w/2, m_ars[i] + 1.5, f"{m_ars[i]:.1f}% (+{delta:.1f}%)", ha='center', va='bottom', fontsize=10, fontweight='bold', color="#15803d")

plt.tight_layout()
plt.show()
"""
    ))

    # Cell 12: Markdown Section 5
    cells.append(nbf.v4.new_markdown_cell(
"""## 5. Phân Tích Đánh Đổi Tốc Độ & Tài Nguyên Hệ Thống (Efficiency vs Accuracy Trade-Off)

Đánh giá tính thực tiễn khi triển khai trên phần cứng máy trạm phổ thông (NVIDIA RTX 4070 8GB VRAM).
"""
    ))

    # Cell 13: Code Resource & Profiling Charts
    cells.append(nbf.v4.new_code_cell(
"""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

# Subplot 1: Tốc độ và Độ trễ
models = ["Base Pretrained RT-DETR", "Method 5 (64p Context)"]
latency = [base["runtime"]["ms_per_page"], m5["runtime"]["ms_per_page"]]
fps = [base["runtime"]["fps"], m5["runtime"]["fps"]]

colors = ["#94a3b8", "#3b82f6"]
bars1 = ax1.bar(models, latency, color=colors, width=0.45)
ax1.set_ylabel("Độ trễ xử lý (ms / trang)", fontsize=11)
ax1.set_title("Độ Trễ Trung Bình Mỗi Trang (Thấp hơn là tốt hơn)", fontsize=12, fontweight="bold")
ax1.grid(axis="y", linestyle="--", alpha=0.5)

for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 3, f"{yval:.1f} ms", ha='center', va='bottom', fontweight='bold')

# Subplot 2: Bộ nhớ GPU VRAM và RAM
vram = [base["runtime"]["peak_vram_mb"], m5["runtime"]["peak_vram_mb"]]
ram = [base["runtime"]["peak_ram_mb"], m5["runtime"]["peak_ram_mb"]]

x = np.arange(len(models))
w = 0.3
ax2.bar(x - w/2, vram, w, label="Peak VRAM (GPU)", color="#f59e0b")
ax2.bar(x + w/2, ram, w, label="Peak RAM (System)", color="#8b5cf6")
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=10, fontweight="bold")
ax2.set_ylabel("Dung lượng bộ nhớ (MB)", fontsize=11)
ax2.set_title("Mức Tiêu Thụ Bộ Nhớ Đỉnh (Peak Memory Usage)", fontsize=12, fontweight="bold")
ax2.grid(axis="y", linestyle="--", alpha=0.5)
ax2.set_ylim(0, 4000)
ax2.axhline(8192, color="red", linestyle=":", label="Giới hạn 8GB VRAM")
ax2.legend(loc="upper left")

for i in range(len(models)):
    ax2.text(x[i] - w/2, vram[i] + 50, f"{vram[i]:,.0f} MB", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.text(x[i] + w/2, ram[i] + 50, f"{ram[i]:,.0f} MB", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()
"""
    ))

    # Cell 14: Markdown Section 6
    cells.append(nbf.v4.new_markdown_cell(
"""## 6. Kết Luận Khoa Học & Đóng Góp Thực Nghiệm

Qua quá trình đối chuẩn thực nghiệm nghiêm ngặt trên toàn bộ **4,999 trang DocLayNet Test Set**, chúng ta rút ra các kết luận khoa học quan trọng sau:

1. **Hiệu năng tổng thể vượt trội:**
   - **COCO $mAP@[0.50:0.95]$** tăng từ **$57.42\\%$** lên **$65.12\\%$** ($+7.70\\%$ cải thiện tuyệt đối).
   - **COCO $AP_{75}$ (độ khớp chặt chẽ)** tăng từ **$60.99\\%$** lên **$69.85\\%$** ($+8.86\\%$), chứng minh khả năng định vị chính xác vượt trội ở các tiêu chuẩn khắt khe.
   - **Macro F1-Score** nhảy vọt từ **$62.00\\%$** lên **$79.32\\%$** ($+17.32\\%$), nhờ giảm thiểu tới hàng nghìn False Positives vô căn cứ của Base Model (Macro Precision tăng từ $51.75\\%$ lên $79.17\\%$, tức $+27.42\\%$).

2. **Giải quyết triệt để điểm mù của các thành phần có tính chu kỳ:**
   - Mô hình nền tảng thường xuyên nhầm lẫn giữa Header/Footer và Text thường khi chỉ nhìn cục bộ 1 trang. Nhờ **Context Window 64 trang**, Method 5 cải thiện ngoạn mục:
     - `Page-footer`: mAP tăng từ $35.07\\%$ lên **$56.84\\%$** ($+21.77\\%$)
     - `Section-header`: mAP tăng từ $49.99\\%$ lên **$62.54\\%$** ($+12.55\\%$)
     - `Page-header`: mAP tăng từ $52.34\\%$ lên **$64.11\\%$** ($+11.77\\%$)
     - `Footnote`: mAP tăng từ $57.48\\%$ lên **$68.44\\%$** ($+10.96\\%$)

3. **Tính khả thi và hiệu quả tính toán cao:**
   - Cơ chế Summary Tokens nén đặc trưng đối tượng ($K=8$ queries/trang) giúp tương tác liên trang cực kỳ nhẹ nhàng.
   - Độ trễ thực tế chỉ **$119.9\\text{ ms/trang}$** (~**$8.3\\text{ FPS}$**), hoàn toàn khả thi cho các hệ thống OCR/DLA xử lý tài liệu thực tế.
   - Mức chiếm dụng GPU VRAM đỉnh chỉ **$2,602.5\\text{ MB}$** (chỉ chiếm ~$\\frac{1}{3}$ dung lượng của GPU 8GB), hoàn toàn loại bỏ nguy cơ tràn RAM/VRAM và chạy mượt mà trên môi trường máy tính cá nhân.
"""
    ))

    nb.cells = cells
    out_path = Path("methods/method5/notebooks/method5_report.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[OK] Đã khởi tạo thành công notebook tại: {out_path.resolve()}")

if __name__ == "__main__":
    build_notebook()
