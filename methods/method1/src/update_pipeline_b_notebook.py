"""Helper script to update method1_pipeline_b.ipynb with standardized inference, evaluation and visualization."""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
nb_path = ROOT / "methods" / "method1" / "notebooks" / "method1_pipeline_b.ipynb"


def to_source(lines):
    return [line + "\n" for line in lines[:-1]] + [lines[-1]] if lines else []


c8_md = [
    "## 4. Biểu Đồ Huấn Luyện (Training History)",
    "Vẽ đường cong mất mát (`Loss`) và độ chính xác phân loại (`Accuracy`) qua từng epoch vừa huấn luyện:",
]

c9_py = [
    'if "history" in locals() and len(history.get("train_loss", [])) > 0:',
    "    fig, axes = plt.subplots(1, 2, figsize=(16, 5))",
    '    epochs_range = range(1, len(history["train_loss"]) + 1)',
    "",
    "    # 1. Loss Curve",
    '    axes[0].plot(epochs_range, history["train_loss"], "o-", label="Train Loss", color="#e74c3c", linewidth=2)',
    '    axes[0].plot(epochs_range, history["val_loss"], "s-", label="Val Loss", color="#3498db", linewidth=2)',
    '    axes[0].set_title("Đường Cong Mất Mát (Cross-Entropy Loss)", fontsize=12, fontweight="bold")',
    '    axes[0].set_xlabel("Epoch")',
    '    axes[0].set_ylabel("Loss")',
    '    axes[0].grid(True, linestyle="--", alpha=0.6)',
    "    axes[0].legend()",
    "",
    "    # 2. Accuracy Curve",
    '    axes[1].plot(epochs_range, [a * 100 for a in history["train_acc"]], "o-", label="Train Acc", color="#2ecc71", linewidth=2)',
    '    axes[1].plot(epochs_range, [a * 100 for a in history["val_acc"]], "s-", label="Val Acc", color="#9b59b6", linewidth=2)',
    '    axes[1].set_title("Độ Chính Xác Phân Loại (Accuracy %)", fontsize=12, fontweight="bold")',
    '    axes[1].set_xlabel("Epoch")',
    '    axes[1].set_ylabel("Accuracy (%)")',
    '    axes[1].grid(True, linestyle="--", alpha=0.6)',
    "    axes[1].legend()",
    "",
    "    plt.tight_layout()",
    "    plt.show()",
    "else:",
    '    print("ℹ️ Lịch sử huấn luyện không có trong bộ nhớ; mô hình tối ưu đã được lưu tại: methods/method1/weights/pipeline_b_transformer_10k_best.pth")',
]

c10_md = [
    "## 1. Cấu Hình & Chạy Suy Luận End-to-End (Prediction)",
    "",
    "Chạy quy trình suy luận 2 giai đoạn hoàn chỉnh của **Method 1**:",
    "* **Giai đoạn 1 (Pipeline A)**: `yolov8n_doclaynet_10k_best.pt` phát hiện toàn bộ bounding boxes (Localization).",
    "* **Giai đoạn 2 (Pipeline B)**: `pipeline_b_transformer_10k_best.pth` (vừa huấn luyện) phân loại nhãn layout cho từng bounding box.",
    "",
    "Kết quả được lưu vào thư mục chuẩn của toàn dự án: `outputs/predictions/method1/<dataset>/<image_id>.json`.",
    "Tương tự các pipeline khác (`docling`, `paddleocr_vl`, `m2doc`, `dots_ocr`), script tự động bỏ qua các ảnh đã có kết quả (`FORCE_INFERENCE = False`) hoặc chạy lại toàn bộ (`FORCE_INFERENCE = True`).",
]

c11_py = [
    "import subprocess",
    "",
    'METHOD = "method1"',
    'DATASET = "doclaynet_test"',
    'SPLIT = "test"',
    "IOU_THRESHOLD = 0.5",
    "LIMIT = 0  # 0: toàn bộ 4,999 ảnh test (~4 phút); hoặc đặt vd 100, 200 để chạy mẫu nhanh",
    "FORCE_INFERENCE = False  # True: chạy lại và ghi đè bằng trọng số mới nhất; False: bỏ qua ảnh đã có prediction",
    "FIGURE_LIMIT = 20  # Số ảnh đối chiếu trực quan xuất ra thư mục báo cáo",
    "",
    'PREDICTIONS_DIR = ROOT / "outputs" / "predictions" / METHOD / DATASET',
    'FIGURES_DIR = ROOT / "outputs" / "reports" / f"{METHOD}_{DATASET}"',
    "FIGURES_DIR.mkdir(parents=True, exist_ok=True)",
    "",
    'print(f"Project: {ROOT}")',
    'print(f"Pipeline: {METHOD} | Dataset: {DATASET}/{SPLIT} | Ngưỡng IoU >= {IOU_THRESHOLD}")',
    'print(f"Chế độ suy luận: {\'Ghi đè bằng weights 10k mới (Force rerun)\' if FORCE_INFERENCE else \'Bỏ qua ảnh đã có kết quả (Skip existing)\'}")',
    'print(f"Thư mục Predictions: {PREDICTIONS_DIR}")',
    'print(f"Thư mục Figures:     {FIGURES_DIR}")',
]

c12_py = [
    'inference_script = ROOT / "methods" / METHOD / "src" / "run_inference.py"',
    "",
    "command = [",
    "    sys.executable,",
    "    str(inference_script),",
    '    "--dataset", DATASET,',
    '    "--limit", str(LIMIT),',
    '    "--force" if FORCE_INFERENCE else "--skip-existing",',
    "]",
    "",
    'print("Running:", " ".join(command))',
    'env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}',
    "proc = subprocess.Popen(",
    "    command,",
    "    cwd=str(ROOT),",
    "    env=env,",
    "    text=True,",
    '    encoding="utf-8",',
    '    errors="replace",',
    "    stdout=subprocess.PIPE,",
    "    stderr=subprocess.STDOUT,",
    ")",
    "",
    "for line in proc.stdout:",
    '    print(line, end="", flush=True)',
    "",
    "if proc.wait() != 0:",
    '    raise RuntimeError(f"Quá trình suy luận thất bại (exit code {proc.returncode}).")',
    "",
    'print(f"\\n✓ Hoàn tất suy luận! Predictions lưu tại: {PREDICTIONS_DIR}")',
]

c13_md = [
    "## 2. Metrics and Evaluation Chart",
    "",
    "Đánh giá toàn diện kết quả của **Method 1** bằng script chuẩn của toàn dự án (`scripts/evaluate_predictions.py`):",
    "* **Strict (Vị trí + Đúng nhãn)**: Bắt buộc IoU >= 0.50 và khớp đúng 1 trong 11 nhãn DocLayNet.",
    "* **Loose (Localization-only)**: Bỏ qua nhãn, đánh giá khả năng bao quát và bắt trúng vùng tài liệu của mô hình.",
    "* **COCO mAP / AR**: mAP@[.5:.95], AP50, AP75, AR và chi tiết AP của từng danh mục layout.",
    "* **Runtime Profiling**: Thống kê thời gian xử lý trung bình mỗi trang (mean/median/p95), peak RAM, peak VRAM GPU.",
    "* **Biểu đồ độ đo**: Tự động vẽ và xuất biểu đồ so sánh `metrics_chart.png` vào thư mục báo cáo.",
]

c14_py = [
    'evaluation_script = ROOT / "scripts" / "evaluate_predictions.py"',
    'chart_file = FIGURES_DIR / "metrics_chart.png"',
    "",
    "command = [",
    "    sys.executable,",
    "    str(evaluation_script),",
    '    "--method", METHOD,',
    '    "--dataset", DATASET,',
    '    "--split", SPLIT,',
    '    "--iou-threshold", str(IOU_THRESHOLD),',
    '    "--loose",',
    '    "--plot", str(chart_file),',
    "]",
    "if LIMIT > 0:",
    '    command.append("--allow-incomplete")',
    "",
    'print("Running evaluation:", " ".join(command))',
    'env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}',
    "proc = subprocess.Popen(",
    "    command,",
    "    cwd=str(ROOT),",
    "    env=env,",
    "    text=True,",
    '    encoding="utf-8",',
    '    errors="replace",',
    "    stdout=subprocess.PIPE,",
    "    stderr=subprocess.STDOUT,",
    ")",
    "",
    "for line in proc.stdout:",
    '    print(line, end="", flush=True)',
    "",
    "if proc.wait() != 0:",
    '    raise RuntimeError(f"Đánh giá thất bại (exit code {proc.returncode}).")',
    "",
    "# Hiển thị trực tiếp biểu đồ độ đo tổng thể",
    "if chart_file.is_file():",
    "    from IPython.display import Image as DisplayImage, display",
    '    print(f"\\nBiểu đồ đánh giá tổng thể ({chart_file.name}):")',
    "    display(DisplayImage(filename=str(chart_file)))",
]

c15_md = [
    "## 3. Ground Truth vs. Prediction Examples",
    "",
    "Trực quan hóa đối chiếu nhãn thực tế (**Ground Truth**) và kết quả dự đoán của mô hình (**Method 1**) cho `FIGURE_LIMIT` ảnh đầu của split.",
    "Sử dụng script dùng chung của dự án (`visualization/scripts/visualize_predictions.py`):",
    "* Hệ màu chuẩn 11 classes DocLayNet.",
    "* Nhãn hiển thị dạng pill badge tránh đè chữ.",
    "* Ghép 2 khung hình song song (Side-by-side): `[Ground Truth] | [Method 1 Prediction]`.",
]

c16_py = [
    'visualizer = ROOT / "visualization" / "scripts" / "visualize_predictions.py"',
    "",
    "command = [",
    "    sys.executable,",
    "    str(visualizer),",
    '    "--method", METHOD,',
    '    "--dataset", DATASET,',
    '    "--split", SPLIT,',
    '    "--limit", str(FIGURE_LIMIT),',
    '    "--output-dir", str(FIGURES_DIR),',
    "]",
    "",
    'print("Running visualization:", " ".join(command))',
    'env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}',
    "proc = subprocess.Popen(",
    "    command,",
    "    cwd=str(ROOT),",
    "    env=env,",
    "    text=True,",
    '    encoding="utf-8",',
    '    errors="replace",',
    "    stdout=subprocess.PIPE,",
    "    stderr=subprocess.STDOUT,",
    ")",
    "",
    "for line in proc.stdout:",
    '    print(line, end="", flush=True)',
    "",
    "if proc.wait() != 0:",
    '    raise RuntimeError(f"Visualization thất bại (exit code {proc.returncode}).")',
    "",
    "figure_files = sorted(",
    '    FIGURES_DIR.glob("*.png"),',
    "    key=lambda path: (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem),",
    ")",
    'example_figures = [f for f in figure_files if f.name != "metrics_chart.png"]',
    "",
    "if example_figures:",
    "    from IPython.display import Image as DisplayImage, display",
    '    print(f"\\nHiển thị các ví dụ đối chiếu trực quan Ground Truth vs Dự đoán Method 1 ({len(example_figures)} ảnh đã sinh):")',
    "    for fig_path in example_figures[:3]:",
    '        print(f"• Trang {fig_path.name}:", flush=True)',
    "        display(DisplayImage(filename=str(fig_path)))",
    "else:",
    '    print(f"Không có hình ảnh đối chiếu nào trong {FIGURES_DIR}")',
]

c17_md = [
    "## Files đầu ra và báo cáo",
    "",
    "- **Predictions JSON**: `outputs/predictions/method1/doclaynet_test/<image_id>.json`",
    "- **Biểu đồ đánh giá**: `outputs/reports/method1_doclaynet_test/metrics_chart.png`",
    "- **Ảnh đối chiếu trực quan**: `outputs/reports/method1_doclaynet_test/<image_id>.png`",
    "- **Ground Truth COCO**: `datasets/doclaynet_test/processed/annotations/test.json`",
    "- **Runtime Summary**: `outputs/runtime/method1/doclaynet_test.json`",
    "- **Trọng số Detector**: `methods/method1/weights/yolov8n_doclaynet_10k_best.pt`",
    "- **Trọng số Classifier**: `methods/method1/weights/pipeline_b_transformer_10k_best.pth`",
]


def main():
    # Validate code
    for label, code_lines in [
        ("c9_py", c9_py),
        ("c11_py", c11_py),
        ("c12_py", c12_py),
        ("c14_py", c14_py),
        ("c16_py", c16_py),
    ]:
        ast.parse("\n".join(code_lines))
        print(f"✓ Syntax valid for {label}")

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Preserve cells 0-7 (training setup and training execution)
    preserved = nb["cells"][:8]

    new_cells = preserved + [
        {"cell_type": "markdown", "metadata": {}, "source": to_source(c8_md)},
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": to_source(c9_py),
        },
        {"cell_type": "markdown", "metadata": {}, "source": to_source(c10_md)},
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": to_source(c11_py),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": to_source(c12_py),
        },
        {"cell_type": "markdown", "metadata": {}, "source": to_source(c13_md)},
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": to_source(c14_py),
        },
        {"cell_type": "markdown", "metadata": {}, "source": to_source(c15_md)},
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": to_source(c16_py),
        },
        {"cell_type": "markdown", "metadata": {}, "source": to_source(c17_md)},
    ]

    nb["cells"] = new_cells
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"Successfully updated {nb_path} with {len(new_cells)} total cells!")


if __name__ == "__main__":
    main()
