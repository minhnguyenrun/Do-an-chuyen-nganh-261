# /// script
# jupytext:
#   formats: ipynb,py:percent
#   text_representation:
#     extension: .py
#     format_name: percent
#     format_version: '1.3'
#     jupytext_version: 1.16.4
# ///
# %% [markdown]
# # dots.ocr — Kaggle report
#
# This is a Jupytext percent-format notebook. Attach the source bundle and a
# DocLayNet dataset as Kaggle inputs. No dataset files are included here.
# %%
from pathlib import Path
import json
import shutil
import subprocess
import sys
import zipfile

SOURCE_CANDIDATES = [
    Path('/kaggle/input/datasets/hongnguynngthhjk/doan261-dots-ocr-source'),
    Path('/kaggle/input/doan261-dots-ocr-source'),
]
SOURCE_ROOT = next((path for path in SOURCE_CANDIDATES if path.exists()), SOURCE_CANDIDATES[0])
WORK_ROOT = Path('/kaggle/working/DO-AN-261')
if not SOURCE_ROOT.exists():
    raise FileNotFoundError(f'Attach the Kaggle source dataset: {SOURCE_ROOT}')
shutil.copytree(SOURCE_ROOT, WORK_ROOT, dirs_exist_ok=True, ignore=shutil.ignore_patterns('dataset-metadata.json'))
ROOT = WORK_ROOT

# Install dependencies from the attached Linux wheelhouse with no network access.
wheelhouse_archive = SOURCE_ROOT / 'kaggle' / 'dots_ocr' / 'wheelhouse.zip'
wheelhouse = WORK_ROOT / 'kaggle' / 'dots_ocr' / 'wheelhouse'
if wheelhouse_archive.exists() and not wheelhouse.exists():
    wheelhouse.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheelhouse_archive) as archive:
        archive.extractall(wheelhouse)
if not wheelhouse.exists():
    raise FileNotFoundError(f'Offline dependency bundle is missing: {wheelhouse}')
requirements_file = ROOT / 'kaggle' / 'dots_ocr' / 'kaggle-requirements.txt'
runtime_requirements = ROOT / 'kaggle' / 'dots_ocr' / 'kaggle-runtime-requirements.txt'
runtime_requirements.write_text(chr(10).join(line for line in requirements_file.read_text(encoding='utf-8').splitlines() if not line.strip().lower().startswith('av')) + chr(10), encoding='utf-8')
print(f'Installing offline dependencies from {requirements_file}; AV is optional for image inference.')
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--find-links', str(wheelhouse), '-r', str(runtime_requirements)], check=True)

# Locate an attached DocLayNet dataset by its processed/images and annotations layout.
INPUT_ROOT = Path('/kaggle/input')
DOCLAYNET_ROOT = Path('/kaggle/input/datasets/hongnguynngthhjk/doclaynet')
candidates = []
for search_root in [DOCLAYNET_ROOT, Path('/kaggle/input/doclaynet'), INPUT_ROOT]:
    if search_root.exists():
        candidates.extend(p for p in search_root.rglob('images') if p.is_dir() and p.parent.name == 'processed' and (p.parent / 'annotations' / 'test.json').exists())
    if candidates:
        break
if not candidates:
    raise FileNotFoundError('Attach a DocLayNet Kaggle dataset containing processed/images and processed/annotations/test.json')
data_root = candidates[0].parent.parent
local_dataset = ROOT / 'datasets' / 'doclaynet_test'
local_dataset.mkdir(parents=True, exist_ok=True)
processed_link = local_dataset / 'processed'
if not processed_link.exists():
    processed_link.symlink_to(data_root / 'processed', target_is_directory=True)
annotation_path = processed_link / 'annotations' / 'test.json'
with annotation_path.open(encoding='utf-8') as f:
    annotation_data = json.load(f)
image_count = len(annotation_data.get('images', []))
box_count = len(annotation_data.get('annotations', []))
if image_count == 0 or box_count == 0:
    raise ValueError(f'DocLayNet annotation file is empty or invalid: {annotation_path} (images={image_count}, boxes={box_count})')
print(f'[OK] Ground truth: {image_count} images, {box_count} boxes')

METHOD = 'dots_ocr'
DATASET = 'doclaynet_test'
SPLIT = 'test'
IOU_THRESHOLD = 0.5
LIMIT = 0
BATCH_SIZE = 8
FORCE_INFERENCE = True
FIGURE_LIMIT = 1
PREDICTIONS_DIR = ROOT / 'outputs' / 'predictions' / METHOD / DATASET
FIGURES_DIR = ROOT / 'outputs' / 'reports' / f'{METHOD}_{DATASET}'
print(f'Project: {ROOT}')
print(f'Data: {data_root / "processed"}')
print(f'Pipeline: {METHOD} | {DATASET}/{SPLIT} | IoU >= {IOU_THRESHOLD}')

# %% [markdown]
# ## 0. Kiểm tra mã nguồn, môi trường và tự động tải trọng số (Setup & Auto-Download)
## Tế bào này kiểm tra sự tồn tại của thư viện phụ thuộc (`transformers`, `qwen-vl-utils`), bộ trọng số chính thức của `dots.ocr` từ Hugging Face Hub (`dots-studio/dots.ocr`), và tập dữ liệu. Nếu chưa có file trọng số, hệ thống sẽ tự động tải về thư mục `methods/dots_ocr/weights/` để người dùng có thể chạy trực tiếp toàn bộ notebook.
# %%
# Kaggle runtime dependencies must already be available in the image.
# This cell intentionally performs no pip/network installation.
required = ['torch', 'transformers', 'qwen_vl_utils', 'accelerate', 'pycocotools', 'PIL', 'cv2']
missing = []
for name in required:
    try:
        __import__(name)
    except ImportError:
        missing.append(name)
if missing:
    raise ImportError(f'Missing Kaggle runtime packages: {missing}. Attach a Linux wheelhouse or use a Kaggle image that includes them.')
weights_dir = ROOT / 'methods' / 'dots_ocr' / 'weights'
print(f'Model weights will be validated/downloaded by run_inference.py: {weights_dir}')
images_dir = ROOT / 'datasets' / DATASET / 'processed' / 'images'
print(f'[OK] Found {len(list(images_dir.glob("*.png")))} images')

# %% [markdown]
# ## 1. Prediction
## Chạy layout detection bằng script `methods/dots_ocr/src/run_inference.py`. Script nạp model Qwen2-VL 1.7B, đưa prompt bóc tách layout và sinh kết quả dự đoán ra `outputs/predictions/dots_ocr/<dataset>/`.
# %%
inference_script = ROOT / "methods" / METHOD / "src" / "run_inference.py"
cmd = [
    sys.executable,
    "-u",
    str(inference_script),
    "--dataset", DATASET,
    "--limit", str(LIMIT),
    "--batch-size", str(BATCH_SIZE),
]
if not FORCE_INFERENCE:
    cmd.append("--skip-existing")
else:
    cmd.append("--force")

print(f"Running: {' '.join(cmd)}\n", flush=True)

process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    cwd=str(ROOT),
    encoding="utf-8",
    errors="replace",
    bufsize=1,
)

for line in process.stdout:
    print(line, end="", flush=True)

process.wait()
if process.returncode != 0:
    raise RuntimeError(f"Inference process exited with code {process.returncode}; see the log above.")

# %% [markdown]
# ## 2. Metrics and Evaluation Chart
## Đánh giá các dự đoán layout theo chuẩn COCO và benchmark DocLayNet:
# - **Strict**: Yêu cầu khớp cả tọa độ (IoU >= threshold) và đúng nhãn layout.
# - **Loose**: Chỉ đánh giá khả năng định vị (localization), bỏ qua nhãn.
# - **COCO mAP**: Tính toán mAP@[0.5:0.95] và mAP@0.5.
# - **Biểu đồ**: Trực quan hóa Precision, Recall, F1 theo từng class và so sánh Strict vs Loose.
# %%
eval_script = ROOT / "scripts" / "evaluate_predictions.py"
chart_path = FIGURES_DIR / "metrics_chart.png"
chart_path.parent.mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable,
    "-u",
    str(eval_script),
    "--method", METHOD,
    "--dataset", DATASET,
    "--split", SPLIT,
    "--iou-threshold", str(IOU_THRESHOLD),
    "--allow-incomplete",
    "--loose",
    "--plot", str(chart_path),
]

print("Running evaluation:\n", flush=True)

process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    cwd=str(ROOT),
    encoding="utf-8",
    errors="replace",
    bufsize=1,
)

for line in process.stdout:
    print(line, end="", flush=True)

process.wait()
if process.returncode != 0:
    raise RuntimeError(f"Evaluation process exited with code {process.returncode}.")

if chart_path.exists():
    from IPython.display import Image, display
    print(f"\nEvaluation Chart ({chart_path}):")
    display(Image(filename=str(chart_path)))

# %% [markdown]
# ## 3. Ground Truth vs. Prediction Examples
## Vẽ bounding box Ground Truth (bên trái) đối chiếu với dự đoán của dots.ocr (bên phải) cạnh nhau trên cùng một ảnh để trực quan hóa chất lượng nhận diện layout.
# %%
viz_script = ROOT / "visualization" / "scripts" / "visualize_predictions.py"
cmd = [
    sys.executable,
    "-u",
    str(viz_script),
    "--method", METHOD,
    "--dataset", DATASET,
    "--split", SPLIT,
    "--limit", str(FIGURE_LIMIT),
    "--output-dir", str(FIGURES_DIR),
]

print(f"Running: {' '.join(cmd)}\n", flush=True)

process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    cwd=str(ROOT),
    encoding="utf-8",
    errors="replace",
    bufsize=1,
)

for line in process.stdout:
    print(line, end="", flush=True)

process.wait()

from IPython.display import Image, display

fig_files = sorted(FIGURES_DIR.glob("*.png"))
sample_files = [f for f in fig_files if f.name != "metrics_chart.png"][:4]
print(f"\nHiển thị {len(sample_files)} ảnh mẫu (Ground Truth vs {METHOD}):")
for f in sample_files:
    print(f"Ảnh: {f.name}")
    display(Image(filename=str(f)))
