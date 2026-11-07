"""Convert the dots.ocr report notebook to Jupytext percent format for Kaggle."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "methods/dots_ocr/notebooks/dots_ocr_report.ipynb"
OUTPUT = NOTEBOOK.with_suffix(".py")


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sections: list[str] = [
        "# /// script\n# jupytext:\n#   formats: ipynb,py:percent\n#   text_representation:\n#     extension: .py\n#     format_name: percent\n#     format_version: '1.3'\n#     jupytext_version: 1.16.4\n# ///\n",
        "# %% [markdown]\n# # dots.ocr — Kaggle report\n#\n# This is a Jupytext percent-format notebook. Attach the source bundle and a\n# DocLayNet dataset as Kaggle inputs. No dataset files are included here.\n",
        """# %%
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
""",
    ]

    skipped_config = False
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown":
            # Avoid duplicating the original introductory/setup prose.
            if src.lstrip().startswith("# dots.ocr"):
                continue
            sections.append("\n# %% [markdown]\n" + "".join(f"# {line}" if line.strip() else "#" for line in src.splitlines(keepends=True)))
            continue
        if not skipped_config and "ROOT = Path.cwd().resolve()" in src:
            skipped_config = True
            continue
        if "check_call" in src and "pip" in src and "install" in src:
            src = (
                "# Kaggle runtime dependencies must already be available in the image.\n"
                "# This cell intentionally performs no pip/network installation.\n"
                "required = ['torch', 'transformers', 'qwen_vl_utils', 'accelerate', 'pycocotools', 'PIL', 'cv2']\n"
                "missing = []\n"
                "for name in required:\n"
                "    try:\n"
                "        __import__(name)\n"
                "    except ImportError:\n"
                "        missing.append(name)\n"
                "if missing:\n"
                "    raise ImportError(f'Missing Kaggle runtime packages: {missing}. Attach a Linux wheelhouse or use a Kaggle image that includes them.')\n"
                "weights_dir = ROOT / 'methods' / 'dots_ocr' / 'weights'\n"
                "print(f'Model weights will be validated/downloaded by run_inference.py: {weights_dir}')\n"
                "images_dir = ROOT / 'datasets' / DATASET / 'processed' / 'images'\n"
                "print(f'[OK] Found {len(list(images_dir.glob(\"*.png\")))} images')\n"
            )
        if "inference_script" in src:
            src = src.replace(
                '    "--limit", str(LIMIT),\n]',
                '    "--limit", str(LIMIT),\n    "--batch-size", str(BATCH_SIZE),\n]',
            )
        if "process.wait()" in src and "process.returncode" in src:
            src = src.replace(
                'if process.returncode != 0:\n    print(f"\\n[Error] Inference process exited with code {process.returncode}", flush=True)',
                'if process.returncode != 0:\n    raise RuntimeError(f"Inference process exited with code {process.returncode}; see the log above.")',
            )
            src = src.replace(
                'process.wait()\n\nif chart_path.exists():',
                'process.wait()\nif process.returncode != 0:\n    raise RuntimeError(f"Evaluation process exited with code {process.returncode}.")\n\nif chart_path.exists():',
            )
        if "process.wait()" in src and "chart_path.exists()" in src:
            src = src.replace(
                "process.wait()\n\nif chart_path.exists():",
                'process.wait()\nif process.returncode != 0:\n    raise RuntimeError(f"Evaluation process exited with code {process.returncode}.")\n\nif chart_path.exists():',
            )
        sections.append("\n# %%\n" + src.rstrip() + "\n")

    OUTPUT.write_text("".join(sections), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
