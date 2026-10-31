"""Build the Kaggle source-only dataset payload for dots.ocr."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "source_dataset"
INCLUDE = [
    "src",
    "scripts",
    "visualization",
    "methods/dots_ocr/src",
    "methods/dots_ocr/requirements.txt",
    "methods/dots_ocr/README.md",
    "methods/dots_ocr/notebooks/dots_ocr_report.py",
    "methods/dots_ocr/weights/README.md",
    "methods/dots_ocr/weights/.gitkeep",
    "kaggle/dots_ocr/kaggle-requirements.txt",
    "kaggle/dots_ocr/wheelhouse",
    "datasets/README.md",
    "README.md",
]


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for rel in INCLUDE:
        src = ROOT / rel
        dst = OUT / rel
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))
        elif src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    (OUT / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "DO-AN-261 Dots OCR Source",
                "id": "hongnguynngthhjk/doan261-dots-ocr-source",
                "licenses": [{"name": "other"}],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (OUT / "KAGGLE_README.md").write_text(
        "# DO-AN-261 dots.ocr source\n\n"
        "Source-only bundle for Kaggle. It contains no datasets, predictions, or model weights.\n"
        "Use the attached DocLayNet input dataset and enable the required Kaggle runtime packages.\n"
        "The model weights are fetched from Hugging Face by the inference script when internet is enabled.\n",
        encoding="utf-8",
    )
    print(f"Prepared {OUT}")


if __name__ == "__main__":
    main()
