"""Run PaddleOCR / PaddleX layout detection on a processed dataset."""

import argparse
import json
import sys
import time
from pathlib import Path
from PIL import Image

# Fix Windows console UTF-8 encoding
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Import torch first on Windows to avoid DLL conflicts (WinError 127 shm.dll) with modelscope/torch
try:
    import torch  # noqa: F401
except ImportError:
    pass

from paddleocr import LayoutDetection

ROOT_DIR = Path(__file__).resolve().parents[3]

# Mapping from PaddleOCR (PP-DocLayout / PP-Structure) labels to DocLayNet 11 categories
PADDLE_TO_DOCLAYNET = {
    # 1: Caption (figure captions, table captions, chart captions)
    "figure_title": 1,
    "table_title": 1,
    "chart_title": 1,
    # 2: Footnote
    "footnote": 2,
    # 3: Formula (displayed or inline mathematical equations)
    "formula": 3,
    # 4: List-item (PP-DocLayout does not separate lists; handled via text/content)
    # 5: Page-footer (running footers)
    "footer": 5,
    "footer_image": 5,
    # 6: Page-header (running headers)
    "header": 6,
    "header_image": 6,
    # 7: Picture (images, figures, charts, diagrams)
    "image": 7,
    "chart": 7,
    # 8: Section-header (headings, paragraph titles, subsection titles)
    "paragraph_title": 8,
    # 9: Table (tables, tabular layouts)
    "table": 9,
    # 10: Text (body text, paragraphs, abstract, bibliography, margin notes, code/algorithms)
    "text": 10,
    "abstract": 10,
    "content": 10,
    "reference": 10,
    "reference_content": 10,
    "aside_text": 10,
    "algorithm": 10,
    # 11: Title (document/report title)
    "doc_title": 11,
}

# Mapping from PaddleOCR labels to PubLayNet 5 categories
PADDLE_TO_PUBLAYNET = {
    # 1: Text
    "text": 1,
    "abstract": 1,
    "content": 1,
    "reference": 1,
    "reference_content": 1,
    "aside_text": 1,
    "algorithm": 1,
    "footnote": 1,
    "header": 1,
    "footer": 1,
    # 2: Title
    "doc_title": 2,
    "paragraph_title": 2,
    # 3: List (PubLayNet list block)
    # 4: Table
    "table": 4,
    # 5: Figure
    "image": 5,
    "chart": 5,
    "figure_title": 5,
    "table_title": 5,
    "chart_title": 5,
}

# Labels without natural equivalent in standard layout benchmarks (DocLayNet/PubLayNet)
# Dropping these eliminates spurious false positives without reducing true positives.
DEFAULT_DROP_LABELS = {
    "seal",             # Official Chinese stamp/seal (not present in DocLayNet/PubLayNet)
    "formula_number",   # Standalone formula number tag (often redundant with equation)
    "number",           # Standalone index/number (creates high FP rate on DocLayNet)
}


def get_mapping(dataset: str) -> dict[str, int]:
    if dataset.lower() == "doclaynet":
        return PADDLE_TO_DOCLAYNET
    elif dataset.lower() == "publaynet":
        return PADDLE_TO_PUBLAYNET
    else:
        return PADDLE_TO_DOCLAYNET


def build_detector(model_name: str = "PP-DocLayout_plus-L", threshold: float = 0.3) -> LayoutDetection:
    """Build PaddleOCR LayoutDetection predictor with CPU oneDNN bug workaround."""
    return LayoutDetection(
        model_name=model_name,
        threshold=threshold,
        enable_mkldnn=False,  # Avoid Paddle 3.x PIR oneDNN instruction bug on CPU
    )


def run_on_image(
    detector: LayoutDetection,
    image_path: Path,
    mapping: dict[str, int],
    drop_labels: set[str],
) -> dict:
    with Image.open(image_path) as img:
        width, height = img.size

    results = detector.predict(str(image_path))

    detections = []
    for res in results:
        boxes = res.get("boxes", []) if isinstance(res, dict) else getattr(res, "boxes", [])
        for box in boxes:
            label = box["label"]
            if label in drop_labels:
                continue

            coord = box["coordinate"]
            xmin, ymin, xmax, ymax = float(coord[0]), float(coord[1]), float(coord[2]), float(coord[3])
            bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
            score = float(box.get("score", 1.0))
            category_id = mapping.get(label)

            detections.append(
                {
                    "category_id": category_id,
                    "category_name": label,
                    "bbox": bbox,
                    "score": round(score, 4),
                }
            )

    image_id = int(image_path.stem) if image_path.stem.isdigit() else image_path.stem
    return {
        "image_id": image_id,
        "file_name": image_path.name,
        "width": width,
        "height": height,
        "detections": detections,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="doclaynet", help="Dataset name (doclaynet or publaynet)")
    parser.add_argument(
        "--model",
        default="PP-DocLayout_plus-L",
        help="PaddleX layout model name (default: PP-DocLayout_plus-L)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Confidence score threshold (default: 0.3)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of images; 0 processes the full dataset (default).",
    )
    parser.add_argument(
        "--keep-all",
        action="store_true",
        help="Keep all labels including unmapped ones (e.g., seal, number, formula_number) as category_id=None.",
    )
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        help="Skip images that already have a prediction (default).",
    )
    run_mode.add_argument(
        "--force",
        dest="skip_existing",
        action="store_false",
        help="Run inference again and overwrite existing predictions.",
    )
    parser.set_defaults(skip_existing=True)
    args = parser.parse_args()

    images_dir = ROOT_DIR / "datasets" / args.dataset / "processed" / "images"
    output_dir = ROOT_DIR / "outputs" / "predictions" / "paddleocr_vl" / args.dataset
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Thư mục ảnh không tồn tại: {images_dir}")

    image_paths = sorted(
        images_dir.glob("*.png"),
        key=lambda path: (
            0,
            int(path.stem),
        )
        if path.stem.isdigit()
        else (1, path.stem),
    )
    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_existing:
        image_paths = [
            path for path in image_paths if not (output_dir / f"{path.stem}.json").exists()
        ]

    mapping = get_mapping(args.dataset)
    drop_labels = set() if args.keep_all else DEFAULT_DROP_LABELS

    print(
        f"[paddleocr_vl] {len(image_paths)} image(s) to predict from {images_dir} "
        f"(model: {args.model}, threshold: {args.threshold})"
        f"{' (existing predictions skipped)' if args.skip_existing else ' (force overwrite)'}"
    )
    print(
        f"[paddleocr_vl] Mapping strategy: {len(mapping)} mapped labels, "
        f"{'no labels dropped (--keep-all)' if args.keep_all else f'{len(drop_labels)} dropped labels: ' + ', '.join(sorted(drop_labels))}"
    )

    if not image_paths:
        print(f"[paddleocr_vl] No work to do. Predictions are in {output_dir}")
        return

    detector = build_detector(model_name=args.model, threshold=args.threshold)
    total_seconds = 0.0

    for index, image_path in enumerate(image_paths, start=1):
        started = time.time()
        prediction = run_on_image(detector, image_path, mapping, drop_labels)
        elapsed = time.time() - started
        total_seconds += elapsed

        output_path = output_dir / f"{image_path.stem}.json"
        output_path.write_text(
            json.dumps(prediction, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"  [{index}/{len(image_paths)}] {image_path.name}: "
            f"{len(prediction['detections'])} boxes ({elapsed:.2f}s)"
        )

    print(
        f"[paddleocr_vl] Done. Average {total_seconds / len(image_paths):.2f}s/image. "
        f"Predictions: {output_dir}"
    )


if __name__ == "__main__":
    main()
