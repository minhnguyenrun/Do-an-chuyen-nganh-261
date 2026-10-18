"""Run Docling layout detection on a processed dataset."""

import argparse
import json
import sys
import time
from pathlib import Path

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

from docling.datamodel.base_models import DocItemLabel, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, ImageFormatOption

ROOT_DIR = Path(__file__).resolve().parents[3]

LABEL_TO_DOCLAYNET_ID = {
    "caption": 1,
    "footnote": 2,
    "formula": 3,
    "list_item": 4,
    "page_footer": 5,
    "page_header": 6,
    "picture": 7,
    "section_header": 8,
    "table": 9,
    "text": 10,
    "title": 11,
}


def build_converter() -> DocumentConverter:
    options = PdfPipelineOptions()
    options.do_ocr = False
    options.do_table_structure = False
    options.do_picture_classification = False
    options.do_picture_description = False
    options.do_chart_extraction = False
    options.do_code_enrichment = False
    options.do_formula_enrichment = False
    options.generate_page_images = False
    # Keep layout clusters even when they have no OCR text.
    options.layout_options.keep_empty_clusters = True

    return DocumentConverter(
        allowed_formats=[InputFormat.IMAGE],
        format_options={
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=options)
        },
    )


def run_on_image(converter: DocumentConverter, image_path: Path) -> dict:
    document = converter.convert(str(image_path)).document
    page = document.pages[1]
    width, height = page.size.width, page.size.height

    # Lấy box từ page.assembled.elements thay vì doc.iterate_items():
    #  - iterate_items() mặc định bỏ qua content layer "furniture", tức là làm
    #    mất toàn bộ page_header/page_footer mà model vẫn phát hiện được.
    #  - mỗi element giữ cluster gốc của model layout, có confidence -> dùng
    #    làm "score" để tính mAP kiểu COCO.
    detections = []
    for item, _ in document.iterate_items():
        if not item.prov:
            continue

        label = item.label.value
        bbox = item.prov[0].bbox.to_top_left_origin(height)
        detections.append(
            {
                "category_id": LABEL_TO_DOCLAYNET_ID.get(label),
                "category_name": label,
                "bbox": [bbox.l, bbox.t, bbox.r - bbox.l, bbox.b - bbox.t],
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
    parser.add_argument("--dataset", default="doclaynet")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of images; 0 processes the full dataset (default).",
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
    output_dir = ROOT_DIR / "outputs" / "predictions" / "docling" / args.dataset
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

    print(
        f"[docling] {len(image_paths)} image(s) to predict from {images_dir}"
        f"{' (existing predictions skipped)' if args.skip_existing else ' (force overwrite)'}"
    )
    if not image_paths:
        print(f"[docling] No work to do. Predictions are in {output_dir}")
        return

    print(f"[docling] Available Docling labels: {', '.join(label.value for label in DocItemLabel)}")
    converter = build_converter()
    total_seconds = 0.0

    for index, image_path in enumerate(image_paths, start=1):
        started = time.time()
        prediction = run_on_image(converter, image_path)
        elapsed = time.time() - started
        total_seconds += elapsed

        output_path = output_dir / f"{image_path.stem}.json"
        output_path.write_text(
            json.dumps(prediction, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"  [{index}/{len(image_paths)}] {image_path.name}: "
            f"{len(prediction['detections'])} boxes ({elapsed:.2f}s)"
        )

    print(
        f"[docling] Done. Average {total_seconds / len(image_paths):.2f}s/image. "
        f"Predictions: {output_dir}"
    )


if __name__ == "__main__":
    main()
