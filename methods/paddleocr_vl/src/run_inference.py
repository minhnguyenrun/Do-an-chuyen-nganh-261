"""Run the layout stage of the PaddleOCR-VL pipeline on a processed dataset.

PaddleOCR-VL = layout detector (PP-DocLayoutV2) + recognition VLM (PaddleOCR-VL-0.9B).
Only the layout stage produces bounding boxes, so this script runs exactly that stage:
the layout model, hyper-parameters (per-class thresholds, NMS, unclip ratio, merge modes)
and the overlap filter are all taken from the official PaddleOCR-VL pipeline config shipped
with PaddleX (``configs/pipelines/PaddleOCR-VL.yaml``), without running the recognition VLM.
"""

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
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.common.profiling import ResourceMonitor, save_runtime_summary  # noqa: E402
from src.common.splits import apply_split, describe, SPLITS  # noqa: E402

# Official pipeline configs (PaddleX) whose LayoutDetection sub-module is used.
# "PaddleOCR-VL" is the original release (PP-DocLayoutV2); newer releases use PP-DocLayoutV3.
DEFAULT_PIPELINE = "PaddleOCR-VL"

# Bump when PADDLE_TO_DOCLAYNET / drop rules change: predictions made with an older mapping are redone.
MAPPING_VERSION = "v2"

# Mapping from PP-DocLayoutV2/V3 labels to DocLayNet 11 categories
PADDLE_TO_DOCLAYNET = {
    # 1: Caption (figure / table / chart captions). PP-DocLayoutV2's "vision_footnote" is the text attached to
    # a figure / table, which DocLayNet annotates as Caption (chosen on the validation split, see README).
    "figure_title": 1,
    "vision_footnote": 1,
    # 2: Footnote (page footnotes)
    "footnote": 2,
    # 3: Formula (displayed equations; inline formulas stay part of the Text block)
    "display_formula": 3,
    "formula": 3,
    # 4: List-item (PP-DocLayout does not separate lists; they are detected as text)
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
    # 9: Table
    "table": 9,
    # 10: Text (body text, paragraphs, abstract, bibliography, margin notes, code/algorithms)
    "text": 10,
    "vertical_text": 10,
    "abstract": 10,
    "content": 10,
    "reference": 10,
    "reference_content": 10,
    "aside_text": 10,
    "algorithm": 10,
    # 11: Title (document / report title)
    "doc_title": 11,
}

# Mapping from PaddleOCR labels to PubLayNet 5 categories
PADDLE_TO_PUBLAYNET = {
    # 1: Text
    "text": 1,
    "vertical_text": 1,
    "abstract": 1,
    "content": 1,
    "reference": 1,
    "reference_content": 1,
    "aside_text": 1,
    "algorithm": 1,
    "footnote": 1,
    "vision_footnote": 1,
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
}

# Labels without natural equivalent in standard layout benchmarks (DocLayNet/PubLayNet)
# Dropping these eliminates spurious false positives without reducing true positives.
DEFAULT_DROP_LABELS = {
    "seal",             # Official Chinese stamp/seal (not present in DocLayNet/PubLayNet)
    "formula_number",   # Standalone formula number tag (often redundant with equation)
    "inline_formula",   # Inline math is part of a text block, not a layout element
}

# "number" = standalone page number. DocLayNet annotates page numbers as Page-footer / Page-header, so it is
# mapped by vertical position (lower half -> Page-footer, upper half -> Page-header). Selected on the
# validation split: Page-footer F1 0.29 -> 0.60 and micro F1 +1.4 points versus dropping these boxes.
PAGE_NUMBER_LABEL = "number"


def get_mapping(dataset: str) -> dict[str, int]:
    if dataset.lower().startswith("publaynet"):
        return PADDLE_TO_PUBLAYNET
    return PADDLE_TO_DOCLAYNET


def load_layout_config(pipeline: str = DEFAULT_PIPELINE) -> dict:
    """Layout-detection settings of an official PaddleX pipeline config (model, thresholds, NMS, ...)."""
    import paddlex
    import yaml

    config_path = Path(paddlex.__file__).resolve().parent / "configs" / "pipelines" / f"{pipeline}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy pipeline config '{pipeline}' của PaddleX tại {config_path}. "
            "Hãy nâng cấp paddleocr/paddlex hoặc chọn --pipeline khác."
        )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return config["SubModules"]["LayoutDetection"]


def build_detector(layout_config: dict, threshold: float | None = None) -> LayoutDetection:
    """Build the PaddleOCR LayoutDetection predictor with the pipeline's layout settings."""
    kwargs = {
        "model_name": layout_config["model_name"],
        "threshold": threshold if threshold is not None else layout_config.get("threshold"),
        "layout_nms": layout_config.get("layout_nms"),
        "layout_unclip_ratio": layout_config.get("layout_unclip_ratio"),
        "layout_merge_bboxes_mode": layout_config.get("layout_merge_bboxes_mode"),
        "enable_mkldnn": False,  # Avoid Paddle 3.x PIR oneDNN instruction bug on CPU
    }
    return LayoutDetection(**{k: v for k, v in kwargs.items() if v is not None})


def filter_overlap_boxes(layout_det_res: dict) -> dict:
    """Overlap filter applied by the PaddleOCR-VL pipeline right after layout detection."""
    from paddlex.inference.pipelines.paddleocr_vl.uilts import filter_overlap_boxes as pipeline_filter

    return pipeline_filter(layout_det_res, "auto")


def run_on_image(
    detector: LayoutDetection,
    image_path: Path,
    mapping: dict[str, int],
    drop_labels: set[str],
    layout_model: str,
    page_number_as_furniture: bool = True,
) -> dict:
    with Image.open(image_path) as img:
        width, height = img.size

    results = detector.predict(str(image_path))

    detections = []
    for res in results:
        res = filter_overlap_boxes(res if isinstance(res, dict) else dict(res))
        for box in res.get("boxes", []):
            label = box["label"]
            if label in drop_labels:
                continue

            coord = box["coordinate"]
            xmin, ymin, xmax, ymax = float(coord[0]), float(coord[1]), float(coord[2]), float(coord[3])
            bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
            score = float(box.get("score", 1.0))

            if label == PAGE_NUMBER_LABEL and page_number_as_furniture:
                category_id = mapping.get("footer" if (ymin + ymax) / 2 > height / 2 else "header")
            else:
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
        "layout_model": layout_model,
        "mapping_version": MAPPING_VERSION,
        "detections": detections,
    }


def paddle_device_info() -> dict:
    """Device label and peak VRAM measured through Paddle (the profiler only knows about torch)."""
    try:
        import paddle

        device = paddle.device.get_device()
        info = {"device": device}
        if device.startswith("gpu"):
            info["device"] = f"{device} ({paddle.device.cuda.get_device_name()})"
            info["peak_vram_mb"] = paddle.device.cuda.max_memory_allocated() / 2**20
        return info
    except Exception:
        return {}


def is_current_prediction(path: Path, layout_model: str) -> bool:
    """Prediction produced by the current layout model (older PP-DocLayout_plus-L files are redone)."""
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return (
            data.get("layout_model") == layout_model
            and data.get("mapping_version") == MAPPING_VERSION
            and "inference_time_s" in data
        )
    except (OSError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="doclaynet", help="Dataset name (doclaynet or publaynet)")
    parser.add_argument(
        "--split",
        default=None,
        choices=sorted(SPLITS) + ["val", "validation"],
        help="Tap du lieu DocLayNet: train / valid / test (thay cho --dataset doclaynet / doclaynet_val / doclaynet_test).",
    )
    parser.add_argument(
        "--pipeline",
        default=DEFAULT_PIPELINE,
        help="Official PaddleX pipeline whose layout settings are used "
             "(PaddleOCR-VL -> PP-DocLayoutV2; PaddleOCR-VL-1.5 / -1.6 -> PP-DocLayoutV3).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the pipeline's per-class score thresholds with one global threshold.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of images; 0 processes the full dataset (default).",
    )
    parser.add_argument(
        "--no-page-number-mapping",
        action="store_true",
        help="Drop 'number' boxes instead of mapping them to Page-footer / Page-header by position.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Custom output predictions directory (default: outputs/predictions/paddleocr_vl/<dataset>).",
    )
    parser.add_argument(
        "--keep-all",
        action="store_true",
        help="Keep all labels including unmapped ones (e.g., seal, formula_number) as category_id=None.",
    )
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        help="Skip images that already have a prediction from the current layout model (default).",
    )
    run_mode.add_argument(
        "--force",
        dest="skip_existing",
        action="store_false",
        help="Run inference again and overwrite existing predictions.",
    )
    parser.set_defaults(skip_existing=True)
    args = parser.parse_args()
    args.dataset = apply_split(args.dataset, args.split)
    print(f"[paddleocr_vl] Tap du lieu: {describe(args.dataset)}", flush=True)

    images_dir = ROOT_DIR / "datasets" / args.dataset / "processed" / "images"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT_DIR / "outputs" / "predictions" / "paddleocr_vl" / args.dataset
    )
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Thư mục ảnh không tồn tại: {images_dir}")

    layout_config = load_layout_config(args.pipeline)
    layout_model = layout_config["model_name"]

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
            path for path in image_paths
            if not is_current_prediction(output_dir / f"{path.stem}.json", layout_model)
        ]

    mapping = get_mapping(args.dataset)
    drop_labels = set() if args.keep_all else set(DEFAULT_DROP_LABELS)
    page_number_as_furniture = not args.no_page_number_mapping and not args.dataset.lower().startswith("publaynet")
    if not page_number_as_furniture and not args.keep_all:
        drop_labels.add(PAGE_NUMBER_LABEL)

    print(
        f"[paddleocr_vl] {len(image_paths)} image(s) to predict from {images_dir} "
        f"(pipeline: {args.pipeline}, layout model: {layout_model}, "
        f"threshold: {args.threshold if args.threshold is not None else 'per-class from pipeline config'})"
        f"{' (existing predictions skipped)' if args.skip_existing else ' (force overwrite)'}"
    )
    print(
        f"[paddleocr_vl] Mapping strategy: {len(mapping)} mapped labels, "
        f"{'no labels dropped (--keep-all)' if args.keep_all else f'{len(drop_labels)} dropped labels: ' + ', '.join(sorted(drop_labels))}"
        f"{'; page numbers -> Page-footer/Page-header by position' if page_number_as_furniture else ''}"
    )

    if not image_paths:
        print(f"[paddleocr_vl] No work to do. Predictions are in {output_dir}")
        return

    monitor = ResourceMonitor()
    with monitor.time_model_load():
        detector = build_detector(layout_config, threshold=args.threshold)
    print(f"[paddleocr_vl] Layout model {layout_model} ready in {monitor.model_load_seconds:.2f}s", flush=True)

    total = len(image_paths)
    for index, image_path in enumerate(image_paths, start=1):
        with monitor.time_page() as timer:
            prediction = run_on_image(
                detector, image_path, mapping, drop_labels, layout_model, page_number_as_furniture
            )
        prediction["inference_time_s"] = round(timer.seconds, 4)

        output_path = output_dir / f"{image_path.stem}.json"
        output_path.write_text(
            json.dumps(prediction, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if index <= 20 or index % 100 == 0 or index == total:
            print(
                f"  [{index}/{total}] {image_path.name}: "
                f"{len(prediction['detections'])} boxes ({timer.seconds:.2f}s)",
                flush=True,
            )

    summary = monitor.summary(dataset=args.dataset, **paddle_device_info())
    runtime_file = save_runtime_summary("paddleocr_vl", args.dataset, summary)
    print(
        f"[paddleocr_vl] Done. Mean {summary['mean_s_per_page']:.3f}s/page "
        f"(median {summary['median_s_per_page']:.3f}s, p95 {summary['p95_s_per_page']:.3f}s) | "
        f"peak RAM {summary['peak_ram_mb']:.0f} MB. Predictions: {output_dir}",
        flush=True,
    )
    print(f"[paddleocr_vl] Runtime summary: {runtime_file}", flush=True)


if __name__ == "__main__":
    main()
