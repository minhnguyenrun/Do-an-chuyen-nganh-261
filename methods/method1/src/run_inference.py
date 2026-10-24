"""Inference script for Method 1: YOLOv8n BBox Detector + Multimodal Transformer Classifier.

Scans images from datasets/<dataset>/processed/images and writes prediction
JSON files to outputs/predictions/method1/<dataset>/<image_id>.json in the standardized format.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

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

import torch

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from methods.method1.src.pipeline_b.pipeline import DocumentLayoutFullPipeline
from src.common.profiling import ResourceMonitor, save_runtime_summary


def is_current_prediction(path: Path) -> bool:
    """Kiểm tra file prediction có hợp lệ và đầy đủ trường dữ liệu chuẩn hay không."""
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return (
            "image_id" in data
            and "detections" in data
            and "inference_time_s" in data
        )
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="doclaynet_test",
        help="Dataset name (e.g. doclaynet_test, doclaynet, publaynet)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of images; 0 processes the full dataset (default: 0).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for bounding box detection (default: 0.25).",
    )
    parser.add_argument(
        "--detector-weights",
        type=str,
        default=None,
        help="Path to YOLO detector weights (default: methods/method1/weights/yolov8n_doclaynet_10k_best.pt)",
    )
    parser.add_argument(
        "--classifier-weights",
        type=str,
        default=None,
        help="Path to Transformer classifier weights (default: methods/method1/weights/pipeline_b_transformer_10k_best.pth)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for predictions.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda / cpu).",
    )
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        help="Skip images that already have a valid prediction (default).",
    )
    run_mode.add_argument(
        "--force",
        dest="skip_existing",
        action="store_false",
        help="Run inference again and overwrite existing predictions.",
    )
    parser.set_defaults(skip_existing=True)

    args = parser.parse_args()

    # Xác định đường dẫn weights
    weights_dir = ROOT_DIR / "methods" / "method1" / "weights"
    detector_weights = Path(args.detector_weights) if args.detector_weights else (weights_dir / "yolov8n_doclaynet_10k_best.pt")
    if not detector_weights.is_file():
        detector_weights = weights_dir / "yolov8n_doclaynet_bbox_only_best.pt"

    classifier_weights = Path(args.classifier_weights) if args.classifier_weights else (weights_dir / "pipeline_b_transformer_10k_best.pth")
    if not classifier_weights.is_file():
        classifier_weights = weights_dir / "pipeline_b_transformer_best.pth"

    images_dir = ROOT_DIR / "datasets" / args.dataset / "processed" / "images"
    if not images_dir.is_dir():
        # Thử dataset không có hậu tố _test nếu cần
        images_dir = ROOT_DIR / "datasets" / args.dataset / "images"
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Thư mục ảnh không tồn tại: {images_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else (ROOT_DIR / "outputs" / "predictions" / "method1" / args.dataset)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".png", ".jpg", ".jpeg"}
    image_paths = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in image_extensions],
        key=lambda path: (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem),
    )

    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    if args.skip_existing:
        image_paths = [
            p for p in image_paths if not is_current_prediction(output_dir / f"{p.stem}.json")
        ]

    print(
        f"[method1] {len(image_paths)} image(s) to predict from {images_dir}\n"
        f"  Detector: {detector_weights.name} | Classifier: {classifier_weights.name}\n"
        f"  Device: {args.device} | Conf: {args.conf} | Mode: {'skip-existing' if args.skip_existing else 'force-overwrite'}",
        flush=True,
    )

    if not image_paths:
        print(f"[method1] Không có ảnh nào cần xử lý. File dự đoán tại: {output_dir}", flush=True)
        return

    # Khởi tạo mô hình và đo runtime model loading
    monitor = ResourceMonitor()
    with monitor.time_model_load():
        pipeline = DocumentLayoutFullPipeline(
            detector_weights=detector_weights,
            classifier_weights=classifier_weights,
            device=args.device,
            num_classes=11,
        )
    print(f"[method1] Pipeline đã sẵn sàng trong {monitor.model_load_seconds:.2f}s!", flush=True)

    failed = []
    total_images = len(image_paths)

    for idx, img_path in enumerate(image_paths, start=1):
        try:
            with monitor.time_page() as timer:
                res = pipeline.predict_page(img_path, conf_thresh=args.conf)
        except Exception as exc:
            monitor.page_seconds.pop()
            failed.append(img_path.name)
            print(f"  [{idx}/{total_images}] {img_path.name}: FAILED ({type(exc).__name__}: {exc})", flush=True)
            continue

        elapsed = timer.seconds
        image_id = int(img_path.stem) if img_path.stem.isdigit() else img_path.stem

        record = {
            "image_id": image_id,
            "file_name": img_path.name,
            "width": res.get("width", 0),
            "height": res.get("height", 0),
            "inference_time_s": elapsed,
            "det_time_s": res.get("det_time_s", 0.0),
            "cls_time_s": res.get("cls_time_s", 0.0),
            "detections": res.get("detections", []),
        }

        out_path = output_dir / f"{img_path.stem}.json"
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

        if idx % 100 == 0 or idx == total_images or idx <= 10:
            print(
                f"  [{idx}/{total_images}] {img_path.name}: "
                f"{len(record['detections'])} boxes ({elapsed:.3f}s)",
                flush=True,
            )

    if monitor.page_seconds:
        summary = monitor.summary(dataset=args.dataset)
        runtime_file = save_runtime_summary("method1", args.dataset, summary)
        vram = summary.get("peak_vram_mb")
        print(
            f"[method1] Hoàn tất. Mean {summary['mean_s_per_page']:.3f}s/trang "
            f"(median {summary['median_s_per_page']:.3f}s, p95 {summary['p95_s_per_page']:.3f}s) | "
            f"peak RAM {summary.get('peak_ram_mb', 0):.0f} MB | "
            f"peak VRAM {f'{vram:.0f} MB' if vram is not None else 'n/a'} | {summary.get('device', 'cpu')}",
            flush=True,
        )
        print(f"[method1] Runtime summary: {runtime_file}", flush=True)

    print(f"[method1] Predictions hoàn tất tại: {output_dir}", flush=True)
    if failed:
        print(f"[method1] Cảnh báo: {len(failed)} ảnh bị lỗi trong quá trình suy luận.", flush=True)


if __name__ == "__main__":
    main()
