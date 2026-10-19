"""Inference script for M2Doc (Multi-Modal Document Layout Analysis).

Paper: "M2Doc: A Multi-Modal Fusion Approach for Document Layout Analysis" (AAAI 2024)
Scans images from datasets/<dataset>/processed/images and writes prediction
JSON files to outputs/predictions/m2doc/<dataset>/<image_id>.json.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

# Ensure local src directory is on sys.path
SRC_DIR = Path(__file__).resolve().parent
METHOD_DIR = SRC_DIR.parent
ROOT_DIR = METHOD_DIR.parents[1]

# Fix Windows console UTF-8 encoding
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ocr_provider import OCRProvider

M2DOC_CLASSES = (
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
)

# 1-to-1 mapping for DocLayNet
M2DOC_TO_DOCLAYNET = {
    "Caption": 1,
    "Footnote": 2,
    "Formula": 3,
    "List-item": 4,
    "Page-footer": 5,
    "Page-header": 6,
    "Picture": 7,
    "Section-header": 8,
    "Table": 9,
    "Text": 10,
    "Title": 11,
}

# Mapping to PubLayNet 5 categories
M2DOC_TO_PUBLAYNET = {
    "Text": 1,
    "Title": 2,
    "List-item": 3,
    "Table": 4,
    "Picture": 5,
}

GDRIVE_WEIGHTS_ID = "14cf3cmVMWTG8g6w0qMd_lkKYfyR0EDX-"


def ensure_m2doc_source() -> Path:
    """Đảm bảo mã nguồn MMDetection / M2Doc tồn tại cục bộ."""
    from mmcv_shim import setup_m2doc_environment

    setup_m2doc_environment()

    mmdet_dir = METHOD_DIR / "mmdetection"
    if not (mmdet_dir / "mmdet").is_dir():
        print(f"[m2doc] Chưa tìm thấy mã nguồn tại: {mmdet_dir}")
        print("[m2doc] Đang tự động clone mã nguồn M2Doc từ GitHub (https://github.com/johnning2333/M2Doc.git)...")
        import shutil
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/johnning2333/M2Doc.git", tmpdir])
            src_mmdet = Path(tmpdir) / "mmdetection"
            if src_mmdet.is_dir():
                shutil.copytree(src_mmdet, mmdet_dir, dirs_exist_ok=True)
                print("[m2doc] Thiết lập mã nguồn MMDetection thành công!")
            else:
                shutil.copytree(tmpdir, mmdet_dir, dirs_exist_ok=True)
                print("[m2doc] Thiết lập mã nguồn M2Doc thành công!")

    if str(mmdet_dir) not in sys.path:
        sys.path.insert(0, str(mmdet_dir))

    return mmdet_dir


def download_weights_if_missing(weights_path: Path):
    """Tự động tải pre-trained weights từ Google Drive nếu file chưa tồn tại."""
    if not weights_path.exists() or weights_path.stat().st_size < 1000000:
        print(f"[m2doc] Chưa tìm thấy trọng số tại: {weights_path}")
        print(f"[m2doc] Đang tự động tải pre-trained weights từ Google Drive (ID: {GDRIVE_WEIGHTS_ID})...")
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import gdown
        except ImportError:
            import subprocess

            print("[m2doc] Cài đặt thư viện gdown để tải trọng số...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown

        gdown.download(id=GDRIVE_WEIGHTS_ID, output=str(weights_path), quiet=False)
        if not weights_path.exists() or weights_path.stat().st_size < 1000000:
            raise FileNotFoundError(
                f"Tải trọng số thất bại. Vui lòng tải thủ công file cascade_mask_rcnn_m2doc_r50.pth "
                f"vào {weights_path.parent} theo hướng dẫn trong README.md."
            )
        print(f"[m2doc] Tải trọng số thành công ({weights_path.stat().st_size / (1024*1024):.2f} MB)!")


def find_config_file(config_path: Path) -> Path:
    """Tự động tìm kiếm file cấu hình M2Doc nếu đường dẫn mặc định chưa khớp."""
    if config_path.exists():
        return config_path

    m2doc_config_dir = METHOD_DIR / "mmdetection" / "m2doc_config"
    if m2doc_config_dir.exists():
        candidates = [
            "cascade_mask_rcnn_w_m2doc_r50_doclaynet_12epoch.py",
            "cascade_mask_rcnn_w_m2doc_doclaynet.py",
            "dino-4scale_w_m2doc_doclaynet.py",
        ]
        for candidate in candidates:
            p = m2doc_config_dir / candidate
            if p.exists():
                return p
        py_files = sorted(m2doc_config_dir.glob("*.py"))
        if py_files:
            return py_files[0]
    return config_path


def load_m2doc_model(config_path: Path, weights_path: Path, device: str = "cpu"):
    """Loads M2Doc model from config and weights checkpoint."""
    ensure_m2doc_source()

    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS

    init_default_scope("mmdet")
    resolved_config = find_config_file(config_path)
    if not resolved_config.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình M2Doc tại: {config_path} hoặc {resolved_config}")

    cfg = Config.fromfile(str(resolved_config))
    model = MODELS.build(cfg.model)

    if not weights_path.exists():
        download_weights_if_missing(weights_path)

    ckpt = torch.load(str(weights_path), map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def run_on_image(
    model,
    ocr_provider: Optional[OCRProvider],
    image_path: Path,
    mapping: Dict[str, int],
    threshold: float = 0.3,
    device: str = "cpu",
    use_ocr: bool = True,
) -> Dict:
    """Runs inference on a single image and produces standard prediction schema."""
    from mmengine.structures import InstanceData
    from mmdet.structures import DetDataSample

    try:
        data = np.fromfile(str(image_path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        img = None
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    orig_h, orig_w = img.shape[:2]
    target_size = (1025, 1025)
    resized_img = cv2.resize(img, target_size)
    scale_w = target_size[0] / orig_w
    scale_h = target_size[1] / orig_h

    img_tensor = torch.from_numpy(resized_img).permute(2, 0, 1).float().to(device)

    image_id = int(image_path.stem) if image_path.stem.isdigit() else image_path.stem

    data_sample = DetDataSample()
    data_sample.set_metainfo(
        {
            "img_id": image_id,
            "img_path": str(image_path),
            "ori_shape": (orig_h, orig_w),
            "img_shape": target_size,
            "scale_factor": (scale_w, scale_h),
        }
    )

    text_instances = InstanceData()
    if use_ocr and ocr_provider is not None:
        ocr_boxes, ocr_texts = ocr_provider.get_ocr_for_image(
            str(image_path), image_id=image_id if isinstance(image_id, int) else None, target_size=target_size
        )
        if ocr_boxes and ocr_texts:
            text_instances["text_bboxes"] = torch.tensor(ocr_boxes, dtype=torch.float32, device=device)
            text_instances["texts"] = ocr_texts
        else:
            text_instances["text_bboxes"] = torch.zeros((0, 4), dtype=torch.float32, device=device)
            text_instances["texts"] = []
    else:
        text_instances["text_bboxes"] = torch.zeros((0, 4), dtype=torch.float32, device=device)
        text_instances["texts"] = []

    data_sample.text_instances = text_instances

    batch_data = {
        "inputs": [img_tensor],
        "data_samples": [data_sample],
    }

    with torch.no_grad():
        results = model.test_step(batch_data)

    pred_instances = results[0].pred_instances
    pred_bboxes = pred_instances.bboxes.cpu().numpy()
    pred_scores = pred_instances.scores.cpu().numpy()
    pred_labels = pred_instances.labels.cpu().numpy()

    detections = []
    for bbox, score, label_idx in zip(pred_bboxes, pred_scores, pred_labels):
        if score < threshold:
            continue

        raw_label = M2DOC_CLASSES[label_idx] if label_idx < len(M2DOC_CLASSES) else "Text"
        cat_id = mapping.get(raw_label)
        if cat_id is None:
            continue

        # Scale coordinates back to original image resolution
        x1 = float(bbox[0]) / scale_w
        y1 = float(bbox[1]) / scale_h
        x2 = float(bbox[2]) / scale_w
        y2 = float(bbox[3]) / scale_h

        # Clip within image bounds
        x1 = max(0.0, min(x1, float(orig_w)))
        y1 = max(0.0, min(y1, float(orig_h)))
        x2 = max(0.0, min(x2, float(orig_w)))
        y2 = max(0.0, min(y2, float(orig_h)))

        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)

        if w < 2 or h < 2:
            continue

        detections.append(
            {
                "category_id": cat_id,
                "category_name": raw_label,
                "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                "score": round(float(score), 4),
            }
        )

    return {
        "image_id": image_id,
        "file_name": image_path.name,
        "width": orig_w,
        "height": orig_h,
        "detections": detections,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="doclaynet", help="Dataset name (doclaynet or publaynet)")
    parser.add_argument("--threshold", type=float, default=0.3, help="Confidence score threshold (default: 0.3)")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of images; 0 processes the full dataset (default).",
    )
    parser.add_argument("--image-ids", type=str, default="", help="Optional comma-separated image IDs")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(SRC_DIR.parent / "weights" / "cascade_mask_rcnn_m2doc_r50.pth"),
        help="Path to M2Doc checkpoint file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(SRC_DIR.parent / "mmdetection" / "m2doc_config" / "cascade_mask_rcnn_w_m2doc_r50_doclaynet_12epoch.py"),
        help="Path to M2Doc config file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Custom output predictions directory",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: 'cpu' or 'cuda' (default: cuda if available else cpu)",
    )
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR text features (visual only)")

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
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Thư mục ảnh không tồn tại: {images_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else ROOT_DIR / "outputs" / "predictions" / "m2doc" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping = M2DOC_TO_DOCLAYNET if args.dataset == "doclaynet" else M2DOC_TO_PUBLAYNET

    # Collect images with natural sorting
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_paths = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in image_extensions],
        key=lambda path: (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem),
    )

    if args.image_ids:
        selected_ids = {s.strip() for s in args.image_ids.split(",") if s.strip()}
        image_paths = [p for p in image_paths if p.stem in selected_ids or p.name in selected_ids]

    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    if args.skip_existing:
        image_paths = [
            path for path in image_paths if not (output_dir / f"{path.stem}.json").exists()
        ]

    print(
        f"[m2doc] {len(image_paths)} image(s) to predict from {images_dir} "
        f"(threshold: {args.threshold}, device: {args.device}, use_ocr: {not args.no_ocr})"
        f"{' (existing predictions skipped)' if args.skip_existing else ' (force overwrite)'}"
    )

    if not image_paths:
        print(f"[m2doc] No work to do. Predictions are in {output_dir}")
        return

    # Load Model & OCR
    print(f"[m2doc] Loading M2Doc model from: {args.weights}")
    model = load_m2doc_model(Path(args.config), Path(args.weights), device=args.device)
    ocr_provider = OCRProvider() if not args.no_ocr else None
    print("[m2doc] M2Doc model ready for inference!")

    success_count = 0
    total_seconds = 0.0

    for idx, img_path in enumerate(image_paths, 1):
        started = time.time()
        try:
            res = run_on_image(
                model=model,
                ocr_provider=ocr_provider,
                image_path=img_path,
                mapping=mapping,
                threshold=args.threshold,
                device=args.device,
                use_ocr=not args.no_ocr,
            )
            elapsed = time.time() - started
            total_seconds += elapsed

            out_file = output_dir / f"{img_path.stem}.json"
            out_file.write_text(
                json.dumps(res, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            det_count = len(res["detections"])
            print(
                f"  [{idx}/{len(image_paths)}] {img_path.name}: "
                f"{det_count} boxes ({elapsed:.2f}s)"
            )
            success_count += 1
        except Exception as e:
            print(f"  [{idx}/{len(image_paths)}] ERROR processing {img_path.name}: {e}")

    avg_speed = (total_seconds / success_count) if success_count > 0 else 0.0
    print(
        f"[m2doc] Done. Average {avg_speed:.2f}s/image. "
        f"Predictions: {output_dir}"
    )


if __name__ == "__main__":
    main()
