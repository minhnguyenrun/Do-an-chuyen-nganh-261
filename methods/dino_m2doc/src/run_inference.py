"""Inference script for Standalone DINO + M2Doc.

Model: DINO_w_M2Doc (ResNet-50 + Multilingual BERT Early & Late Fusion + Deformable DINO)
Paper: "M2Doc: A Multi-Modal Fusion Approach for Document Layout Analysis" (AAAI 2024)
Achieves 89.3% mAP on DocLayNet test set.

Scans images from datasets/<dataset>/processed/images and writes prediction
JSON files to outputs/predictions/dino_m2doc/<dataset>/<image_id>.json.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

try:
    import pyarrow.dataset  # noqa: F401 — must be imported before cv2/torch on Windows
    import pyarrow.parquet  # noqa: F401
except ImportError:
    pass

import cv2
import numpy as np
import torch

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

# Ensure paths
SRC_DIR = Path(__file__).resolve().parent
METHOD_DIR = SRC_DIR.parent
ROOT_DIR = METHOD_DIR.parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mmcv_shim import setup_m2doc_environment
from ocr_provider import OCRProvider
from src.common.profiling import ResourceMonitor, save_runtime_summary

# DINO + M2Doc Classes (11 classes of DocLayNet)
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

# Official Google Drive ID for DINO + M2Doc (ResNet-50, 12 epochs, 89.3 mAP)
GDRIVE_DINO_M2DOC_R50_ID = "11ZwmLLqz-SaehUIsuZg5incXOTgqeIwF"


def resolve_weights_path(user_weights_path: Optional[Union[str, Path]] = None) -> Path:
    """Xác định đường dẫn trọng số DINO + M2Doc, tự động dùng lại từ method3 nếu có."""
    if user_weights_path:
        p = Path(user_weights_path)
        if p.exists() and p.stat().st_size > 100_000_000:
            return p

    # 1. Kiểm tra trong chính thư mục methods/dino_m2doc/weights
    local_weights = METHOD_DIR / "weights" / "dino_m2doc_r50.pth"
    if local_weights.exists() and local_weights.stat().st_size > 100_000_000:
        return local_weights

    # 2. Kiểm tra trong thư mục methods/method3/weights (tái sử dụng tránh nhân đôi file 1.56GB)
    method3_weights = ROOT_DIR / "methods" / "method3" / "weights" / "dino_m2doc_r50.pth"
    if method3_weights.exists() and method3_weights.stat().st_size > 100_000_000:
        print(f"[dino_m2doc] Tái sử dụng trọng số từ Method 3: {method3_weights}")
        return method3_weights

    # 3. Nếu chưa có, tải về local_weights
    print(f"[dino_m2doc] Chưa tìm thấy trọng số pre-trained DINO + M2Doc.")
    print(f"[dino_m2doc] Đang tự động tải về từ Google Drive (ID: {GDRIVE_DINO_M2DOC_R50_ID})...")
    local_weights.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except ImportError:
        import subprocess

        print("[dino_m2doc] Đang cài đặt thư viện gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

    gdown.download(id=GDRIVE_DINO_M2DOC_R50_ID, output=str(local_weights), quiet=False)
    if not local_weights.exists() or local_weights.stat().st_size < 100_000_000:
        raise FileNotFoundError(
            f"Không thể tải trọng số DINO + M2Doc. Vui lòng tải thủ công vào: {local_weights}"
        )
    print(f"[dino_m2doc] Đã tải xong trọng số ({local_weights.stat().st_size / (1024**2):.2f} MB)!")
    return local_weights


def load_dino_m2doc_model(
    config_path: Optional[Union[str, Path]] = None,
    weights_path: Optional[Union[str, Path]] = None,
    device: str = "cpu",
):
    """Khởi tạo và nạp trọng số mô hình DINO_w_M2Doc."""
    setup_m2doc_environment()

    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS

    init_default_scope("mmdet")

    if config_path is None:
        config_path = (
            ROOT_DIR
            / "methods"
            / "m2doc"
            / "mmdetection"
            / "m2doc_config"
            / "dino-4scale_w_m2doc_r50_doclaynet_12epoch.py"
        )
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy config DINO+M2Doc tại: {config_path}")

    resolved_weights = resolve_weights_path(weights_path)

    print(f"[dino_m2doc] Đang nạp cấu hình từ: {config_path.name}")
    print(f"[dino_m2doc] Đang nạp trọng số từ: {resolved_weights}")

    cfg = Config.fromfile(str(config_path))
    if hasattr(cfg.model, "backbone") and "init_cfg" in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    model = MODELS.build(cfg.model)
    ckpt = torch.load(str(resolved_weights), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"[dino_m2doc] Nạp trọng số thành công! "
        f"(missing keys: {len(missing)}, unexpected keys: {len(unexpected)})"
    )

    model.to(device)
    model.eval()
    return model


def run_on_image(
    model,
    ocr_provider: Optional[OCRProvider],
    image_path: Path,
    threshold: float = 0.3,
    device: str = "cpu",
    use_ocr: bool = True,
) -> Dict:
    """Chạy suy luận DINO_w_M2Doc trên 1 ảnh và trả về cấu trúc detection COCO chuẩn."""
    from mmengine.structures import InstanceData
    from mmdet.structures import DetDataSample

    try:
        data = np.fromfile(str(image_path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        img = None
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")

    orig_h, orig_w = img.shape[:2]
    target_size = (1025, 1025)
    resized_img = cv2.resize(img, target_size) if (orig_w, orig_h) != target_size else img
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
            str(image_path),
            image_id=image_id if isinstance(image_id, int) else None,
            target_size=target_size,
            orig_size=(orig_w, orig_h),
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

    t0 = time.perf_counter()
    with torch.no_grad():
        prepped = model.data_preprocessor(batch_data, False)
        results = model.predict(prepped["inputs"], prepped["data_samples"], rescale=False)
    page_time = time.perf_counter() - t0

    pred_instances = results[0].pred_instances
    pred_bboxes = pred_instances.bboxes.detach().cpu().numpy()
    pred_scores = pred_instances.scores.detach().cpu().numpy()
    pred_labels = pred_instances.labels.detach().cpu().numpy()

    detections = []
    for bbox, score, label_idx in zip(pred_bboxes, pred_scores, pred_labels):
        if score < threshold:
            continue

        raw_label = M2DOC_CLASSES[int(label_idx)] if int(label_idx) < len(M2DOC_CLASSES) else "Text"
        cat_id = M2DOC_TO_DOCLAYNET.get(raw_label, 10)

        # Scale coordinates back to original resolution
        x1 = float(bbox[0]) / scale_w
        y1 = float(bbox[1]) / scale_h
        x2 = float(bbox[2]) / scale_w
        y2 = float(bbox[3]) / scale_h

        # Clip bounds
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
        "inference_time_s": round(page_time, 4),
        "detections": detections,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="doclaynet_test", help="Tên tập dữ liệu (doclaynet_test, doclaynet...)")
    parser.add_argument("--split", default="test", help="Split tên (test, train, val)")
    parser.add_argument("--limit", type=int, default=0, help="Giới hạn số ảnh xử lý (0 = tất cả)")
    parser.add_argument("--config", default=None, help="Đường dẫn file cấu hình MMDetection")
    parser.add_argument("--weights", default=None, help="Đường dẫn file checkpoint trọng số .pth")
    parser.add_argument("--device", default=None, help="Thiết bị tính toán (cuda hoặc cpu)")
    parser.add_argument("--threshold", "--score-thr", type=float, default=0.3, dest="threshold", help="Ngưỡng score lọc box")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Bỏ qua các ảnh đã có file dự đoán")
    parser.add_argument("--force", action="store_true", help="Ghi đè dự đoán cũ kể cả khi đã tồn tại")
    parser.add_argument("--no-ocr", action="store_true", help="Tắt module OCR, dùng dummy boxes")
    parser.add_argument("--output-dir", default=None, help="Thư mục xuất file dự đoán")
    args = parser.parse_args()

    if args.force:
        args.skip_existing = False

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    images_dir = ROOT_DIR / "datasets" / args.dataset / "processed" / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh tại: {images_dir}")

    image_files = sorted(
        images_dir.glob("*.png"),
        key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem),
    )
    if not image_files:
        image_files = sorted(
            images_dir.glob("*.jpg"),
            key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem),
        )

    if not image_files:
        raise FileNotFoundError(f"Không có ảnh nào trong: {images_dir}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT_DIR / "outputs" / "predictions" / "dino_m2doc" / args.dataset
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lọc các ảnh cần xử lý
    to_process = []
    for img_path in image_files:
        pred_file = output_dir / f"{img_path.stem}.json"
        if args.skip_existing and pred_file.exists():
            continue
        to_process.append(img_path)

    if args.limit > 0:
        to_process = to_process[: args.limit]

    print(
        f"[dino_m2doc] {len(to_process)} image(s) to predict from {images_dir} "
        f"(threshold: {args.threshold}, device: {device}, use_ocr: {not args.no_ocr}) "
        f"({'skip existing' if args.skip_existing else 'overwrite'})"
    )

    if not to_process:
        print("[dino_m2doc] Không còn ảnh nào cần xử lý. Hoàn tất!")
        return

    # Khởi tạo mô hình và OCR
    monitor = ResourceMonitor()
    with monitor.time_model_load():
        model = load_dino_m2doc_model(config_path=args.config, weights_path=args.weights, device=device)
    print(f"[dino_m2doc] DINO + M2Doc sẵn sàng cho suy luận sau {monitor.model_load_seconds:.2f}s!")

    ocr_provider = None if args.no_ocr else OCRProvider(dataset=args.dataset)

    failed = []
    total_pages = len(to_process)

    for idx, img_path in enumerate(to_process, 1):
        try:
            with monitor.time_page() as timer:
                pred_dict = run_on_image(
                    model=model,
                    ocr_provider=ocr_provider,
                    image_path=img_path,
                    threshold=args.threshold,
                    device=device,
                    use_ocr=not args.no_ocr,
                )
        except Exception as e:
            if monitor.page_seconds:
                monitor.page_seconds.pop()
            failed.append(img_path.name)
            print(f"  [{idx}/{total_pages}] LỖI xử lý {img_path.name}: {e}", flush=True)
            continue

        elapsed = timer.seconds
        pred_dict["inference_time_s"] = round(elapsed, 4)

        out_path = output_dir / f"{img_path.stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(pred_dict, f, indent=2, ensure_ascii=False)

        num_boxes = len(pred_dict["detections"])
        if idx <= 20 or idx % 50 == 0 or idx == total_pages:
            print(f"  [{idx}/{total_pages}] {img_path.name}: {num_boxes} boxes ({elapsed:.2f}s)", flush=True)

    if monitor.page_seconds:
        summary = monitor.summary(dataset=args.dataset, split=args.split, threshold=args.threshold)
        runtime_file = save_runtime_summary("dino_m2doc", args.dataset, summary)
        vram = summary.get("peak_vram_mb")
        print(
            f"[dino_m2doc] Done. Mean {summary.get('mean_s_per_page', 0):.3f}s/page "
            f"(median {summary.get('median_s_per_page', 0):.3f}s, p95 {summary.get('p95_s_per_page', 0):.3f}s) | "
            f"peak RAM {summary.get('peak_ram_mb', 0):.0f} MB | "
            f"peak VRAM {f'{vram:.0f} MB' if vram is not None else 'n/a'} | {summary.get('device', device)}",
            flush=True,
        )
        print(f"[dino_m2doc] Runtime summary: {runtime_file}", flush=True)
    print(f"[dino_m2doc] Predictions: {output_dir}", flush=True)
    if failed:
        raise SystemExit(f"[dino_m2doc] {len(failed)} ảnh bị lỗi: {', '.join(failed[:20])}")


if __name__ == "__main__":
    main()
