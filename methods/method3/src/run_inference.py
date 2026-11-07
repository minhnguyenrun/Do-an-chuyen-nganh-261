"""End-to-End Inference Script for Method 3 (Docling Scout -> Book-Level Encoder -> DINO + M2Doc).

Thực hiện quy trình 2 Pipeline cho sách / tài liệu nhiều trang:
1. Pipeline 1 (Docling Scout - Frozen): Quét toàn bộ cửa sổ trang sách để trích xuất đặc trưng
   cấu trúc R_1, R_2, ..., R_N.
2. Cầu nối (Book-Level Encoder): Tổng hợp ngữ cảnh toàn sách C_{book}.
3. Pipeline 2 (DINO + M2Doc + Context Fusion): Chạy DINO_w_M2Doc (dino_m2doc_r50.pth) kết hợp
   ngữ cảnh C_{book} và OCR để xuất kết quả JSON chuẩn COCO tại outputs/predictions/method3/<dataset>/.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

try:
    import pyarrow.dataset  # noqa: F401
    import pyarrow.parquet  # noqa: F401
except ImportError:
    pass

import torch

SRC_DIR = Path(__file__).resolve().parent
METHOD_DIR = SRC_DIR.parent
ROOT_DIR = METHOD_DIR.parents[1]
M2DOC_SRC_DIR = ROOT_DIR / "methods" / "m2doc" / "src"

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

for p in (SRC_DIR, M2DOC_SRC_DIR, ROOT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ocr_provider import OCRProvider
from pipeline1_docling import DoclingPageScout
from pipeline2_dino_m2doc import (
    DINO_M2Doc_BookContextPipeline,
    M2DOC_TO_DOCLAYNET,
    load_base_dino_m2doc,
)
from src.common.profiling import ResourceMonitor, save_runtime_summary


def is_current_prediction(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return "detections" in data and "inference_time_s" in data
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="doclaynet_test", help="Dataset name (doclaynet, doclaynet_test, ...)")
    parser.add_argument("--threshold", type=float, default=0.3, help="Confidence score threshold (default: 0.3)")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of pages (0 = full dataset)")
    parser.add_argument("--window-size", type=int, default=8, help="Book window size for inter-page context")
    parser.add_argument(
        "--fusion-mode",
        choices=["feature", "box", "both"],
        default="feature",
        help="Context fusion mode: 'feature' (Cách A), 'box' (Cách B), or 'both'",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(METHOD_DIR / "weights" / "dino_m2doc_r50.pth"),
        help="Path to DINO + M2Doc checkpoint (dino_m2doc_r50.pth)",
    )
    parser.add_argument(
        "--context-weights",
        type=str,
        default=str(METHOD_DIR / "weights" / "method3_book_context_best.pth"),
        help="Optional path to trained Book-Level Context & Fusion weights",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(
            ROOT_DIR
            / "methods"
            / "m2doc"
            / "mmdetection"
            / "m2doc_config"
            / "dino-4scale_w_m2doc_r50_doclaynet_12epoch.py"
        ),
        help="Path to DINO + M2Doc MMDetection config",
    )
    parser.add_argument("--output-dir", type=str, default="", help="Custom output predictions directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR text features")

    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument("--skip-existing", dest="skip_existing", action="store_true")
    run_mode.add_argument("--force", dest="skip_existing", action="store_false")
    parser.set_defaults(skip_existing=True)

    args = parser.parse_args()

    images_dir = ROOT_DIR / "datasets" / args.dataset / "processed" / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Thư mục ảnh không tồn tại: {images_dir}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT_DIR / "outputs" / "predictions" / "method3" / args.dataset
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_paths = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in image_extensions],
        key=lambda path: (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem),
    )

    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    if args.skip_existing:
        image_paths = [p for p in image_paths if not is_current_prediction(output_dir / f"{p.stem}.json")]

    print(
        f"[method3] {len(image_paths)} page(s) to process from {images_dir} "
        f"(fusion_mode={args.fusion_mode}, window={args.window_size}, device={args.device})",
        flush=True,
    )
    if not image_paths:
        print(f"[method3] All predictions up-to-date in {output_dir}")
        return

    monitor = ResourceMonitor()
    with monitor.time_model_load():
        ocr_provider = None
        if not args.no_ocr:
            ocr_provider = OCRProvider(dataset=args.dataset)
            ocr_provider._build_parquet_index()

        # 1. Khởi tạo Pipeline 1: Docling Scout (ưu tiên đọc từ predictions/docling nếu đã có sẵn)
        existing_docling_dir = ROOT_DIR / "outputs" / "predictions" / "docling" / args.dataset
        scout = DoclingPageScout(
            cache_dir=ROOT_DIR / "outputs" / "cache" / "method3_docling" / args.dataset,
            existing_predictions_dir=existing_docling_dir if existing_docling_dir.is_dir() else None,
            lazy_init=True,
        )

        # 2. Khởi tạo Pipeline 2: DINO + M2Doc + Book Context Fusion
        base_dino = load_base_dino_m2doc(args.config, args.weights, device=args.device)
        pipeline2 = DINO_M2Doc_BookContextPipeline(
            base_detector=base_dino,
            num_classes=11,
            embed_dim=256,
            fusion_mode=args.fusion_mode,
        ).to(args.device)

        ctx_ckpt_path = Path(args.context_weights)
        if ctx_ckpt_path.is_file():
            ctx_ckpt = torch.load(str(ctx_ckpt_path), map_location=args.device, weights_only=False)
            if "book_encoder" in ctx_ckpt:
                pipeline2.book_encoder.load_state_dict(ctx_ckpt["book_encoder"], strict=False)
            if "box_context_refiner" in ctx_ckpt:
                pipeline2.box_context_refiner.load_state_dict(ctx_ckpt["box_context_refiner"], strict=False)
            print(f"[method3] Đã nạp trọng số Book-Level Context từ: {ctx_ckpt_path.name}")

        pipeline2.eval()

    # Chia chuỗi trang thành các cửa sổ sách (Book Windows)
    total_pages = len(image_paths)
    win_size = max(1, args.window_size)
    processed_count = 0

    for start_idx in range(0, total_pages, win_size):
        window_paths = image_paths[start_idx : start_idx + win_size]

        # BƯỚC 1: Pipeline 1 (Docling Scout) đọc toàn bộ các trang trong cửa sổ sách
        book_reps = scout.extract_book(window_paths)

        # BƯỚC 2: Book-Level Encoder tổng hợp ngữ cảnh liên trang C_{book}
        with torch.no_grad():
            c_book, c_pages = pipeline2.encode_book_context(book_reps, device=args.device)

        # BƯỚC 3: Pipeline 2 (DINO + M2Doc + Context Fusion) dự đoán chi tiết từng trang
        for local_i, img_path in enumerate(window_paths):
            processed_count += 1
            with monitor.time_page() as timer:
                res = pipeline2.predict_page_with_context(
                    image_path=img_path,
                    book_context=c_book,
                    page_context_vec=c_pages[local_i],
                    ocr_provider=ocr_provider,
                    threshold=args.threshold,
                    use_ocr=not args.no_ocr,
                    device=args.device,
                )
            res["inference_time_s"] = round(timer.seconds, 4)
            out_file = output_dir / f"{img_path.stem}.json"
            out_file.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

            if processed_count <= 20 or processed_count % 50 == 0 or processed_count == total_pages:
                print(
                    f"  [{processed_count}/{total_pages}] {img_path.name}: "
                    f"{len(res['detections'])} boxes ({timer.seconds:.2f}s)",
                    flush=True,
                )

    if monitor.page_seconds:
        summary = monitor.summary(dataset=args.dataset)
        save_runtime_summary("method3", args.dataset, summary)
        print(f"[method3] Hoàn tất! Kết quả lưu tại: {output_dir}")


if __name__ == "__main__":
    main()
