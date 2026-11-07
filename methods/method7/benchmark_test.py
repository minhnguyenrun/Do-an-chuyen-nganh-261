"""Benchmark chính thức Method 7 trên toàn bộ 4,999 trang DocLayNet Test Set.

Nguyên tắc kiểm thử trung thực:
1. Heatmap 22 kênh sinh từ dự đoán của Base model, không có nhãn GT của tập test.
2. Không lọc confidence threshold (chuẩn top-100 COCO).
3. Đánh giá pycocotools trực tiếp trên test.json gốc.
4. Đo đạc tốc độ FPS thực tế và Peak VRAM.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from methods.method7.common import (
    BASE_MODEL, REPORT_DIR, TEST_ANNO, TEST_IMG, WEIGHTS_DIR,
    load_coco,
)
from methods.method7.dataset import Method7Dataset, collate
from methods.method7.models.rtdetr_method7 import SpatialSidecarRTDetrMethod7
from methods.method7.evaluator import evaluate_method7_coco_map


def main():
    parser = argparse.ArgumentParser(description="Benchmark Method 7 trên 4,999 trang test")
    parser.add_argument("--checkpoint", type=str, default=str(WEIGHTS_DIR / "method7_best.pth"))
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    print("=" * 80)
    print("🚀 BẮT ĐẦU BENCHMARK TOÀN DIỆN METHOD 7 TRÊN 4,999 TRANG TEST")
    print(f"  Checkpoint: {args.checkpoint}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_coco = load_coco(TEST_ANNO)
    test_ds = Method7Dataset(test_coco, TEST_IMG, split="test", drop_empty=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate,
        pin_memory=True,
    )

    model = SpatialSidecarRTDetrMethod7(
        pretrained_model_path=BASE_MODEL,
        num_classes=17,
        heatmap_channels=22,
        freeze_backbone=True,
    ).to(device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        fallback = WEIGHTS_DIR / "method7_best.pth"
        if fallback.exists():
            ckpt_path = fallback
        else:
            print(f"[Warning] Không tìm thấy checkpoint tại {ckpt_path}, dùng weights gốc!")
            ckpt_path = None

    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"✓ Đã nạp checkpoint: {ckpt_path.name}")
        if "fusion_gates" in ckpt:
            print(f"  Fusion Gates: {ckpt['fusion_gates']}")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    res = evaluate_method7_coco_map(model, test_loader, test_coco, device)

    torch.cuda.synchronize()
    total_time = time.perf_counter() - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
    fps = len(test_ds) / max(1e-4, total_time)

    res["runtime"] = {
        "pages": len(test_ds),
        "total_time_sec": total_time,
        "fps": fps,
        "ms_per_page": (total_time / len(test_ds)) * 1000.0,
        "peak_vram_mb": peak_vram,
    }

    print(f"\n✓ KẾT QUẢ METHOD 7 TRÊN {len(test_ds):,} TRANG TEST:")
    print(f"  - COCO mAP@[.50:.95]: {res['map']*100:.2f}%")
    print(f"  - COCO AP50:          {res['map50']*100:.2f}%")
    print(f"  - COCO AP75:          {res['map75']*100:.2f}%")
    print(f"  - Tốc độ suy luận:    {fps:.1f} FPS ({total_time:.1f}s)")
    print(f"  - Peak VRAM:          {peak_vram:.1f} MB")

    # Đối chiếu với Base RT-DETRv2 (chấm cùng pycocotools)
    base_eval_path = REPORT_DIR / "base_test_eval.json"
    base_res = {}
    if base_eval_path.exists():
        with open(base_eval_path, "r", encoding="utf-8") as f:
            base_res = json.load(f)

    comp = {
        "method7": res,
        "base_model": base_res,
    }

    out_json = REPORT_DIR / "test_benchmark_comparison.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(comp, f, indent=2, ensure_ascii=False)
    print(f"✓ Đã lưu kết quả so sánh tại: {out_json}")


if __name__ == "__main__":
    main()
