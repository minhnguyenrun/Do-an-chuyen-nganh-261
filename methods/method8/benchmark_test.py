"""Method 8 - Benchmark Độc Lập Trên 4,999 Trang Test Set (COCO mAP@[.50:.95])."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from methods.method8.common import (
    BASE_MODEL, CACHE_DIR, REPORT_DIR, TEST_ANNO, TEST_IMG_640,
    TEST_IMG_SRC, WEIGHTS_DIR, coco_eval, load_coco,
)
from methods.method8.dataset import Method8FastDataset, collate
from methods.method8.models.rtdetr_method8 import SpatialSidecarRTDetrMethod8
from methods.method8.evaluator import evaluate_method8_coco_map


def main():
    print("=" * 80)
    print("🧪 BENCHMARK METHOD 8 TRÊN TẬP TEST CHÍNH THỨC (4,999 TRANG)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị: {device}")

    # Đọc nhãn test
    test_coco = load_coco(TEST_ANNO)
    print(f"Số trang test: {len(test_coco['images']):,}")

    test_ds = Method8FastDataset(
        test_coco, split="test", img_640_dir=TEST_IMG_640, img_raw_dir=TEST_IMG_SRC,
        allowed_books=None, drop_empty=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate, pin_memory=True
    )

    ckpt_path = WEIGHTS_DIR / "method8_best.pth"
    if not ckpt_path.exists():
        print(f"Chưa tìm thấy {ckpt_path.name}. Hãy huấn luyện trước!")
        return

    model = SpatialSidecarRTDetrMethod8(
        pretrained_model_path=BASE_MODEL,
        num_classes=17,
        heatmap_channels=22,
        freeze_backbone=True,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    t0 = time.time()
    results = evaluate_method8_coco_map(model, test_loader, test_coco, device)
    total_time = time.time() - t0
    fps = len(test_ds) / total_time

    print(f"\n🏆 KẾT QUẢ METHOD 8:")
    print(f" - mAP@[.50:.95]: {results['map']*100:.2f}%")
    print(f" - AP50: {results['map50']*100:.2f}%")
    print(f" - AP75: {results['map75']*100:.2f}%")
    print(f" - Tốc độ: {fps:.1f} FPS ({total_time:.1f}s / {len(test_ds)} trang)")

    out_file = REPORT_DIR / "test_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Đã lưu kết quả tại: {out_file}")


if __name__ == "__main__":
    main()
