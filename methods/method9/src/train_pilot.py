"""Train Pilot YOLOv26x trên 1,500 trang DocLayNet.

Mục tiêu:
- Nạp trọng số pretrained yolo26x.pt (58.99M params).
- Fine-tune trên tập Pilot 11 lớp văn bản của DocLayNet.
- Đánh giá phong độ ban đầu: mAP50, mAP50-95, Loss, Tốc độ FPS, VRAM.
- Xuất báo cáo tóm tắt vào methods/method9/report/pilot_train_results.json.
"""

import json
import os
from pathlib import Path
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[3]
M9_DIR = ROOT_DIR / "methods" / "method9"
DATA_YAML = M9_DIR / "data" / "doclaynet_pilot.yaml"
WEIGHTS_PATH = ROOT_DIR / "weights" / "yolo26_doclaynet_pretrained.pt"
REPORT_DIR = M9_DIR / "report"
WEIGHTS_DIR = M9_DIR / "weights"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def train_pilot(epochs: int = 5, batch: int = 8, imgsz: int = 640):
    print("=" * 80)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN PILOT METHOD 9: YOLOv26 PRE-TRAINED TRÊN DOCLAYNET (11 LỚP)")
    print(f" - Model Pretrained : {WEIGHTS_PATH}")
    print(f" - Dataset YAML     : {DATA_YAML}")
    print(f" - Epochs           : {epochs}")
    print(f" - Batch Size       : {batch}")
    print(f" - Image Size       : {imgsz}")
    print("=" * 80)

    device = 0 if torch.cuda.is_available() else "cpu"

    # 1. Khởi tạo mô hình YOLOv26 đã pre-trained trên DocLayNet
    model = YOLO(str(WEIGHTS_PATH))
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"✅ Đã nạp thành công YOLOv26 ({total_params:,} tham số)")
    print(f"📋 Các lớp đã căn chỉnh sẵn: {model.names}")

    # 2. Huấn luyện Pilot
    t0 = time.time()
    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=2,
        project=str(M9_DIR / "runs"),
        name="pilot_yolo26_doclaynet",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=5e-5,
        lrf=0.01,
        warmup_epochs=0.5,
        plots=True,
        save=True,
        save_period=1,
        val=True,
        amp=True,
    )
    total_train_time = time.time() - t0

    # 3. Đánh giá chi tiết trên tập Val sau fine-tuning
    print("\n🔍 ĐANG CHẠY ĐÁNH GIÁ CHI TIẾT TRÊN TẬP VAL...")
    t_val_start = time.time()
    val_results = model.val(data=str(DATA_YAML), split="val", batch=batch, imgsz=imgsz, device=device)
    val_time = time.time() - t_val_start

    # Lấy metrics
    box_metrics = val_results.box
    map50_95 = float(box_metrics.map)
    map50 = float(box_metrics.map50)
    map75 = float(box_metrics.map75)
    maps_per_class = [float(x) for x in box_metrics.maps]

    categories = [
        "Caption", "Footnote", "Formula", "List-item", "Page-footer",
        "Page-header", "Picture", "Section-header", "Table", "Text", "Title"
    ]
    per_class_summary = {}
    for i, cat in enumerate(categories):
        if i < len(maps_per_class):
            per_class_summary[cat] = {
                "mAP50_95": maps_per_class[i],
            }

    # Copy best weight sang methods/method9/weights/
    best_pt = M9_DIR / "runs" / "pilot_yolo26_doclaynet" / "weights" / "best.pt"
    if best_pt.exists():
        import shutil
        dst_best = WEIGHTS_DIR / "yolo26_doclaynet_best.pt"
        shutil.copy2(best_pt, dst_best)
        print(f"✅ Đã lưu checkpoint tốt nhất vào: {dst_best}")

    report_data = {
        "model": "YOLOv26-DocLayNet",
        "parameters": total_params,
        "epochs": epochs,
        "batch_size": batch,
        "image_size": imgsz,
        "train_samples": 1500,
        "val_samples": 500,
        "total_train_time_sec": total_train_time,
        "val_eval_time_sec": val_time,
        "fps_eval": 500 / max(val_time, 1e-4),
        "metrics": {
            "mAP50_95": map50_95,
            "mAP50": map50,
            "mAP75": map75,
            "per_class": per_class_summary,
        }
    }

    report_path = REPORT_DIR / "pilot_train_results_doclaynet.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"🏆 KẾT QUẢ ĐÁNH GIÁ PHONG ĐỘ METHOD 9 (YOLOv26 PRETRAINED DOCLAYNET):")
    print(f" - mAP@[.50:.95] : {map50_95*100:.2f}%")
    print(f" - mAP@50        : {map50*100:.2f}%")
    print(f" - mAP@75        : {map75*100:.2f}%")
    print(f" - Thời gian Train {epochs} Epochs : {total_train_time/60:.2f} phút")
    print(f" - Tốc độ Suy Luận Val       : {report_data['fps_eval']:.1f} FPS")
    print("=" * 80)
    for cat, m in per_class_summary.items():
        print(f"   * {cat:<16}: mAP@[.50:.95] = {m['mAP50_95']*100:.2f}%")
    print("=" * 80)

    return report_data


if __name__ == "__main__":
    train_pilot(epochs=3, batch=8, imgsz=640)
