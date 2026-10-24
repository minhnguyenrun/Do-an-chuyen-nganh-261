import json
import shutil
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[3]
YAML_PATH = ROOT / "methods/method1/data/doclaynet_bbox_only.yaml"
WEIGHTS_DIR = ROOT / "methods/method1/weights"
RUNS_DIR = ROOT / "methods/method1/train_runs"
VIS_DIR = ROOT / "methods/method1/visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)
(VIS_DIR / "before").mkdir(parents=True, exist_ok=True)
(VIS_DIR / "after").mkdir(parents=True, exist_ok=True)
(VIS_DIR / "comparisons").mkdir(parents=True, exist_ok=True)

RUN_NAME = "yolov8n_doclaynet_10k"
BASE_WEIGHTS = ROOT / "methods/method1/weights/yolov8n_doclaynet_bbox_only_best.pt"
EPOCHS = 25
BATCH = 16
IMGSZ = 640
DEVICE = "0"

def main():
    print("=" * 85, flush=True)
    print("TIẾN TRÌNH HUẤN LUYỆN PIPELINE A (YOLOV8N BBOX ONLY) TRÊN 10,604 ẢNH DOCLAYNET", flush=True)
    print(f"Base Weights : {BASE_WEIGHTS}", flush=True)
    print(f"Dataset YAML : {YAML_PATH}", flush=True)
    print(f"Epochs       : {EPOCHS} | Batch: {BATCH} | ImgSz: {IMGSZ} | Device: {DEVICE}", flush=True)
    print(f"Run Name     : {RUN_NAME}", flush=True)
    print("=" * 85, flush=True)

    # 1. Huấn luyện mô hình
    model = YOLO(str(BASE_WEIGHTS))
    t0 = time.time()
    results = model.train(
        data=str(YAML_PATH),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        single_cls=True,
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=True,
        plots=True,
        save=True,
        workers=2,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=1.0,
        verbose=True,
    )
    train_duration = time.time() - t0
    print(f"\n✓ Huấn luyện 10,604 ảnh hoàn tất trong {train_duration / 60:.2f} phút!", flush=True)

    best_pt = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
    dest_best = WEIGHTS_DIR / f"{RUN_NAME}_best.pt"
    if best_pt.is_file():
        shutil.copy2(best_pt, dest_best)
        print(f"✓ Đã lưu trọng số tốt nhất: {dest_best} ({dest_best.stat().st_size / 1e6:.2f} MB)", flush=True)

    # 2. Đánh giá chỉ số SAU KHI TRAIN
    print("\n" + "=" * 85, flush=True)
    print("ĐÁNH GIÁ CHỈ SỐ MÔ HÌNH SAU KHI HUẤN LUYỆN (10,604 ẢNH)", flush=True)
    print("=" * 85, flush=True)
    model_after = YOLO(str(dest_best))
    val_metrics = model_after.val(data=str(YAML_PATH), split="val", imgsz=IMGSZ, device=DEVICE, verbose=False)
    p = val_metrics.results_dict["metrics/precision(B)"]
    r = val_metrics.results_dict["metrics/recall(B)"]
    map50 = val_metrics.results_dict["metrics/mAP50(B)"]
    map50_95 = val_metrics.results_dict["metrics/mAP50-95(B)"]
    f1 = 2 * p * r / (p + r + 1e-8)

    results_after = {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "map50": float(map50),
        "map50_95": float(map50_95),
        "train_duration_min": float(train_duration / 60.0),
        "epochs": EPOCHS,
    }
    with open(VIS_DIR / "metrics_after.json", "w", encoding="utf-8") as f:
        json.dump(results_after, f, indent=2)

    print(f"Metrics Sau Train: P={p:.4f} | R={r:.4f} | F1={f1:.4f} | mAP@0.5={map50:.4f} | mAP@0.5:0.95={map50_95:.4f}", flush=True)

    # 3. Sinh ảnh Visualization & Ghép Side-by-Side Before vs After
    print("\n" + "=" * 85, flush=True)
    print("SINH ẢNH VISUALIZATION & SO SÁNH TRỰC QUAN BEFORE VS AFTER", flush=True)
    print("=" * 85, flush=True)
    with open(VIS_DIR / "sample_images.json", encoding="utf-8") as f:
        sample_imgs = json.load(f)

    for idx, img_p in enumerate(sample_imgs):
        # Dự đoán với model sau train
        res_after = model_after(img_p, imgsz=IMGSZ, conf=0.25, device=DEVICE)[0]
        out_after = VIS_DIR / "after" / f"sample_{idx+1}.jpg"
        res_after.save(filename=str(out_after))

        # Đọc ảnh before và after để ghép so sánh
        out_before = VIS_DIR / "before" / f"sample_{idx+1}.jpg"
        img_b = cv2.imread(str(out_before))
        img_a = cv2.imread(str(out_after))

        if img_b is not None and img_a is not None:
            # Resize cùng chiều cao nếu cần
            h = min(img_b.shape[0], img_a.shape[0])
            w_b = int(img_b.shape[1] * (h / img_b.shape[0]))
            w_a = int(img_a.shape[1] * (h / img_a.shape[0]))
            img_b_resized = cv2.resize(img_b, (w_b, h))
            img_a_resized = cv2.resize(img_a, (w_a, h))

            # Header banner
            banner_b = np.full((50, w_b, 3), (40, 40, 40), dtype=np.uint8)
            cv2.putText(banner_b, "BEFORE: 1,928 Images", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
            banner_a = np.full((50, w_a, 3), (40, 40, 40), dtype=np.uint8)
            cv2.putText(banner_a, "AFTER: 10,604 Images", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 100), 2)

            col_b = np.vstack([banner_b, img_b_resized])
            col_a = np.vstack([banner_a, img_a_resized])
            separator = np.full((col_b.shape[0], 10, 3), (255, 255, 255), dtype=np.uint8)

            combo = np.hstack([col_b, separator, col_a])
            combo_path = VIS_DIR / "comparisons" / f"comparison_sample_{idx+1}.jpg"
            cv2.imwrite(str(combo_path), combo)
            print(f"✓ Đã tạo ảnh so sánh trực quan: {combo_path.name}", flush=True)

    print("\n✓ TOÀN BỘ TIẾN TRÌNH HUẤN LUYỆN, ĐO ĐẠC & VISUALIZATION ĐÃ HOÀN TẤT!", flush=True)

if __name__ == "__main__":
    main()
