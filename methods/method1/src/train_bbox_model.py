import argparse
import shutil
import sys
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[3]
YAML_PATH = ROOT / "methods" / "method1" / "data" / "doclaynet_bbox_only.yaml"
WEIGHTS_DIR = ROOT / "methods" / "method1" / "weights"
RUNS_DIR = ROOT / "methods" / "method1" / "train_runs"


def main():
    parser = argparse.ArgumentParser(description="Pretrain / Fine-tune YOLOv8 on DocLayNet Bounding Boxes (1-Class / Class-Agnostic)")
    parser.add_argument("--base-weights", default="methods/method1/weights/yolov8n.pt", help="Trọng số khởi tạo ban đầu")
    parser.add_argument("--epochs", type=int, default=5, help="Số lượng epoch huấn luyện")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh đầu vào (640 hoặc 1024)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--name", default="yolov8n_bbox_only", help="Tên phiên huấn luyện")
    parser.add_argument("--device", default="0", help="Device (0 hoặc cpu)")
    args = parser.parse_args()

    base_weights_path = ROOT / args.base_weights if not Path(args.base_weights).is_absolute() else Path(args.base_weights)
    if not base_weights_path.is_file():
        # Fallback to yolov8n.pt
        base_weights_path = "yolov8n.pt"

    print("=" * 85)
    print(f"KHỞI ĐỘNG PRETRAIN MÔ HÌNH PHÁT HIỆN BOUNDING BOX (1-CLASS / CLASS-AGNOSTIC)")
    print(f"Base Weights : {base_weights_path}")
    print(f"Dataset YAML : {YAML_PATH}")
    print(f"Epochs       : {args.epochs} | ImgSz: {args.imgsz} | Batch: {args.batch} | Device: {args.device}")
    print(f"Output Runs  : {RUNS_DIR / args.name}")
    print("=" * 85)

    model = YOLO(str(base_weights_path))

    results = model.train(
        data=str(YAML_PATH),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        single_cls=True,        # Bắt buộc chế độ 1-class / Class-agnostic
        project=str(RUNS_DIR),
        name=args.name,
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

    best_pt = RUNS_DIR / args.name / "weights" / "best.pt"
    dest_best = WEIGHTS_DIR / f"{args.name}_best.pt"

    if best_pt.is_file():
        shutil.copy2(best_pt, dest_best)
        print(f"\n✓ Huấn luyện hoàn tất! Trọng số tối ưu đã được lưu tại:")
        print(f"  -> {dest_best} ({dest_best.stat().st_size / 1e6:.2f} MB)")
    else:
        print(f"\n[warn] Không tìm thấy file {best_pt}")

    return results


if __name__ == "__main__":
    main()
