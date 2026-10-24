import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

TRAIN_ANNOTATION = ROOT / "datasets" / "doclaynet" / "processed" / "annotations" / "train.json"
TRAIN_IMG_DIR = ROOT / "datasets" / "doclaynet" / "processed" / "images"
TRAIN_LABEL_DIR = ROOT / "datasets" / "doclaynet" / "processed" / "labels"

TEST_ANNOTATION = ROOT / "datasets" / "doclaynet_test" / "processed" / "annotations" / "test.json"
TEST_IMG_DIR = ROOT / "datasets" / "doclaynet_test" / "processed" / "images"
TEST_LABEL_DIR = ROOT / "datasets" / "doclaynet_test" / "processed" / "labels"

OUTPUT_DATA_DIR = ROOT / "methods" / "method1" / "data"
OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)


def convert_coco_to_yolo_single_class(annotation_path: Path, img_dir: Path, label_dir: Path) -> list[str]:
    """Chuyen doi toan bo bounding box trong COCO annotation thanh file txt YOLO (1 class: id=0)."""
    label_dir.mkdir(parents=True, exist_ok=True)
    with open(annotation_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Nhom annotations theo image_id
    anns_by_img = {}
    for ann in coco.get("annotations", []):
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    valid_img_paths = []
    num_boxes = 0

    for img_info in coco.get("images", []):
        img_id = img_info["id"]
        file_name = img_info["file_name"]
        img_p = img_dir / file_name

        if not img_p.is_file():
            continue

        valid_img_paths.append(str(img_p.resolve()))

        w_img = float(img_info.get("width", 1025))
        h_img = float(img_info.get("height", 1025))

        # Ten file label trung stem voi anh
        label_file = label_dir / f"{Path(file_name).stem}.txt"
        lines = []

        for ann in anns_by_img.get(img_id, []):
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue

            # Quy doi toa do thanh center_x, center_y, norm_w, norm_h trong [0, 1]
            cx = (x + w / 2.0) / w_img
            cy = (y + h / 2.0) / h_img
            nw = w / w_img
            nh = h / h_img

            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            # Tat ca quy ve class_id = 0 (bbox / layout element)
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            num_boxes += 1

        with open(label_file, "w", encoding="utf-8") as lf:
            lf.write("\n".join(lines))

    print(f"✓ Hoàn tất {annotation_path.name}: {len(valid_img_paths)} ảnh, {num_boxes:,} bounding boxes (1 class).")
    return valid_img_paths


def main():
    print("=" * 80)
    print("CHUẨN BỊ DỮ LIỆU DOCLAYNET 1-CLASS (BOUNDING BOX ONLY) CHO YOLO")
    print("=" * 80)

    # 1. Chuyen doi tap train
    print(f"\n1. Chuyển đổi tập train từ: {TRAIN_ANNOTATION}")
    train_paths = convert_coco_to_yolo_single_class(TRAIN_ANNOTATION, TRAIN_IMG_DIR, TRAIN_LABEL_DIR)

    # 2. Chuyen doi tap test (dung 500 anh lam validation, so con lai lam test)
    print(f"\n2. Chuyển đổi tập test từ: {TEST_ANNOTATION}")
    test_paths = convert_coco_to_yolo_single_class(TEST_ANNOTATION, TEST_IMG_DIR, TEST_LABEL_DIR)

    val_paths = test_paths[:500]
    eval_test_paths = test_paths[500:]

    # Ghi file danh sach duong dan anh
    train_txt = OUTPUT_DATA_DIR / "train_images.txt"
    val_txt = OUTPUT_DATA_DIR / "val_images.txt"
    test_txt = OUTPUT_DATA_DIR / "test_images.txt"

    train_txt.write_text("\n".join(train_paths), encoding="utf-8")
    val_txt.write_text("\n".join(val_paths), encoding="utf-8")
    test_txt.write_text("\n".join(eval_test_paths), encoding="utf-8")

    print(f"\n✓ Đã tạo danh sách train ({len(train_paths):,} ảnh): {train_txt}")
    print(f"✓ Đã tạo danh sách val ({len(val_paths):,} ảnh): {val_txt}")
    print(f"✓ Đã tạo danh sách test ({len(eval_test_paths):,} ảnh): {test_txt}")

    # 3. Tao file cau hinh dataset YAML cho YOLO
    yaml_content = f"""# Cấu hình Dataset DocLayNet 1-Class (Chỉ Bounding Box thuần túy)
path: {ROOT.as_posix()}
train: {train_txt.as_posix()}
val: {val_txt.as_posix()}
test: {test_txt.as_posix()}

nc: 1
names:
  0: bbox
"""
    yaml_file = OUTPUT_DATA_DIR / "doclaynet_bbox_only.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")
    print(f"\n✓ Đã tạo file cấu hình YOLO dataset YAML: {yaml_file}")


if __name__ == "__main__":
    main()
