"""Chuẩn bị dữ liệu Pilot cho Method 9 (YOLOv26x).

Quy trình:
1. Độc lập chọn theo CUỐN SÁCH: 
   - ~1,500 ảnh cho tập Train (từ train.json).
   - ~500 ảnh cho tập Val (từ validation.json).
2. Tạo hardlink ảnh 640x640 từ datasets/doclaynet_640 vào methods/method9/data/images/ (không tốn thêm dung lượng đĩa).
3. Chuyển đổi nhãn COCO 11 lớp thành nhãn YOLO chuẩn (0..10) lưu vào methods/method9/data/labels/.
4. Tạo file cấu hình doclaynet_pilot.yaml cho Ultralytics YOLOv26x.
"""

from collections import defaultdict
import json
import os
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[3]
M9_DIR = ROOT_DIR / "methods" / "method9"
DATA_DIR = M9_DIR / "data"

IMG_SRC_TRAIN = ROOT_DIR / "datasets" / "doclaynet_640" / "train"
IMG_SRC_VAL = ROOT_DIR / "datasets" / "doclaynet_640" / "val"

ANNO_TRAIN = ROOT_DIR / "datasets" / "doclaynet" / "processed" / "annotations" / "train.json"
ANNO_VAL = ROOT_DIR / "datasets" / "doclaynet_val" / "processed" / "annotations" / "validation.json"

CATEGORIES = [
    {"id": 1, "name": "Caption"},
    {"id": 2, "name": "Footnote"},
    {"id": 3, "name": "Formula"},
    {"id": 4, "name": "List-item"},
    {"id": 5, "name": "Page-footer"},
    {"id": 6, "name": "Page-header"},
    {"id": 7, "name": "Picture"},
    {"id": 8, "name": "Section-header"},
    {"id": 9, "name": "Table"},
    {"id": 10, "name": "Text"},
    {"id": 11, "name": "Title"},
]


def prepare_split(
    anno_path: Path,
    img_src_dir: Path,
    img_dst_dir: Path,
    lbl_dst_dir: Path,
    target_count: int,
    seed: int = 42,
) -> int:
    import numpy as np

    img_dst_dir.mkdir(parents=True, exist_ok=True)
    lbl_dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {anno_path.name}...")
    with open(anno_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Gom nhóm annotations theo image_id
    anns_by_img = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_img[ann["image_id"]].append(ann)

    # Gom nhóm images theo cuốn sách (original_filename)
    by_book = defaultdict(list)
    for im in coco.get("images", []):
        book = im.get("original_filename") or "_".join(im["file_name"].split("_")[:-1])
        by_book[book].append(im)

    rng = np.random.RandomState(seed)
    book_names = sorted(list(by_book.keys()))
    rng.shuffle(book_names)

    selected_images = []
    selected_books = []
    for b in book_names:
        pages = by_book[b]
        selected_images.extend(pages)
        selected_books.append(b)
        if len(selected_images) >= target_count:
            selected_images = selected_images[:target_count]
            break

    print(f"Đã chọn {len(selected_books)} cuốn sách ({len(selected_images):,} trang) cho {img_dst_dir.name}")

    count = 0
    for im in selected_images:
        img_id = im["id"]
        w_orig = float(im.get("width", 1025))
        h_orig = float(im.get("height", 1025))

        src_jpg = img_src_dir / f"{img_id}.jpg"
        dst_jpg = img_dst_dir / f"{img_id}.jpg"

        if not src_jpg.exists():
            continue

        if not dst_jpg.exists():
            try:
                os.link(src_jpg, dst_jpg)
            except OSError:
                import shutil
                shutil.copy2(src_jpg, dst_jpg)

        # Chuyển đổi nhãn sang YOLO format (11 lớp: 0..10)
        lbl_file = lbl_dst_dir / f"{img_id}.txt"
        lines = []
        for ann in anns_by_img.get(img_id, []):
            cid = ann["category_id"]
            if not (1 <= cid <= 11):
                continue
            yolo_cls = cid - 1

            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue

            cx = max(0.0, min(1.0, (x + w / 2.0) / w_orig))
            cy = max(0.0, min(1.0, (y + h / 2.0) / h_orig))
            nw = max(0.0, min(1.0, w / w_orig))
            nh = max(0.0, min(1.0, h / h_orig))

            lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        with open(lbl_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

        count += 1

    return count


def main():
    print("=== CHUẨN BỊ DỮ LIỆU PILOT METHOD 9 (YOLOv26x) ===")
    train_count = prepare_split(
        ANNO_TRAIN,
        IMG_SRC_TRAIN,
        DATA_DIR / "images" / "train",
        DATA_DIR / "labels" / "train",
        target_count=1500,
        seed=2026,
    )

    val_count = prepare_split(
        ANNO_VAL,
        IMG_SRC_VAL,
        DATA_DIR / "images" / "val",
        DATA_DIR / "labels" / "val",
        target_count=500,
        seed=2026,
    )

    print(f"\n✅ Đã tạo xong tập dữ liệu Pilot:")
    print(f" - Train: {train_count} ảnh")
    print(f" - Val:   {val_count} ảnh")

    # Tạo YAML cấu hình dataset
    yaml_content = f"""# DocLayNet 11-Class Pilot Dataset for Method 9 (YOLOv26x)
path: {DATA_DIR.as_posix()}
train: images/train
val: images/val

nc: 11
names:
  0: Caption
  1: Footnote
  2: Formula
  3: List-item
  4: Page-footer
  5: Page-header
  6: Picture
  7: Section-header
  8: Table
  9: Text
  10: Title
"""
    yaml_path = DATA_DIR / "doclaynet_pilot.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"✅ Đã tạo cấu hình YAML tại: {yaml_path}")


if __name__ == "__main__":
    main()
