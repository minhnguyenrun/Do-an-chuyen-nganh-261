"""Select 50 diverse training books and 10 diverse validation books.
Đảm bảo cân bằng số trang (ngắn, vừa, dài) và tối đa hóa độ phủ các lớp (Class Diversity).
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]


def analyze_and_select_books():
    train_path = ROOT_DIR / "datasets" / "doclaynet" / "processed" / "annotations" / "train.json"
    test_path = ROOT_DIR / "datasets" / "doclaynet_test" / "processed" / "annotations" / "test.json"

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 1. Phân tích tập Train
    train_books = defaultdict(list)
    img_to_cats_train = defaultdict(Counter)
    for ann in train_data["annotations"]:
        img_to_cats_train[ann["image_id"]][ann["category_id"]] += 1

    for img in train_data["images"]:
        orig = img.get("original_filename", "unknown")
        train_books[orig].append(img)

    train_book_stats = []
    for bname, bimgs in train_books.items():
        cat_counts = Counter()
        for img in bimgs:
            cat_counts.update(img_to_cats_train[img["id"]])
        train_book_stats.append({
            "name": bname,
            "num_pages": len(bimgs),
            "cats": dict(cat_counts),
            "unique_cats": len(cat_counts),
            "total_boxes": sum(cat_counts.values()),
        })

    # Phân nhóm theo độ dài sách (Buckets):
    # Bucket 1: Ngắn (3 - 10 trang) -> 12 sách
    # Bucket 2: Vừa (11 - 25 trang) -> 15 sách
    # Bucket 3: Dài (26 - 50 trang) -> 15 sách
    # Bucket 4: Rất dài (51 - 90 trang) -> 8 sách
    # Tổng: 50 sách
    b1 = [b for b in train_book_stats if 3 <= b["num_pages"] <= 10]
    b2 = [b for b in train_book_stats if 11 <= b["num_pages"] <= 25]
    b3 = [b for b in train_book_stats if 26 <= b["num_pages"] <= 50]
    b4 = [b for b in train_book_stats if 51 <= b["num_pages"] <= 90]

    # Sắp xếp mỗi bucket theo độ đa dạng nhãn (unique_cats) và tổng số box
    b1_sorted = sorted(b1, key=lambda x: (x["unique_cats"], x["total_boxes"]), reverse=True)[:12]
    b2_sorted = sorted(b2, key=lambda x: (x["unique_cats"], x["total_boxes"]), reverse=True)[:15]
    b3_sorted = sorted(b3, key=lambda x: (x["unique_cats"], x["total_boxes"]), reverse=True)[:15]
    b4_sorted = sorted(b4, key=lambda x: (x["unique_cats"], x["total_boxes"]), reverse=True)[:8]

    selected_train = b1_sorted + b2_sorted + b3_sorted + b4_sorted
    assert len(selected_train) == 50, f"Expected 50 books, got {len(selected_train)}"

    # 2. Phân tích tập Test để chọn 10 sách Validation
    test_books = defaultdict(list)
    img_to_cats_test = defaultdict(Counter)
    for ann in test_data["annotations"]:
        img_to_cats_test[ann["image_id"]][ann["category_id"]] += 1

    for img in test_data["images"]:
        orig = img.get("original_filename", "unknown")
        test_books[orig].append(img)

    test_book_stats = []
    for bname, bimgs in test_books.items():
        cat_counts = Counter()
        for img in bimgs:
            cat_counts.update(img_to_cats_test[img["id"]])
        test_book_stats.append({
            "name": bname,
            "num_pages": len(bimgs),
            "cats": dict(cat_counts),
            "unique_cats": len(cat_counts),
            "total_boxes": sum(cat_counts.values()),
        })

    # Chọn 10 sách val đa dạng độ dài: 3 ngắn (5-10p), 4 vừa (11-25p), 3 dài (26-50p)
    tb1 = sorted([b for b in test_book_stats if 5 <= b["num_pages"] <= 10], key=lambda x: x["unique_cats"], reverse=True)[:3]
    tb2 = sorted([b for b in test_book_stats if 11 <= b["num_pages"] <= 25], key=lambda x: x["unique_cats"], reverse=True)[:4]
    tb3 = sorted([b for b in test_book_stats if 26 <= b["num_pages"] <= 50], key=lambda x: x["unique_cats"], reverse=True)[:3]
    selected_val = tb1 + tb2 + tb3
    assert len(selected_val) == 10, f"Expected 10 val books, got {len(selected_val)}"

    # Thống kê tổng quan
    total_train_pages = sum(b["num_pages"] for b in selected_train)
    total_val_pages = sum(b["num_pages"] for b in selected_val)

    overall_train_cats = Counter()
    for b in selected_train:
        overall_train_cats.update(b["cats"])

    overall_val_cats = Counter()
    for b in selected_val:
        overall_val_cats.update(b["cats"])

    output_info = {
        "train_books": [b["name"] for b in selected_train],
        "val_books": [b["name"] for b in selected_val],
        "total_train_pages": total_train_pages,
        "total_val_pages": total_val_pages,
        "train_cat_distribution": dict(overall_train_cats),
        "val_cat_distribution": dict(overall_val_cats),
        "train_book_details": selected_train,
        "val_book_details": selected_val,
    }

    out_file = ROOT_DIR / "methods" / "method5" / "data" / "selected_books.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_info, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("📚 HOÀN TẤT TUYỂN CHỌN 50 SÁCH TRAIN & 10 SÁCH VAL ĐA DẠNG")
    print("=" * 80)
    print(f"✓ Tuyển chọn: {len(selected_train)} sách Train ({total_train_pages:,} trang).")
    print(f"✓ Tuyển chọn: {len(selected_val)} sách Val ({total_val_pages:,} trang).")
    print(f"✓ Phân bố trang sách Train:")
    print(f"   - 3-10 trang:   12 cuốn (Tổng: {sum(b['num_pages'] for b in b1_sorted)} trang)")
    print(f"   - 11-25 trang:  15 cuốn (Tổng: {sum(b['num_pages'] for b in b2_sorted)} trang)")
    print(f"   - 26-50 trang:  15 cuốn (Tổng: {sum(b['num_pages'] for b in b3_sorted)} trang)")
    print(f"   - 51-90 trang:   8 cuốn (Tổng: {sum(b['num_pages'] for b in b4_sorted)} trang)")
    print(f"\n✓ Phân bố Class trong 50 sách Train:")
    for cat_id, cnt in sorted(overall_train_cats.items(), key=lambda x: x[1], reverse=True):
        print(f"   Cat {cat_id:>2}: {cnt:>6,} boxes")
    print(f"\n✓ Đã lưu danh sách sách tại: {out_file}")


if __name__ == "__main__":
    analyze_and_select_books()
