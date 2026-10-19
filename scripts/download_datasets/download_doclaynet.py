"""Tải một phần bộ dữ liệu DocLayNet từ HuggingFace Hub.

Dataset gốc (docling-project/DocLayNet-v1.2) được chia thành nhiều shard
.parquet cho mỗi split (train/validation/test), mỗi shard ~450-520MB. Script
này tải các shard theo thứ tự train -> validation -> test cho tới khi đạt
khoảng dung lượng chỉ định (mặc định 1GB), rồi (tuỳ chọn) trích ảnh +
annotation ra dạng COCO-style JSON trong datasets/doclaynet/processed/.

Usage:
    python download_doclaynet.py --max-size-gb 1
    python download_doclaynet.py --max-size-gb 1 --no-extract
    python download_doclaynet.py --max-size-gb 0.5 --max-images 250 --delete-raw
"""

import argparse
import io
import json
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from tqdm import tqdm

REPO_ID = "docling-project/DocLayNet-v1.2"
SPLITS = ["train", "validation", "test"]

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

THIS_DIR = Path(__file__).resolve().parent
RAW_DIR = THIS_DIR / "../../datasets/doclaynet/raw"
PROCESSED_DIR = THIS_DIR / "../../datasets/doclaynet/processed"


def list_shards(api: HfApi):
    """Trả về danh sách (split, rfilename, size_bytes) theo thứ tự ưu tiên split."""
    info = api.dataset_info(REPO_ID, files_metadata=True)
    by_split = {split: [] for split in SPLITS}
    for sibling in info.siblings:
        for split in SPLITS:
            if sibling.rfilename.startswith(f"data/{split}-"):
                size = sibling.size or 0
                by_split[split].append((sibling.rfilename, size))
    for split in SPLITS:
        by_split[split].sort()
    shards = []
    for split in SPLITS:
        for rfilename, size in by_split[split]:
            shards.append((split, rfilename, size))
    return shards


def download_shards(max_size_bytes: int) -> list:
    """Tải shard cho tới khi đạt max_size_bytes. Trả về list (split, local_path)."""
    api = HfApi()
    shards = list_shards(api)

    downloaded = []
    total = 0
    for split, rfilename, size in shards:
        if total >= max_size_bytes:
            break
        print(f"[download] {rfilename} ({size / 1e6:.0f} MB)")
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=rfilename,
            local_dir=str(RAW_DIR),
        )
        downloaded.append((split, local_path))
        total += size

    print(f"[download] Tong cong: {total / 1e9:.2f} GB ({len(downloaded)} shard)")
    return downloaded


def delete_raw_files(downloaded: list):
    """Xoa cac file parquet da tai de giai phong dung luong o cung."""
    freed_bytes = 0
    for split, local_path in downloaded:
        p = Path(local_path)
        if p.exists():
            try:
                size = p.stat().st_size
                p.unlink()
                freed_bytes += size
                print(f"[cleanup] Da xoa file raw: {p.name} ({size / 1e6:.1f} MB)")
            except Exception as e:
                print(f"[cleanup] Khong the xoa {p.name}: {e}")
    print(f"[cleanup] Tong dung luong raw da giai phong: {freed_bytes / 1e6:.1f} MB")


def extract_processed(downloaded: list, max_images: int | None = None):
    """Trich anh (.png) + annotation COCO-style tu cac shard da tai."""
    import pandas as pd

    images_dir = PROCESSED_DIR / "images"
    annotations_dir = PROCESSED_DIR / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    by_split = {}
    for split, local_path in downloaded:
        by_split.setdefault(split, []).append(local_path)

    for split, paths in by_split.items():
        coco = {"images": [], "annotations": [], "categories": CATEGORIES}
        ann_id = 1
        img_count = 0
        stop_extraction = False

        for path in paths:
            if stop_extraction:
                break
            df = pd.read_parquet(path, columns=[
                "image", "bboxes", "category_id", "area", "metadata",
            ])
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{split}:{Path(path).name}"):
                if max_images is not None and img_count >= max_images:
                    stop_extraction = True
                    break

                meta = row["metadata"]
                image_id = int(meta["image_id"])
                file_name = f"{image_id}.png"

                img_bytes = row["image"]["bytes"] if isinstance(row["image"], dict) else row["image"]
                img = Image.open(io.BytesIO(img_bytes))
                img.save(images_dir / file_name)

                coco["images"].append({
                    "id": image_id,
                    "file_name": file_name,
                    "width": int(meta["coco_width"]),
                    "height": int(meta["coco_height"]),
                    "doc_category": meta["doc_category"],
                })

                for bbox, category_id, area in zip(row["bboxes"], row["category_id"], row["area"]):
                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": int(category_id),
                        "bbox": [float(x) for x in bbox],
                        "area": float(area),
                    })
                    ann_id += 1

                img_count += 1

        out_path = annotations_dir / f"{split}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(coco, f)
        print(f"[extract] {out_path} ({len(coco['images'])} anh, {len(coco['annotations'])} box)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-size-gb", type=float, default=1.0,
                         help="Dung luong toi da can tai, tinh bang GB (mac dinh 1.0)")
    parser.add_argument("--no-extract", action="store_true",
                         help="Chi tai raw parquet, khong convert sang processed/")
    parser.add_argument("--max-images", type=int, default=None,
                         help="So luong anh toi da can trich xuat (mac dinh: trich xuat tat ca)")
    parser.add_argument("--delete-raw", action="store_true",
                         help="Xoa file raw parquet sau khi trich xuat xong de tiet kiem o cung")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = download_shards(int(args.max_size_gb * 1e9))

    if not args.no_extract and downloaded:
        extract_processed(downloaded, max_images=args.max_images)
        if args.delete_raw:
            delete_raw_files(downloaded)


if __name__ == "__main__":
    main()
