"""Tải bộ dữ liệu DocLayNet từ HuggingFace Hub (một phần hoặc toàn bộ).

Dataset gốc (docling-project/DocLayNet-v1.2) được chia thành nhiều shard
.parquet cho mỗi split (train 72 / validation 7 / test 6 shard, ~400-520MB mỗi
shard, tổng ~40GB, 80,863 trang). Script tải các shard theo thứ tự split chỉ
định (mặc định train -> validation -> test) cho tới khi đạt dung lượng giới hạn
(mặc định 1GB, hoặc --all để tải hết), rồi (tuỳ chọn) trích ảnh + annotation ra
dạng COCO-style JSON.

image_id của DocLayNet đánh lại từ 0 trong MỖI split, nên mỗi split được lưu như
một "dataset" riêng để ảnh không ghi đè lên nhau:
    train      -> datasets/doclaynet/processed/       (annotations/train.json)
    validation -> datasets/doclaynet_val/processed/   (annotations/validation.json)
    test       -> datasets/doclaynet_test/processed/  (annotations/test.json)
=> chạy method/đánh giá trên test: --dataset doclaynet_test --split test

Mỗi shard được tải -> trích -> (tuỳ chọn) xoá raw ngay, nên --all --delete-raw
chỉ cần thêm ~0.5GB tạm ngoài dung lượng ảnh (~36GB). Shard đã trích xong được
ghi lại trong annotations/_shards/, chạy lại script sẽ bỏ qua (tiếp tục được khi
bị ngắt).

Usage:
    python download_doclaynet.py --max-size-gb 1
    python download_doclaynet.py --splits test --max-images 20 --delete-raw --output-dataset doclaynet_test
    python download_doclaynet.py --max-size-gb 0.5 --max-images 250 --delete-raw
    python download_doclaynet.py --all --splits test --delete-raw
    python download_doclaynet.py --all --splits test validation train --delete-raw
"""

import argparse
import io
import json
import os
from pathlib import Path
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from tqdm import tqdm

REPO_ID = "docling-project/DocLayNet-v1.2"
SPLITS = ["train", "validation", "test"]
SPLIT_DATASETS = {"train": "doclaynet", "validation": "doclaynet_val", "test": "doclaynet_test"}

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
DATASETS_DIR = (THIS_DIR / "../../datasets").resolve()
RAW_DIR = DATASETS_DIR / "doclaynet" / "raw"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def processed_dir(split: str) -> Path:
    return DATASETS_DIR / SPLIT_DATASETS[split] / "processed"


def shard_record_path(split: str, rfilename: str) -> Path:
    """File ghi annotation của 1 shard đã trích xong (đánh dấu shard hoàn tất)."""
    return processed_dir(split) / "annotations" / "_shards" / f"{Path(rfilename).stem}.json"


def list_shards(api: HfApi, splits: list) -> list:
    """Trả về danh sách (split, rfilename, size_bytes) theo thứ tự split chỉ định."""
    info = api.dataset_info(REPO_ID, files_metadata=True)
    by_split = {split: [] for split in splits}
    for sibling in info.siblings:
        for split in splits:
            if sibling.rfilename.startswith(f"data/{split}-"):
                by_split[split].append((sibling.rfilename, sibling.size or 0))
    shards = []
    for split in splits:
        for rfilename, size in sorted(by_split[split]):
            shards.append((split, rfilename, size))
    return shards


def download_shard(rfilename: str, attempts: int = 5) -> Path:
    """Tai 1 shard, thu lai khi loi mang tam thoi (HF CDN thinh thoang ngat ket noi)."""
    for attempt in range(1, attempts + 1):
        try:
            return Path(hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=rfilename,
                local_dir=str(RAW_DIR),
            ))
        except Exception as e:
            if attempt == attempts:
                raise
            wait = 30 * attempt
            print(f"[download] Loi khi tai {rfilename} (lan {attempt}/{attempts}): "
                  f"{type(e).__name__}: {str(e)[:200]}. Thu lai sau {wait}s...", flush=True)
            time.sleep(wait)


def delete_raw_file(local_path: Path) -> None:
    """Xoa file parquet da trich xong de giai phong dung luong o cung."""
    try:
        size = local_path.stat().st_size
        local_path.unlink()
        print(f"[cleanup] Da xoa file raw: {local_path.name} ({size / 1e6:.1f} MB)")
    except OSError as e:
        print(f"[cleanup] Khong the xoa {local_path.name}: {e}")


def extract_shard(split: str, rfilename: str, local_path: Path, max_images: int | None) -> int:
    """Trich anh (.png) + annotation cua 1 shard; ghi ra annotations/_shards/<shard>.json."""
    import pandas as pd

    images_dir = processed_dir(split) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(local_path, columns=["image", "bboxes", "category_id", "area", "metadata"])
    complete = max_images is None or max_images >= len(df)
    if not complete:
        df = df.iloc[:max_images]

    images, annotations = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{split}:{Path(rfilename).name}"):
        meta = row["metadata"]
        image_id = int(meta["image_id"])
        file_name = f"{image_id}.png"

        img_bytes = row["image"]["bytes"] if isinstance(row["image"], dict) else row["image"]
        if img_bytes[:8] == PNG_MAGIC:
            (images_dir / file_name).write_bytes(img_bytes)  # ảnh gốc đã là PNG, không cần encode lại
        else:
            Image.open(io.BytesIO(img_bytes)).save(images_dir / file_name)

        images.append({
            "id": image_id,
            "file_name": file_name,
            "width": int(meta["coco_width"]),
            "height": int(meta["coco_height"]),
            "doc_category": meta["doc_category"],
        })
        for bbox, category_id, area in zip(row["bboxes"], row["category_id"], row["area"]):
            annotations.append({
                "image_id": image_id,
                "category_id": int(category_id),
                "bbox": [float(x) for x in bbox],
                "area": float(area),
            })

    record = shard_record_path(split, rfilename)
    record.parent.mkdir(parents=True, exist_ok=True)
    tmp = record.with_suffix(".tmp")
    tmp.write_text(json.dumps({"complete": complete, "images": images, "annotations": annotations}), encoding="utf-8")
    os.replace(tmp, record)
    return len(images)


def merge_split(split: str) -> None:
    """Gop annotation cua moi shard da trich thanh annotations/<split>.json (COCO)."""
    records = sorted((processed_dir(split) / "annotations" / "_shards").glob("*.json"))
    coco = {"images": [], "annotations": [], "categories": CATEGORIES}
    for record in records:
        part = json.loads(record.read_text(encoding="utf-8"))
        coco["images"].extend(part["images"])
        for ann in part["annotations"]:
            coco["annotations"].append({"id": len(coco["annotations"]) + 1, **ann})

    out_path = processed_dir(split) / "annotations" / f"{split}.json"
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(coco), encoding="utf-8")
    os.replace(tmp, out_path)
    print(f"[extract] {out_path} ({len(coco['images'])} anh, {len(coco['annotations'])} box)")


def is_shard_complete(split: str, rfilename: str) -> bool:
    record = shard_record_path(split, rfilename)
    if not record.exists():
        return False
    # Ban ghi khong co "complete" van duoc coi la day du.
    return json.loads(record.read_text(encoding="utf-8")).get("complete", True)


def extracted_count(split: str) -> int:
    records = (processed_dir(split) / "annotations" / "_shards").glob("*.json")
    return sum(len(json.loads(r.read_text(encoding="utf-8"))["images"]) for r in records)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-size-gb", type=float, default=1.0,
                         help="Dung luong toi da can tai, tinh bang GB (mac dinh 1.0)")
    parser.add_argument("--all", action="store_true",
                         help="Tai toan bo cac split chi dinh, bo qua --max-size-gb")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=SPLITS,
                         help="Cac split can tai, theo thu tu (mac dinh: train validation test)")
    parser.add_argument("--no-extract", action="store_true",
                         help="Chi tai raw parquet, khong convert sang processed/")
    parser.add_argument("--max-images", type=int, default=None,
                         help="So luong anh toi da can trich xuat moi split (mac dinh: tat ca)")
    parser.add_argument("--delete-raw", action="store_true",
                         help="Xoa file raw parquet ngay sau khi trich xong tung shard")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    max_size_bytes = float("inf") if args.all else int(args.max_size_gb * 1e9)
    shards = list_shards(HfApi(), args.splits)
    print(f"[download] {len(shards)} shard trong {', '.join(args.splits)} "
          f"({sum(s for _, _, s in shards) / 1e9:.1f} GB)")

    total = 0
    for index, (split, rfilename, size) in enumerate(shards, start=1):
        if total >= max_size_bytes:
            break
        total += size
        prefix = f"[{index}/{len(shards)}] {rfilename} ({size / 1e6:.0f} MB)"

        if not args.no_extract and is_shard_complete(split, rfilename):
            print(f"{prefix}: da trich truoc do, bo qua")
            continue

        remaining = None
        if args.max_images is not None and not args.no_extract:
            remaining = args.max_images - extracted_count(split)
            if remaining <= 0:
                print(f"{prefix}: da du --max-images cho split {split}, bo qua")
                continue

        print(f"{prefix}: dang tai...", flush=True)
        local_path = download_shard(rfilename)
        if args.no_extract:
            continue

        extract_shard(split, rfilename, local_path, remaining)
        merge_split(split)
        if args.delete_raw:
            delete_raw_file(local_path)

    print(f"[download] Xong. Tong cong {min(total, max_size_bytes) / 1e9:.2f} GB shard da xu ly.")


if __name__ == "__main__":
    main()
