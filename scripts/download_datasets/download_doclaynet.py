"""Tải bộ dữ liệu DocLayNet từ HuggingFace Hub (tăng tốc hf_transfer + pipeline song song)."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import sys
import time

# Bật hf_transfer (Rust multi-threaded download)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

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
    return processed_dir(split) / "annotations" / "_shards" / f"{Path(rfilename).stem}.json"


def list_shards(api: HfApi, splits: list) -> list:
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
            wait = 15 * attempt
            print(f"[download] Loi khi tai {rfilename} (lan {attempt}/{attempts}): {e}. Thu lai sau {wait}s...", flush=True)
            time.sleep(wait)


def delete_raw_file(local_path: Path) -> None:
    try:
        size = local_path.stat().st_size
        local_path.unlink()
        print(f"[cleanup] Da xoa: {local_path.name} ({size / 1e6:.1f} MB)", flush=True)
    except OSError as e:
        print(f"[cleanup] Khong the xoa {local_path.name}: {e}", flush=True)


def extract_shard(split: str, rfilename: str, local_path: Path, max_images: int | None = None) -> int:
    import pandas as pd

    images_dir = processed_dir(split) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(local_path, columns=["image", "bboxes", "category_id", "area", "metadata"])
    complete = max_images is None or max_images >= len(df)
    if not complete:
        df = df.iloc[:max_images]

    images, annotations, img_tasks = [], [], []
    for row in df.itertuples(index=False):
        meta = row.metadata
        image_id = int(meta["image_id"])
        file_name = f"{image_id}.png"

        img_bytes = row.image["bytes"] if isinstance(row.image, dict) else row.image
        img_tasks.append((images_dir / file_name, img_bytes))

        images.append({
            "id": image_id,
            "file_name": file_name,
            "width": int(meta["coco_width"]),
            "height": int(meta["coco_height"]),
            "doc_category": meta["doc_category"],
            "original_filename": meta.get("original_filename"),
            "page_no": int(meta["page_no"]) if "page_no" in meta and meta["page_no"] is not None else None,
            "num_pages": int(meta["num_pages"]) if "num_pages" in meta and meta["num_pages"] is not None else None,
            "collection": meta.get("collection"),
        })
        for bbox, category_id, area in zip(row.bboxes, row.category_id, row.area):
            annotations.append({
                "image_id": image_id,
                "category_id": int(category_id),
                "bbox": [float(x) for x in bbox],
                "area": float(area),
            })

    def _write_img(task):
        p, b = task
        if b[:8] == PNG_MAGIC:
            p.write_bytes(b)
        else:
            Image.open(io.BytesIO(b)).save(p)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_write_img, img_tasks))

    record = shard_record_path(split, rfilename)
    record.parent.mkdir(parents=True, exist_ok=True)
    tmp = record.with_suffix(".tmp")
    tmp.write_text(json.dumps({"complete": complete, "images": images, "annotations": annotations}), encoding="utf-8")
    os.replace(tmp, record)
    return len(images)


def merge_split(split: str) -> None:
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
    print(f"[merge] {out_path.name}: {len(coco['images']):,} anh, {len(coco['annotations']):,} box", flush=True)


def is_shard_complete(split: str, rfilename: str) -> bool:
    record = shard_record_path(split, rfilename)
    if not record.exists():
        return False
    return json.loads(record.read_text(encoding="utf-8")).get("complete", True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", default=True)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=["train", "validation", "test"])
    parser.add_argument("--delete-raw", action="store_true", default=True)
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    shards = list_shards(HfApi(), args.splits)

    pending = []
    for split, rfilename, size in shards:
        if is_shard_complete(split, rfilename):
            continue
        pending.append((split, rfilename, size))

    print(f"[pipeline] Co {len(pending)}/{len(shards)} shard can xu ly (tong ~{sum(s[2] for s in pending)/1e9:.1f} GB)...", flush=True)
    if not pending:
        print("[pipeline] Tat ca shard da hoan tat!")
        for split in args.splits:
            merge_split(split)
        return

    # Pipeline: Tai truoc shard tiep theo song song voi trich xuat
    with ThreadPoolExecutor(max_workers=1) as download_pool:
        next_download = download_pool.submit(download_shard, pending[0][1])

        for idx, (split, rfilename, size) in enumerate(pending):
            prefix = f"[{idx+1}/{len(pending)}] {Path(rfilename).name} ({size / 1e6:.0f} MB)"
            t0_step = time.perf_counter()

            local_path = next_download.result()

            if idx + 1 < len(pending):
                next_download = download_pool.submit(download_shard, pending[idx + 1][1])

            t0_ext = time.perf_counter()
            n_imgs = extract_shard(split, rfilename, local_path)
            t_ext = time.perf_counter() - t0_ext

            if args.delete_raw:
                delete_raw_file(local_path)

            t_total = time.perf_counter() - t0_step
            print(f"{prefix} -> Da trich {n_imgs} anh trong {t_ext:.2f}s (chu ky: {t_total:.1f}s)", flush=True)

            if (idx + 1) % 5 == 0 or idx + 1 == len(pending):
                merge_split(split)

    for split in args.splits:
        merge_split(split)
    print("[pipeline] Hoan thanh 100% toan bo du lieu!")


if __name__ == "__main__":
    main()
