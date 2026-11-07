"""Tải bộ dữ liệu DocLayNet từ HuggingFace Hub (một phần hoặc toàn bộ, hỗ trợ hf_transfer + pipeline song song).

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
    python download_doclaynet.py --lightweight --splits test --max-images 50 --sample-output-dir sample
    python download_doclaynet.py --max-size-gb 1
    python download_doclaynet.py --splits test --max-images 20 --delete-raw --output-dataset doclaynet_test
    python download_doclaynet.py --max-size-gb 0.5 --max-images 250 --delete-raw
    python download_doclaynet.py --all --splits test --delete-raw
    python download_doclaynet.py --all --splits test validation train --delete-raw
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

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


def read_sample_url(url: str, max_bytes: int = 25_000_000) -> bytes:
    """Bounded downloads for the small sample mode, never a parquet shard."""
    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": "DocLayNet-lightweight-sample/1.0"})
            with urlopen(request, timeout=60) as response:
                data = response.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise ValueError("Sample response is too large; refusing a large download")
            return data
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def download_lightweight_sample(split: str, count: int, output_dir: Path, *, offset: int = 0) -> Path:
    """Download selected viewer images + COCO boxes without fetching full parquet files.

    https://huggingface.co/docs/dataset-viewer/en/rows
    The viewer may serialize JPEG previews; save PNG and scale boxes to actual pixels.
    Checkpoint after every image so interrupted runs can resume.
    """
    if split not in SPLITS or type(count) is not int or not 1 <= count <= 1000 or offset < 0:
        raise ValueError("Sample requires a valid split, 1..1000 images and nonnegative offset")
    output_dir = Path(output_dir).resolve()
    images_dir = output_dir / "images"
    annotations_dir = output_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = annotations_dir / f"{split}.json"
    identity = {"source": REPO_ID, "mode": "lightweight", "split": split, "offset": offset}
    for other_path in annotations_dir.glob("*.json"):
        if other_path == manifest_path:
            continue
        other_info = json.loads(other_path.read_text(encoding="utf-8")).get("info", {})
        if other_info.get("mode") == "lightweight" and other_info.get("split") != split:
            raise ValueError("Sample directory belongs to another split; choose a different --sample-output-dir")
    coco = {"info": identity | {"requested_images": count}, "images": [], "annotations": [], "categories": CATEGORIES}
    if manifest_path.exists():
        cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(cached.get("info", {}).get(k) != v for k, v in identity.items()):
            raise ValueError("Sample directory belongs to another split/offset; choose a different --sample-output-dir")
        coco = cached
        coco["info"]["requested_images"] = count
        if len(coco["images"]) >= count and all((images_dir / image["file_name"]).is_file() for image in coco["images"][:count]):
            print(f"[sample] Da co {count} anh: {manifest_path}. Khong tai lai.")
            return manifest_path
    known = {image["id"]: image for image in coco["images"]}
    transferred = 0

    def checkpoint():
        tmp = manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, manifest_path)

    for start in range(offset, offset + count, 5):
        length = min(5, offset + count - start)
        query = urlencode({"dataset": REPO_ID, "config": "default", "split": split,
                           "offset": start, "length": length})
        payload = read_sample_url("https://datasets-server.huggingface.co/rows?" + query)
        transferred += len(payload)
        rows = json.loads(payload)["rows"]
        if len(rows) != length:
            raise ValueError("Dataset returned fewer rows than requested; lower --max-images or --sample-offset")
        for item in rows:
            row = item["row"]
            required = {"image", "metadata", "bboxes", "category_id"}
            if required.intersection(item.get("truncated_cells", [])):
                raise ValueError("Required sample fields were truncated by the viewer")
            meta = row["metadata"]
            image_id = int(meta["image_id"])
            file_name = f"{image_id}.png"
            path = images_dir / file_name
            if image_id in known and path.is_file():
                continue
            url = row["image"]["src"]
            if not url.startswith("https://"):
                raise ValueError("Sample image must use HTTPS")
            image_bytes = read_sample_url(url)
            transferred += len(image_bytes)
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.load()
                width, height = image.size
                png = io.BytesIO()
                image.convert("RGB").save(png, format="PNG")
            path.write_bytes(png.getvalue())
            if len(row["bboxes"]) != len(row["category_id"]):
                raise ValueError(f"Annotation length mismatch at image {image_id}")
            if image_id in known:
                # Re-downloaded a missing file: its original annotations already exist.
                continue
            sx, sy = width / int(meta["coco_width"]), height / int(meta["coco_height"])
            record = {"id": image_id, "file_name": file_name, "width": width, "height": height,
                      "doc_category": meta.get("doc_category", ""), "row_index": item["row_idx"]}
            coco["images"].append(record)
            known[image_id] = record
            for bbox, category_id in zip(row["bboxes"], row["category_id"]):
                x, y, w, h = (float(v) for v in bbox)
                coco["annotations"].append({"id": len(coco["annotations"]) + 1,
                    "image_id": image_id, "category_id": int(category_id),
                    "bbox": [x * sx, y * sy, w * sx, h * sy], "area": w * h * sx * sy})
            checkpoint()
            print(f"[sample] {len(coco['images'])}/{count} anh, da nhan {transferred / 1e6:.1f} MB", flush=True)
    checkpoint()
    print(f"[sample] Xong: {len(coco['images'])} anh + COCO annotations. Khong tai parquet shard.")
    return manifest_path


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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-size-gb", type=float, default=1.0,
                         help="Dung luong toi da can tai, tinh bang GB (mac dinh 1.0)")
    parser.add_argument("--all", action="store_true", default=True,
                         help="Tai toan bo cac split chi dinh, bo qua --max-size-gb")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=SPLITS,
                         help="Cac split can tai, theo thu tu (mac dinh: train validation test)")
    parser.add_argument("--no-extract", action="store_true",
                         help="Chi tai raw parquet, khong convert sang processed/")
    parser.add_argument("--max-images", type=int, default=None,
                         help="So luong anh toi da can trich xuat moi split (mac dinh: tat ca)")
    parser.add_argument("--delete-raw", action="store_true", default=True,
                         help="Xoa file raw parquet ngay sau khi trich xong tung shard")
    parser.add_argument("--lightweight", action="store_true",
                         help="Tai tung anh + annotation qua dataset viewer, khong tai parquet shard")
    parser.add_argument("--sample-output-dir", type=Path,
                         default=DATASETS_DIR / "doclaynet_sample" / "processed")
    parser.add_argument("--sample-offset", type=int, default=0)
    args = parser.parse_args()

    if args.lightweight:
        if len(args.splits) != 1 or args.no_extract:
            parser.error("--lightweight needs one --splits value and cannot use --no-extract")
        if args.max_images is None or not 1 <= args.max_images <= 1000 or args.sample_offset < 0:
            parser.error("--lightweight needs --max-images between 1 and 1000 and nonnegative --sample-offset")
        download_lightweight_sample(args.splits[0], args.max_images, args.sample_output_dir, offset=args.sample_offset)
        return

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
