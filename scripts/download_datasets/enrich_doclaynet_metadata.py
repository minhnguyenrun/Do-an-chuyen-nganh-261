"""
Bổ sung metadata (original_filename, page_no, num_pages, collection)
vào các file COCO JSON của DocLayNet (test.json và train.json).

Script sử dụng đa luồng (multithreading) để tải nhanh metadata từ
Hugging Face datasets-server API mà không cần tải lại 40GB ảnh.
"""

import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

THIS_DIR = Path(__file__).resolve().parent
DATASETS_DIR = (THIS_DIR / "../../datasets").resolve()

DATASET_SPLITS = [
    {
        "split": "test",
        "json_path": DATASETS_DIR / "doclaynet_test" / "processed" / "annotations" / "test.json",
        "shards_dir": DATASETS_DIR / "doclaynet_test" / "processed" / "annotations" / "_shards",
    },
    {
        "split": "train",
        "json_path": DATASETS_DIR / "doclaynet" / "processed" / "annotations" / "train.json",
        "shards_dir": DATASETS_DIR / "doclaynet" / "processed" / "annotations" / "_shards",
    },
]

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def fetch_batch(split: str, offset: int, max_retries: int = 4):
    url = f"https://datasets-server.huggingface.co/rows?dataset=docling-project%2FDocLayNet-v1.2&config=default&split={split}&offset={offset}&limit=100"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return [r["row"]["metadata"] for r in data.get("rows", [])]
        except Exception as e:
            if attempt == max_retries:
                print(f"⚠️ Lỗi tải split={split}, offset={offset}: {e}", flush=True)
                return []
            time.sleep(2 * attempt)


def download_split_metadata(split: str, target_count: int, max_workers: int = 5):
    print(f"\n🔄 Đang tải metadata cho split '{split}' ({target_count:,} ảnh)...", flush=True)
    offsets = list(range(0, target_count + 100, 100))
    metadata_by_id = {}
    completed_batches = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_batch, split, off): off for off in offsets}
        for future in as_completed(futures):
            rows = future.result()
            for r in rows:
                img_id = int(r["image_id"])
                metadata_by_id[img_id] = {
                    "original_filename": r.get("original_filename"),
                    "page_no": int(r["page_no"]) if "page_no" in r and r["page_no"] is not None else None,
                    "num_pages": int(r["num_pages"]) if "num_pages" in r and r["num_pages"] is not None else None,
                    "collection": r.get("collection"),
                }
            completed_batches += 1
            if completed_batches % 10 == 0 or completed_batches == len(offsets):
                elapsed = time.time() - t0
                print(f"  [{split}] Tiến độ: {completed_batches}/{len(offsets)} batches ({len(metadata_by_id):,} ảnh đã map, {elapsed:.1f}s)", flush=True)

    print(f"✅ Hoàn thành tải metadata cho split '{split}': {len(metadata_by_id):,} ảnh ({time.time()-t0:.1f}s).", flush=True)
    return metadata_by_id


def enrich_json_file(json_path: Path, metadata_map: dict):
    if not json_path.exists():
        return 0
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    updated_count = 0
    for img in data.get("images", []):
        img_id = int(img["id"])
        if img_id in metadata_map:
            meta = metadata_map[img_id]
            img["original_filename"] = meta["original_filename"]
            img["page_no"] = meta["page_no"]
            img["num_pages"] = meta["num_pages"]
            img["collection"] = meta["collection"]
            updated_count += 1

    tmp_path = json_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp_path, json_path)
    return updated_count


def main():
    print("=" * 70)
    print(" BỔ SUNG METADATA SÁCH / FILE GỐC CHO TẬP DỮ LIỆU DOCLAYNET")
    print("=" * 70)

    for item in DATASET_SPLITS:
        split = item["split"]
        json_path = item["json_path"]
        shards_dir = item["shards_dir"]

        if not json_path.exists():
            print(f"Bỏ qua split '{split}': không tìm thấy {json_path}")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        total_images = len(data.get("images", []))

        # Tải metadata cho split này
        metadata_map = download_split_metadata(split, total_images)

        # Cập nhật file JSON chính
        print(f"💾 Đang ghi metadata vào file chính: {json_path.name}...")
        enriched = enrich_json_file(json_path, metadata_map)
        print(f"  ✅ Đã bổ sung metadata cho {enriched:,} / {total_images:,} ảnh trong {json_path.name}")

        # Cập nhật các shard files (nếu có)
        if shards_dir.exists():
            for shard_file in sorted(shards_dir.glob("*.json")):
                enrich_json_file(shard_file, metadata_map)
            print(f"  ✅ Đã cập nhật toàn bộ shard files trong {shards_dir.name}")

    print("\n🎉 TẤT CẢ DỮ LIỆU ĐÃ ĐƯỢC BỔ SUNG ĐẦY ĐỦ THÀNH CÔNG!")


if __name__ == "__main__":
    main()
