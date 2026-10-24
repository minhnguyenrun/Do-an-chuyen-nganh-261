import sys
import time
from pathlib import Path

# Add script dir to path
sys.path.insert(0, str(Path("scripts/download_datasets").resolve()))
import download_doclaynet as dd

TARGET_SHARDS = 11  # Shards 0 through 10 -> ~10,604 train images

def main():
    print(f"=== TẢI VÀ TRÍCH XUẤT DOCLAYNET TRAIN (MỤC TIÊU: {TARGET_SHARDS} SHARDS) ===", flush=True)
    shards = [f"data/train-{i:05d}-of-00072.parquet" for i in range(TARGET_SHARDS)]
    
    for i, rfilename in enumerate(shards):
        prefix = f"[{i+1}/{len(shards)}] {rfilename}"
        if dd.is_shard_complete("train", rfilename):
            print(f"{prefix}: Đã trích xuất trước đó, bỏ qua.", flush=True)
            continue
        
        t0 = time.time()
        print(f"\n{prefix}: Đang tải từ HuggingFace...", flush=True)
        local_path = dd.download_shard(rfilename)
        dl_time = time.time() - t0
        print(f"{prefix}: Tải xong ({local_path.stat().st_size / 1e6:.1f} MB) trong {dl_time:.1f}s. Đang trích xuất ảnh...", flush=True)
        
        t1 = time.time()
        count = dd.extract_shard("train", rfilename, local_path, max_images=None)
        dd.merge_split("train")
        dd.delete_raw_file(local_path)
        ex_time = time.time() - t1
        print(f"{prefix}: Trích xuất {count} ảnh thành công trong {ex_time:.1f}s.", flush=True)

    total_images = dd.extracted_count("train")
    print(f"\n=== HOÀN TẤT: Tổng cộng {total_images} ảnh train trong split 'train' ===", flush=True)

if __name__ == "__main__":
    main()
