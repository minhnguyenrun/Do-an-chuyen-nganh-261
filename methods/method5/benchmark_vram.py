"""Benchmark VRAM and Throughput for Method 5 with different batch sizes.
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from methods.method5.models.rtdetr_method5 import BookLayoutRTDetrMethod5
from methods.method5.data.doclaynet_book_dataset import DocLayNetBookDataset, collate_book_fn


def benchmark_batch_sizes(batch_sizes=[4, 8, 12], window_size=64, num_steps=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print("=" * 80)
    print(f"🚀 BENCHMARK VRAM & TỐC ĐỘ METHOD 5 TRÊN GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Context Window: {window_size} trang | Precision: AMP FP16 (Autocast)")
    print(f"   Tổng VRAM vật lý GPU: {total_mem:.2f} GB")
    print("=" * 80)

    train_anno = ROOT_DIR / "datasets" / "doclaynet" / "processed" / "annotations" / "train.json"
    train_img = ROOT_DIR / "datasets" / "doclaynet" / "processed" / "images"

    dataset = DocLayNetBookDataset(
        annotation_file=train_anno,
        images_dir=train_img,
        image_size=640,
        window_size=window_size,
        limit_pages=200,
        is_train=True,
    )

    results = []

    for bs in batch_sizes:
        print(f"\n--- Đang thử nghiệm với batch_size = {bs} ---")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, collate_fn=collate_book_fn)
        model = BookLayoutRTDetrMethod5(num_classes=17, num_summary_tokens=8).to(device)
        model.train()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")

        data_iter = iter(dataloader)
        
        # Warmup 2 steps
        try:
            for _ in range(2):
                batch = next(data_iter)
                pixel_values = batch["pixel_values"].to(device)
                labels = [{k: v.to(device) for k, v in l.items()} for l in batch["labels"]]
                context_elements = batch["context_elements"]

                optimizer.zero_grad()
                with torch.amp.autocast("cuda"):
                    outputs = model(
                        pixel_values=pixel_values,
                        labels=labels,
                        context_elements=context_elements,
                    )
                    loss = outputs["loss"]
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            t0 = time.time()
            total_images_processed = 0

            for step in range(num_steps):
                batch = next(data_iter)
                pixel_values = batch["pixel_values"].to(device)
                labels = [{k: v.to(device) for k, v in l.items()} for l in batch["labels"]]
                context_elements = batch["context_elements"]

                optimizer.zero_grad()
                with torch.amp.autocast("cuda"):
                    outputs = model(
                        pixel_values=pixel_values,
                        labels=labels,
                        context_elements=context_elements,
                    )
                    loss = outputs["loss"]
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_images_processed += len(pixel_values)

            torch.cuda.synchronize()
            elapsed = time.time() - t0
            fps = total_images_processed / elapsed
            ms_per_batch = (elapsed / num_steps) * 1000

            peak_alloc = torch.cuda.max_memory_allocated(device) / (1024**3)
            peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)

            print(f"   ✓ Thành công!")
            print(f"     Peak Allocated: {peak_alloc:.2f} GB | Peak Reserved: {peak_reserved:.2f} GB")
            print(f"     Throughput:     {fps:.1f} ảnh/giây ({ms_per_batch:.1f} ms/batch)")

            results.append({
                "batch_size": bs,
                "status": "PASS",
                "peak_alloc_gb": peak_alloc,
                "peak_reserved_gb": peak_reserved,
                "fps": fps,
                "ms_per_batch": ms_per_batch,
            })

        except torch.cuda.OutOfMemoryError as e:
            print(f"   ❌ CUDA Out Of Memory tại batch_size = {bs}!")
            results.append({
                "batch_size": bs,
                "status": "OOM",
                "peak_alloc_gb": -1,
                "peak_reserved_gb": -1,
                "fps": -1,
                "ms_per_batch": -1,
            })
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"   ❌ Lỗi: {e}")
            import traceback
            traceback.print_exc()
            break

    # Tổng kết bảng so sánh
    print("\n" + "=" * 80)
    print(f"{'Batch Size':<12} | {'VRAM Allocated':<16} | {'VRAM Reserved':<16} | {'Tốc độ (ảnh/s)':<16} | {'Trạng thái':<10}")
    print("-" * 80)
    for r in results:
        if r["status"] == "PASS":
            vram_pct = (r['peak_reserved_gb'] / total_mem) * 100
            print(f"{r['batch_size']:<12} | {r['peak_alloc_gb']:>13.2f} GB | {r['peak_reserved_gb']:>13.2f} GB ({vram_pct:.0f}%) | {r['fps']:>14.1f} | {'✅ An toàn':<10}")
        else:
            print(f"{r['batch_size']:<12} | {'OOM':>16} | {'OOM':>16} | {'-':>16} | {'❌ Quá tải':<10}")
    print("=" * 80)


if __name__ == "__main__":
    benchmark_batch_sizes(batch_sizes=[4, 8, 12])
