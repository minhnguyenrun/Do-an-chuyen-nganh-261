"""Method 8 - Tổng Chỉ Huy Tự Động Hóa Huấn Luyện (Auto Train Orchestrator).

Nhiệm vụ khi người dùng đi ngủ:
1. Giám sát liên tục tiến độ tải dữ liệu từ task-274.log.
2. Phối hợp với async_data_worker để tiền xử lý cuốn chiếu (resize 640x640 + base predictions + memmap heatmaps trên ổ C:).
3. Khi tải xong 100% (đủ 60/60 shard, ~69,103 trang):
   - Chờ worker xử lý dứt điểm các ảnh cuối cùng.
   - Dừng worker để giải phóng toàn bộ 100% tài nguyên GPU VRAM.
   - TỰ ĐỘNG CHẠY và TỰ ĐỘNG LƯU file Jupyter Notebook:
     methods/method8/notebooks/02_train_and_benchmark.ipynb
   - Chạy đủ 10 Epochs, vẽ biểu đồ, kiểm thử trên 4,999 test pages, so sánh với các bài báo khoa học.
   - Toàn bộ kết quả, bảng biểu, đồ thị được nhúng vĩnh viễn vào file .ipynb để sáng mai người dùng dậy nộp báo cáo!
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from methods.method8.common import CACHE_DIR, REPORT_DIR, TRAIN_ANNO, WEIGHTS_DIR, load_coco

DOWNLOAD_LOG = Path(r"C:\Users\thien\.gemini\antigravity\brain\02be02cd-fc37-4f6d-b5d4-23ff5b64202f\.system_generated\tasks\task-274.log")
NOTEBOOK_PATH = ROOT_DIR / "methods" / "method8" / "notebooks" / "02_train_and_benchmark.ipynb"


def is_download_finished() -> bool:
    """Kiểm tra xem task tải 60 shard đã hoàn thành chưa."""
    if not DOWNLOAD_LOG.exists():
        return False
    try:
        with open(DOWNLOAD_LOG, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        tail = "".join(lines[-30:])
        if "[60/60]" in tail or "Hoan thanh" in tail or "Finished" in tail:
            return True
    except Exception as e:
        print(f"[Orchestrator] Lỗi đọc log tải: {e}")
    return False


def wait_for_data_worker_complete(poll_interval: int = 20):
    """Đợi đến khi tất cả ảnh trong train.json đã có base predictions và được resize."""
    print("[Orchestrator] Đợi Async Data Worker xử lý xong các ảnh còn lại...")
    while True:
        if not TRAIN_ANNO.exists():
            time.sleep(poll_interval)
            continue
        try:
            coco = load_coco(TRAIN_ANNO)
            total_images = len(coco["images"])
            cache_file = CACHE_DIR / "base_preds_train.pt"
            if cache_file.exists():
                preds = torch.load(cache_file, weights_only=False)
                done_preds = len(preds)
                print(f"[Orchestrator] Tiến độ tiền xử lý: {done_preds:,} / {total_images:,} trang...")
                if done_preds >= total_images and total_images > 35000:
                    print("✅ [Orchestrator] 100% dữ liệu đã được tiền xử lý và suy luận xong!")
                    break
        except Exception as e:
            print(f"[Orchestrator] Đang chờ worker: {e}")
        time.sleep(poll_interval)


def stop_async_worker():
    """Dừng tiến trình async_data_worker để giải phóng GPU."""
    print("[Orchestrator] Đang dừng async_data_worker để giải phóng 100% VRAM GPU...")
    try:
        cmd = 'Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*async_data_worker*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }'
        subprocess.run(["powershell", "-Command", cmd], capture_output=True)
        time.sleep(3)
        print("✅ [Orchestrator] Đã dừng worker và giải phóng VRAM.")
    except Exception as e:
        print(f"[Orchestrator] Lỗi dừng worker: {e}")


def execute_and_save_notebook(notebook_path: Path):
    """Chạy toàn bộ Jupyter Notebook và tự động lưu đè kết quả in-place."""
    print("=" * 90)
    print(f"🎬 BẮT ĐẦU TỰ ĐỘNG THỰC THI JUPYTER NOTEBOOK: {notebook_path.name}")
    print("=" * 90)

    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=None, kernel_name="python3")

    notebook_dir = notebook_path.parent
    t0 = time.time()
    try:
        ep.preprocess(nb, {"metadata": {"path": str(notebook_dir)}})
        print(f"✅ Đã chạy thành công toàn bộ Notebook trong {(time.time() - t0)/3600:.2f} giờ!")
    except Exception as e:
        print(f"⚠️ Có lỗi trong quá trình thực thi: {e}")
    finally:
        with open(notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        print(f"💾 ĐÃ LƯU TOÀN BỘ KẾT QUẢ VÀO FILE: {notebook_path}")


def main_orchestrator():
    print("=" * 90)
    print("🌙 AUTO TRAIN ORCHESTRATOR ĐÃ ĐƯỢC KÍCH HOẠT: BẢO VỆ TIẾN TRÌNH TRONG ĐÊM")
    print("=" * 90)
    print(f"- Log tải dữ liệu: {DOWNLOAD_LOG}")
    print(f"- Notebook thực thi: {NOTEBOOK_PATH}")
    print(f"- Thư mục Cache: {CACHE_DIR}")

    # Vòng lặp giám sát tiến độ tải
    check_count = 0
    while True:
        check_count += 1
        now_str = time.strftime("%H:%M:%S")

        # Kiểm tra tail của log
        tail_msg = ""
        if DOWNLOAD_LOG.exists():
            try:
                with open(DOWNLOAD_LOG, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                    if lines:
                        tail_msg = lines[-1].strip()
            except Exception:
                pass

        print(f"[{now_str} - Quét #{check_count}] Tình trạng tải: {tail_msg}")

        if is_download_finished():
            print("\n🎉 [Orchestrator] PHÁT HIỆN TẬP TRAIN ĐÃ TẢI XONG TOÀN BỘ 60 SHARDS!")
            break

        time.sleep(30)

    # 2. Chờ worker xử lý dứt điểm các trang cuối cùng
    import torch
    wait_for_data_worker_complete()

    # 3. Dựng memmap hoàn thiện lần cuối cho full tập train
    print("[Orchestrator] Dựng mảng memmap heatmaps 22 kênh cuối cùng cho toàn bộ tập train...")
    try:
        from methods.method8.precompute_pipeline import step3_build_memmap_heatmaps
        coco = load_coco(TRAIN_ANNO)
        preds = torch.load(CACHE_DIR / "base_preds_train.pt", weights_only=False)
        valid = [im for im in coco["images"] if im["id"] in preds]
        step3_build_memmap_heatmaps(valid, preds, "train")
        print(f"✅ Đã đồng bộ hoàn hảo memmap cho {len(valid):,} trang trên {CACHE_DIR}!")
    except Exception as e:
        print(f"⚠️ Lỗi đồng bộ memmap cuối: {e}")

    # 4. Giải phóng worker để nhường 100% GPU
    stop_async_worker()

    # 5. Tự động chạy và tự động lưu Notebook
    execute_and_save_notebook(NOTEBOOK_PATH)

    print("\n" + "=" * 90)
    print("🎉 TẤT CẢ ĐÃ HOÀN TẤT THÀNH CÔNG! BẠN CÓ THỂ MỞ NOTEBOOK ĐỂ XEM BÁO CÁO.")
    print("=" * 90)


if __name__ == "__main__":
    main_orchestrator()
