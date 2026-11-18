"""Method 8 - Tiến Trình Chạy Ngầm Xử Lý Dữ Liệu Tự Động (Async Data Worker).

Nhiệm vụ:
1. Theo dõi thư mục ảnh DocLayNet tải về theo thời gian thực.
2. Tự động Pre-resize các ảnh mới về 640x640 JPEG 95 lưu sang SSD (datasets/doclaynet_640/train/).
3. Tự động chạy Base RT-DETRv2 suy luận sinh Base Predictions cho các ảnh mới tải về.
4. Tự động cập nhật Heatmap 22 kênh memmap (Book Prior + Page Prior) cho tập ảnh hiện có.
5. Ghi cache nguyên tử (atomic save) an toàn, không xung đột với Jupyter Notebook.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import RTDetrV2ForObjectDetection

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from methods.method8.common import (
    BASE_MODEL, CACHE_DIR, DATA_640_DIR, HEATMAP_SIZE, IMAGE_SIZE, NUM_DLN_CLASSES,
    PRIOR_SCORE_THRESH, TRAIN_ANNO, TRAIN_IMG_640, TRAIN_IMG_SRC, WINDOW_SIZE,
    batch_resize_images, book_of, load_coco, load_image_640_tensor, render_pred_heatmap,
)


class Fast640Dataset(Dataset):
    def __init__(self, images: List[dict], img_640_dir: Path):
        self.images = images
        self.img_640_dir = img_640_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im = self.images[idx]
        stem = Path(im["file_name"]).stem
        img_p = self.img_640_dir / f"{stem}.jpg"
        if img_p.exists():
            t = load_image_640_tensor(img_p)
        else:
            t = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)
        h = float(im.get("height", 1025))
        w = float(im.get("width", 1025))
        return t, im["id"], h, w


def collate_fast(batch):
    return (
        torch.stack([b[0] for b in batch]),
        [b[1] for b in batch],
        [b[2] for b in batch],
        [b[3] for b in batch],
    )


def run_worker_loop(poll_interval: int = 15, batch_size: int = 16):
    print("=" * 80)
    print("🚀 KHỞI ĐỘNG WORKER CHẠY NGẦM: TỰ ĐỘNG TIỀN XỬ LÝ & SUY LUẬN DỮ LIỆU METHOD 8")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Thiết bị suy luận: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"[Paths] Nguồn ảnh thô: {TRAIN_IMG_SRC}")
    print(f"[Paths] Đích ảnh 640 SSD: {TRAIN_IMG_640}")
    print(f"[Paths] Cache dự đoán: {CACHE_DIR}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_IMG_640.mkdir(parents=True, exist_ok=True)

    cache_file = CACHE_DIR / "base_preds_train.pt"
    preds_dict: Dict[int, np.ndarray] = {}
    if cache_file.exists():
        try:
            preds_dict = torch.load(cache_file, weights_only=False)
            print(f"[Cache] Đã nạp {len(preds_dict):,} dự đoán có sẵn trong cache.")
        except Exception as e:
            print(f"[Cache] Chưa thể đọc cache: {e}")

    model = None

    cycle = 0
    while True:
        cycle += 1
        now_str = time.strftime("%H:%M:%S")
        print(f"\n--- [Vòng lặp #{cycle} lúc {now_str}] Đang quét dữ liệu mới tải về... ---")

        if not TRAIN_ANNO.exists():
            print(f"[Cảnh báo] Chưa tìm thấy file nhãn: {TRAIN_ANNO}. Chờ {poll_interval}s...")
            time.sleep(poll_interval)
            continue

        try:
            coco = load_coco(TRAIN_ANNO)
        except Exception as e:
            print(f"[Lỗi đọc nhãn JSON (đang ghi dở?)]: {e}. Chờ {poll_interval}s...")
            time.sleep(poll_interval)
            continue

        total_images_in_anno = len(coco["images"])

        # 1. Quét các file ảnh .png thực sự đã tải về ổ đĩa
        all_fnames = [im["file_name"] for im in coco["images"]]
        new_resized, already_resized = batch_resize_images(
            all_fnames, src_dir=TRAIN_IMG_SRC, dst_dir=TRAIN_IMG_640, max_workers=8, quality=95
        )
        if new_resized > 0:
            print(f"⚡ [Resize] Vừa resize thành công {new_resized:,} ảnh mới sang 640x640 JPEG 95!")
        print(f"📦 [SSD 640] Đã có sẵn: {already_resized + new_resized:,} / {total_images_in_anno:,} trang.")

        # 2. Lọc danh sách các ảnh đã resize nhưng CHƯA có trong base_preds_train.pt
        missing_infer = []
        for im in coco["images"]:
            pid = im["id"]
            stem = Path(im["file_name"]).stem
            img_640_file = TRAIN_IMG_640 / f"{stem}.jpg"
            if img_640_file.exists() and pid not in preds_dict:
                missing_infer.append(im)

        print(f"🔍 [Kiểm tra] Số ảnh sẵn sàng chờ Base RT-DETRv2 suy luận: {len(missing_infer):,} trang.")

        # 3. Chạy Base RT-DETRv2 suy luận nếu có ảnh mới
        if missing_infer:
            if model is None:
                print(f"[Model] Đang nạp mô hình Base RT-DETRv2 ({BASE_MODEL}) lên {device}...")
                model = RTDetrV2ForObjectDetection.from_pretrained(BASE_MODEL).to(device)
                model.eval()

            print(f"🧠 [Inference] Bắt đầu suy luận {len(missing_infer):,} ảnh (batch_size={batch_size})...")
            loader = DataLoader(
                Fast640Dataset(missing_infer, TRAIN_IMG_640),
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                collate_fn=collate_fast,
                pin_memory=True,
            )

            t0 = time.time()
            done_in_batch = 0
            with torch.no_grad():
                for pix, img_ids, hs, ws in loader:
                    pix = pix.to(device, non_blocking=True)
                    with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16):
                        out = model(pix)

                    logits = out.logits
                    boxes = out.pred_boxes
                    probs = torch.sigmoid(logits)[:, :, :NUM_DLN_CLASSES]

                    bs = pix.size(0)
                    for i in range(bs):
                        img_id = img_ids[i]
                        p_i = probs[i]
                        b_i = boxes[i]

                        mask = p_i >= PRIOR_SCORE_THRESH
                        if mask.any():
                            cls_ids, query_ids = torch.where(mask.t())
                            scores = p_i[query_ids, cls_ids]
                            matched_boxes = b_i[query_ids]
                            arr = torch.cat([
                                cls_ids.unsqueeze(1).float(),
                                scores.unsqueeze(1),
                                matched_boxes,
                            ], dim=1).cpu().numpy()
                        else:
                            arr = np.zeros((0, 6), dtype=np.float32)

                        preds_dict[img_id] = arr
                        done_in_batch += 1

                    if done_in_batch % 500 == 0 or done_in_batch == len(missing_infer):
                        fps = done_in_batch / max(time.time() - t0, 1e-4)
                        print(f"  -> Tiến độ: {done_in_batch}/{len(missing_infer)} ảnh ({fps:.1f} FPS)...")

            # Lưu cache nguyên tử (atomic write)
            tmp_cache = CACHE_DIR / "base_preds_train_tmp.pt"
            torch.save(preds_dict, tmp_cache)
            if tmp_cache.exists():
                os.replace(tmp_cache, cache_file)
            print(f"💾 [Cache] Đã lưu tổng cộng {len(preds_dict):,} dự đoán vào {cache_file.name}!")

            # 4. Tự động cập nhật Heatmap memmap khi có thêm dữ liệu đáng kể
            try:
                available_with_preds = [im for im in coco["images"] if im["id"] in preds_dict]
                print(f"🗺️ [Heatmap] Đang đồng bộ memmap cho {len(available_with_preds):,} trang...")
                from methods.method8.precompute_pipeline import step3_build_memmap_heatmaps
                step3_build_memmap_heatmaps(available_with_preds, base_preds=preds_dict, split="train")
                print(f"✅ [Heatmap] Hoàn tất đồng bộ memmap 22 kênh!")
            except Exception as e:
                print(f"⚠️ [Heatmap Sync Warning]: {e}")

        else:
            print(f"💤 [Idle] Chưa có thêm ảnh mới cần xử lý. Hiện đã hoàn thành: {len(preds_dict):,} trang.")

        # Nghỉ giữa các vòng lặp để tiếp tục đợi ảnh đang tải về
        print(f"⏳ Tạm nghỉ {poll_interval} giây trước vòng quét tiếp theo...")
        time.sleep(poll_interval)


if __name__ == "__main__":
    poll = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    run_worker_loop(poll_interval=poll)
