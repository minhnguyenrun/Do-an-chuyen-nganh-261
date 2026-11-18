"""Method 8 - Pipeline Tiền Xử Lý Gia Tăng (Incremental Fast Preprocessing).

Tính năng:
1. Pre-resize ảnh Offline sang SSD (640x640 JPEG 95) - tự động bỏ qua ảnh đã resize.
2. Suy luận Base RT-DETRv2 sinh Base Annotations - lưu cache, chỉ suy luận ảnh mới.
3. Sinh Heatmap 22 kênh memmap (11 Book Prior + 11 Page Prior) chuẩn khoa học không leak nhãn.
4. Tương thích hoàn toàn với Jupyter Notebook hoặc chạy qua dòng lệnh.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    PRIOR_SCORE_THRESH, TEST_ANNO, TEST_IMG_640, TEST_IMG_SRC, TRAIN_ANNO,
    TRAIN_IMG_640, TRAIN_IMG_SRC, VAL_ANNO, VAL_IMG_640, VAL_IMG_SRC, WINDOW_SIZE,
    book_of, load_coco, batch_resize_images, load_image_640_tensor, render_pred_heatmap,
)


class Fast640Dataset(Dataset):
    """Dataset đọc ảnh 640x640 trực tiếp từ SSD để suy luận Base RT-DETRv2."""
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


def step1_sync_and_resize_images(
    coco: dict,
    src_dir: Path,
    dst_dir: Path,
    max_workers: int = 8,
) -> Tuple[int, int, List[dict]]:
    """Bước 1: Quét và resize các ảnh khả dụng về 640x640 SSD."""
    all_fnames = [im["file_name"] for im in coco["images"]]
    print(f"[Step 1] Đang kiểm tra {len(all_fnames)} ảnh trong metadata...")
    new_done, already_done = batch_resize_images(
        all_fnames, src_dir=src_dir, dst_dir=dst_dir, max_workers=max_workers, quality=95
    )
    # Lọc danh sách các ảnh đã thực sự có file 640x640 sẵn sàng
    available_images = []
    for im in coco["images"]:
        stem = Path(im["file_name"]).stem
        if (dst_dir / f"{stem}.jpg").exists():
            available_images.append(im)

    print(f"[Step 1] Hoàn tất: {new_done} ảnh mới resize, {already_done} ảnh đã có sẵn.")
    print(f"[Step 1] Tổng số ảnh 640x640 sẵn sàng: {len(available_images)} / {len(coco['images'])}")
    return new_done, already_done, available_images


def step2_compute_base_predictions(
    images_to_run: List[dict],
    img_640_dir: Path,
    split: str,
    device: torch.device,
    batch_size: int = 16,
) -> Dict[int, np.ndarray]:
    """Bước 2: Chạy Base RT-DETRv2 sinh base predictions (hỗ trợ incremental cache)."""
    cache_file = CACHE_DIR / f"base_preds_{split}.pt"
    preds_dict: Dict[int, np.ndarray] = {}
    if cache_file.exists():
        try:
            preds_dict = torch.load(cache_file, weights_only=False)
            print(f"[Step 2] Đã nạp {len(preds_dict)} predictions có sẵn từ {cache_file.name}")
        except Exception as e:
            print(f"[Step 2] Không thể nạp cache cũ, sẽ chạy lại: {e}")

    # Lọc những ảnh chưa có trong cache
    missing_images = [im for im in images_to_run if im["id"] not in preds_dict]
    if not missing_images:
        print("[Step 2] Tất cả ảnh đã có base predictions. Bỏ qua chạy mô hình.")
        return preds_dict

    print(f"[Step 2] Cần chạy Base RT-DETRv2 trên {len(missing_images)} ảnh mới...")
    model = RTDetrV2ForObjectDetection.from_pretrained(BASE_MODEL).to(device)
    model.eval()

    loader = DataLoader(
        Fast640Dataset(missing_images, img_640_dir),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fast,
        pin_memory=True,
    )

    t0 = time.time()
    count = 0
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

                # Lọc box theo PRIOR_SCORE_THRESH để vẽ heatmap
                mask = p_i >= PRIOR_SCORE_THRESH
                if mask.any():
                    cls_ids, query_ids = torch.where(mask.t())
                    scores = p_i[query_ids, cls_ids]
                    matched_boxes = b_i[query_ids]
                    # N, 6: [cls, score, cx, cy, w, h]
                    arr = torch.cat([
                        cls_ids.unsqueeze(1).float(),
                        scores.unsqueeze(1),
                        matched_boxes,
                    ], dim=1).cpu().numpy()
                else:
                    arr = np.zeros((0, 6), dtype=np.float32)

                preds_dict[img_id] = arr
                count += 1

            if count % 500 == 0 or count == len(missing_images):
                fps = count / max(time.time() - t0, 1e-4)
                print(f"[Step 2] Đã xử lý {count}/{len(missing_images)} ảnh ({fps:.1f} FPS)...")

    # Lưu cập nhật cache
    torch.save(preds_dict, cache_file)
    print(f"[Step 2] Đã lưu tổng cộng {len(preds_dict)} predictions vào {cache_file}")
    return preds_dict


def step3_build_memmap_heatmaps(
    available_images: List[dict],
    base_preds: Dict[int, np.ndarray],
    split: str,
) -> Tuple[Path, Path]:
    """Bước 3: Dựng Heatmap 22 kênh (11 Book Prior + 11 Page Prior) lưu memmap float16."""
    print(f"[Step 3] Đang dựng heatmaps 22 kênh cho split '{split}'...")
    # Lọc những ảnh có trong base_preds
    valid_images = [im for im in available_images if im["id"] in base_preds]
    num_imgs = len(valid_images)

    # Gom theo sách
    books = defaultdict(list)
    for im in valid_images:
        books[book_of(im)].append(im)

    # Sắp xếp các trang trong từng sách theo page_no
    for bname in books:
        books[bname].sort(key=lambda x: x.get("page_no", 0))

    # Tính trước single page heatmap [11, 80, 80]
    single_hm: Dict[int, np.ndarray] = {}
    for im in valid_images:
        single_hm[im["id"]] = render_pred_heatmap(base_preds[im["id"]], grid=HEATMAP_SIZE)

    memmap_path = CACHE_DIR / f"heatmaps_{split}.npy"
    index_path = CACHE_DIR / f"heatmap_index_{split}.json"
    memmap_tmp = CACHE_DIR / f"heatmaps_{split}_tmp.npy"
    index_tmp = CACHE_DIR / f"heatmap_index_{split}_tmp.json"

    # Tạo memmap file trên file tạm tránh xung đột lock file trên Windows
    hm_mmap = np.lib.format.open_memmap(
        memmap_tmp,
        mode="w+",
        dtype=np.float16,
        shape=(num_imgs, 22, HEATMAP_SIZE, HEATMAP_SIZE),
    )

    idx_map = {}
    row = 0
    for bname, pages in books.items():
        n = len(pages)
        p_hms = np.stack([single_hm[p["id"]] for p in pages], axis=0)  # [n, 11, 80, 80]

        for i, page in enumerate(pages):
            # Book Prior = Trung bình cửa sổ WINDOW_SIZE, loại trừ chính trang i
            i0 = max(0, i - WINDOW_SIZE // 2)
            i1 = min(n, i + WINDOW_SIZE // 2 + 1)
            neighbor_mask = np.ones(i1 - i0, dtype=bool)
            neighbor_mask[i - i0] = False

            if neighbor_mask.any():
                book_prior = p_hms[i0:i1][neighbor_mask].mean(axis=0)
            else:
                book_prior = np.zeros((NUM_DLN_CLASSES, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)

            page_prior = p_hms[i]
            # Ghép 11 kênh Book Prior + 11 kênh Page Prior -> 22 kênh
            combined = np.concatenate([book_prior, page_prior], axis=0)
            hm_mmap[row] = combined.astype(np.float16)
            idx_map[int(page["id"])] = row
            row += 1

    hm_mmap.flush()
    del hm_mmap

    with open(index_tmp, "w", encoding="utf-8") as f:
        json.dump(idx_map, f)

    # Thay thế an toàn
    try:
        if memmap_path.exists():
            memmap_path.unlink()
        memmap_tmp.replace(memmap_path)
        if index_path.exists():
            index_path.unlink()
        index_tmp.replace(index_path)
    except Exception as e:
        print(f"[Warning] File gốc đang mở bởi kernel khác, giữ file tmp: {e}")
        return memmap_tmp, index_tmp

    print(f"[Step 3] Đã lưu {num_imgs} heatmaps vào {memmap_path.name} ({memmap_path.stat().st_size / (1024**2):.1f} MB)")
    print(f"[Step 3] Đã lưu heatmap index vào {index_path.name}")
    return memmap_path, index_path


def run_full_precompute_pipeline(split: str = "train", max_images: Optional[int] = None):
    """Hàm chạy trọn gói pipeline từ Step 1 đến Step 3."""
    print(f"==================================================")
    print(f"BẮT ĐẦU PIPELINE TIỀN XỬ LÝ METHOD 8 (Split: {split})")
    print(f"==================================================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị: {device}")

    if split == "train":
        coco = load_coco(TRAIN_ANNO)
        src_dir = TRAIN_IMG_SRC
        dst_dir = TRAIN_IMG_640
    elif split == "val":
        coco = load_coco(VAL_ANNO)
        src_dir = VAL_IMG_SRC
        dst_dir = VAL_IMG_640
    else:
        coco = load_coco(TEST_ANNO)
        src_dir = TEST_IMG_SRC
        dst_dir = TEST_IMG_640

    if max_images is not None:
        coco["images"] = coco["images"][:max_images]

    # Step 1
    new_done, already_done, available_images = step1_sync_and_resize_images(
        coco, src_dir=src_dir, dst_dir=dst_dir
    )

    # Step 2
    base_preds = step2_compute_base_predictions(
        available_images, img_640_dir=dst_dir, split=split, device=device
    )

    # Step 3
    step3_build_memmap_heatmaps(
        available_images, base_preds=base_preds, split=split
    )

    print(f"==================================================")
    print(f"HOÀN THÀNH TIỀN XỬ LÝ CHO {len(available_images)} TRANG '{split}'!")
    print(f"==================================================")


if __name__ == "__main__":
    split_arg = sys.argv[1] if len(sys.argv) > 1 else "train"
    run_full_precompute_pipeline(split=split_arg)
