"""Method 7 - Bước 0: Chạy Base RT-DETRv2 (Heron) trên toàn bộ trang train + test.

Đầu ra (methods/method7/cache/):
- base_preds_{split}.pt : {image_id: np.ndarray [N,6] (heron_cls, score, cx, cy, w, h)} với score >= 0.3
- heatmaps_{split}.npy  : memmap float16 [num_images, 22, 80, 80]  (11 Book Prior + 11 Page Prior)
- heatmap_index_{split}.json : {image_id: row}
- base_test_eval.json   : COCO mAP của Base trên 4,999 trang test (cùng cách chấm với Method 7)

Book Prior của trang i = trung bình heatmap dự đoán của các trang trong cửa sổ 64 trang
cùng cuốn sách, LOẠI TRỪ chính trang i. Page Prior = heatmap dự đoán của trang i.
Không dùng bất kỳ nhãn GT nào để tạo heatmap.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from methods.method7.common import (
    BASE_MODEL, CACHE_DIR, HEATMAP_SIZE, NUM_DLN_CLASSES, PRIOR_SCORE_THRESH, REPORT_DIR,
    TEST_ANNO, TEST_IMG, TRAIN_ANNO, TRAIN_IMG, WINDOW_SIZE,
    book_of, coco_eval, load_coco, load_image_tensor, render_pred_heatmap,
    to_coco_dets, topk_detections,
)


class ImgDS(Dataset):
    def __init__(self, images, img_dir):
        self.images, self.img_dir = images, img_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        im = self.images[i]
        t, (h, w) = load_image_tensor(self.img_dir / im["file_name"])
        return t, im["id"], h, w


def collate(b):
    return torch.stack([x[0] for x in b]), [x[1] for x in b], [x[2] for x in b], [x[3] for x in b]


@torch.no_grad()
def run_base(model, coco, img_dir, device, collect_eval: bool):
    loader = DataLoader(ImgDS(coco["images"], img_dir), batch_size=16, shuffle=False,
                        num_workers=4, collate_fn=collate, pin_memory=True)
    preds, dets = {}, []
    n, t_model = 0, 0.0
    t0 = time.perf_counter()
    for bi, (x, ids, hs, ws) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        torch.cuda.synchronize()
        tm = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(pixel_values=x)
        probs = out.logits.float().sigmoid()
        boxes = out.pred_boxes.float()
        torch.cuda.synchronize()
        t_model += time.perf_counter() - tm
        for i, iid in enumerate(ids):
            s, c = probs[i, :, :NUM_DLN_CLASSES].max(-1)
            keep = s >= PRIOR_SCORE_THRESH
            arr = torch.cat([c[keep, None].float(), s[keep, None], boxes[i][keep]], 1).cpu().numpy()
            preds[iid] = arr.astype(np.float32)
            if collect_eval:
                sc, cl, bx = topk_detections(probs[i], boxes[i], 100)
                dets += to_coco_dets(iid, sc.cpu(), cl.cpu(), bx.cpu(), ws[i], hs[i])
        n += len(ids)
        if (bi + 1) % 50 == 0:
            el = time.perf_counter() - t0
            print(f"  {n:,}/{len(loader.dataset):,} trang | {n/el:.1f} trang/s (end-to-end)")
    total = time.perf_counter() - t0
    return preds, dets, {"pages": n, "total_sec": total, "model_sec": t_model,
                         "fps_end_to_end": n / total, "fps_model_only": n / t_model}


def build_heatmaps(coco, preds, split):
    by_book = defaultdict(list)
    for im in coco["images"]:
        by_book[book_of(im)].append(im)
    index = {}
    rows = len(coco["images"])
    mm = np.lib.format.open_memmap(CACHE_DIR / f"heatmaps_{split}.npy", mode="w+",
                                   dtype=np.float16, shape=(rows, 2 * NUM_DLN_CLASSES, HEATMAP_SIZE, HEATMAP_SIZE))
    r = 0
    for book, pages in by_book.items():
        pages.sort(key=lambda x: x.get("page_no", 0))
        page_hm = np.stack([render_pred_heatmap(preds.get(p["id"], np.zeros((0, 6), np.float32))) for p in pages])
        csum = np.concatenate([np.zeros_like(page_hm[:1]), np.cumsum(page_hm, 0)], 0)
        for idx, p in enumerate(pages):
            s = max(0, idx - WINDOW_SIZE // 2)
            e = min(len(pages), idx + WINDOW_SIZE // 2 + 1)
            cnt = (e - s) - 1  # loại trừ chính trang hiện tại
            book_hm = (csum[e] - csum[s] - page_hm[idx]) / cnt if cnt > 0 else np.zeros_like(page_hm[idx])
            mm[r, :NUM_DLN_CLASSES] = book_hm.astype(np.float16)
            mm[r, NUM_DLN_CLASSES:] = page_hm[idx].astype(np.float16)
            index[p["id"]] = r
            r += 1
    mm.flush()
    with open(CACHE_DIR / f"heatmap_index_{split}.json", "w") as f:
        json.dump({str(k): v for k, v in index.items()}, f)
    print(f"✓ Heatmap {split}: {rows:,} trang -> heatmaps_{split}.npy")


def main():
    from transformers import RTDetrV2ForObjectDetection
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    model = RTDetrV2ForObjectDetection.from_pretrained(BASE_MODEL).to(device).eval()

    for split, anno, img_dir in [("test", TEST_ANNO, TEST_IMG), ("train", TRAIN_ANNO, TRAIN_IMG)]:
        print(f"\n=== Base RT-DETRv2 suy luận trên {split} ===")
        coco = load_coco(anno)
        preds, dets, timing = run_base(model, coco, img_dir, device, collect_eval=(split == "test"))
        torch.save(preds, CACHE_DIR / f"base_preds_{split}.pt")
        print(f"✓ {split}: {timing}")
        if split == "test":
            res = coco_eval(coco, dets)
            res["timing"] = timing
            res["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20
            with open(REPORT_DIR / "base_test_eval.json", "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
            print(f"✓ BASE trên 4,999 trang test: mAP={res['map']*100:.2f}  AP50={res['map50']*100:.2f}  AP75={res['map75']*100:.2f}")
        build_heatmaps(coco, preds, split)


if __name__ == "__main__":
    main()
