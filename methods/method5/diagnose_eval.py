import sys
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from methods.method5.models.rtdetr_method5 import BookLayoutRTDetrMethod5
from methods.method5.data.doclaynet_book_dataset import DocLayNetBookDataset, collate_book_fn
from methods.method5.models.evaluator import evaluate_coco_map, DOCLAYNET_CATEGORIES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_anno = ROOT_DIR / "datasets" / "doclaynet_test" / "processed" / "annotations" / "test.json"
val_img = ROOT_DIR / "datasets" / "doclaynet_test" / "processed" / "images"

print("=" * 80)
print("🔍 BẮT ĐẦU CHẨN ĐOÁN SỐ LIỆU VALIDATION EPOCH 1")
print("=" * 80)

# 1. Nạp checkpoint Epoch 1
model = BookLayoutRTDetrMethod5(num_classes=17, num_summary_tokens=8).to(device)
ckpt_path = ROOT_DIR / "methods" / "method5" / "weights" / "method5_epoch_1.pth"
if not ckpt_path.exists():
    ckpt_path = ROOT_DIR / "methods" / "method5" / "weights" / "method5_latest.pth"

print(f"Loading weights from: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# 2. Đánh giá trên tập 30 trang (như pilot run)
print("\n--- TEST TRÊN 30 TRANG ---")
ds_30 = DocLayNetBookDataset(annotation_file=val_anno, images_dir=val_img, image_size=640, window_size=64, limit_pages=30, is_train=False)
dl_30 = DataLoader(ds_30, batch_size=4, shuffle=False, collate_fn=collate_book_fn)
res_30_base = evaluate_coco_map(model, dl_30, device=device, use_fusion=False)
res_30_m5 = evaluate_coco_map(model, dl_30, device=device, use_fusion=True)
print(f"30 pages Base: mAP={res_30_base['map']*100:.2f}%, AP50={res_30_base['map50']*100:.2f}%")
print(f"30 pages M5:   mAP={res_30_m5['map']*100:.2f}%, AP50={res_30_m5['map50']*100:.2f}%")

# 3. Đánh giá trên tập 200 trang
print("\n--- TEST TRÊN 200 TRANG ---")
ds_200 = DocLayNetBookDataset(annotation_file=val_anno, images_dir=val_img, image_size=640, window_size=64, limit_pages=200, is_train=False)
dl_200 = DataLoader(ds_200, batch_size=4, shuffle=False, collate_fn=collate_book_fn)
res_200_base = evaluate_coco_map(model, dl_200, device=device, use_fusion=False)
res_200_m5 = evaluate_coco_map(model, dl_200, device=device, use_fusion=True)
print(f"200 pages Base: mAP={res_200_base['map']*100:.2f}%, AP50={res_200_base['map50']*100:.2f}%")
print(f"200 pages M5:   mAP={res_200_m5['map']*100:.2f}%, AP50={res_200_m5['map50']*100:.2f}%")

print("\n--- CHI TIẾT TỪNG CLASS TRÊN 200 TRANG (M5) ---")
for cat in DOCLAYNET_CATEGORIES:
    cid = cat["id"]
    cname = cat["name"]
    ap = res_200_m5["per_category"].get(cid, {}).get("ap", 0.0)
    print(f"  {cname:<18}: {ap*100:.2f}%")
