"""Ve vi du mot trang bi TACH (1 GT -> nhieu box du doan) hoac GOP (nhieu GT -> 1 box du doan).

Ben trai: Ground Truth, ben phai: du doan. Cac box thuoc nhom tach/gop duoc to dam
(xanh = GT, do = du doan), cac box con lai ve mo. Nhom tach/gop duoc xac dinh bang
`src.common.metrics.analyze_image` (cung dinh nghia voi bao cao phan tich loi).

Vi du:
    python visualization/scripts/visualize_split_merge.py --method dots_mocr --split test \
        --kind split --image-id 197
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.common.metrics import analyze_image  # noqa: E402
from src.common.splits import resolve  # noqa: E402

TITLES = {
    "split": "BỊ TÁCH: 1 GT bị chia thành nhiều box dự đoán",
    "merge": "BỊ GỘP: nhiều GT bị gộp thành 1 box dự đoán",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True)
    parser.add_argument("--split", default="test", help="train / valid / test")
    parser.add_argument("--kind", choices=["split", "merge"], required=True)
    parser.add_argument("--image-id", type=int, default=None,
                        help="Trang can ve; bo trong de lay trang it box nhat co nhom tach/gop tu 3 manh tro len")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    info = resolve(args.split)
    coco = json.loads((ROOT_DIR / "datasets" / info.dataset / "processed" / "annotations"
                       / f"{info.annotation}.json").read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in coco["categories"]}
    gt_by_image = {}
    for ann in coco["annotations"]:
        gt_by_image.setdefault(ann["image_id"], []).append(ann)
    pred_dir = ROOT_DIR / "outputs" / "predictions" / args.method / info.dataset

    def analyse(image_id):
        pred = json.loads((pred_dir / f"{image_id}.json").read_text(encoding="utf-8"))["detections"]
        gts = gt_by_image.get(image_id, [])
        result = analyze_image(gts, pred, iou_threshold=args.iou_threshold)
        group = next((g for g in result["groups"] if g["type"] == args.kind), None)
        return gts, pred, group

    if args.image_id is None:
        best = None
        for path in pred_dir.glob("*.json"):
            image_id = json.loads(path.read_text(encoding="utf-8"))["image_id"]
            gts, _, group = analyse(image_id)
            size = len(group["pred"] if args.kind == "split" else group["gt"]) if group else 0
            if size >= 3 and (best is None or len(gts) < best[0]):
                best = (len(gts), image_id)
        if best is None:
            raise SystemExit(f"Khong tim thay trang nao co nhom '{args.kind}' trong {pred_dir}")
        args.image_id = best[1]

    gts, pred, group = analyse(args.image_id)
    if group is None:
        raise SystemExit(f"Trang {args.image_id} khong co nhom '{args.kind}' o IoU {args.iou_threshold}")

    image_path = ROOT_DIR / "datasets" / info.dataset / "processed" / "images" / f"{args.image_id}.png"
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.8))
    panels = ((axes[0], gts, "Ground Truth", group["gt"], "#2563eb"),
              (axes[1], pred, f"{args.method} dự đoán", group["pred"], "#dc2626"))
    for ax, boxes, side, highlighted, color in panels:
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(f"{side}: {len(highlighted)} box trong nhóm (tô đậm)", fontsize=11)
        for index, box in enumerate(boxes):
            x, y, w, h = box["bbox"]
            inside = index in highlighted
            edge = color if inside else "#9ca3af"
            ax.add_patch(Rectangle((x, y), w, h, fill=inside, facecolor=edge, alpha=0.18 if inside else 0,
                                   edgecolor=edge, linewidth=2.6 if inside else 0.9))
            if inside:
                label = names.get(box.get("category_id"), box.get("category_name", ""))
                ax.text(x + 3, y + 12, label, color="white", fontsize=7.5,
                        bbox=dict(facecolor=edge, pad=1, edgecolor="none"))
    fig.suptitle(f"Ảnh {args.image_id} ({info.name}) — {TITLES[args.kind]}", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir = Path(args.output_dir) if args.output_dir else ROOT_DIR / "outputs" / "reports" / f"{args.method}_{info.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"case_{args.kind}_{args.image_id}.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[case] {args.kind}: trang {args.image_id}, {len(group['gt'])} GT <-> {len(group['pred'])} box du doan -> {out_path}")


if __name__ == "__main__":
    main()
