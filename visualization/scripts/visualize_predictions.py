"""Ve bounding box Ground Truth vs. du doan cua mot method, canh nhau, de
dua vao bao cao.

Doc annotation goc tu datasets/<dataset>/processed/annotations/<split>.json
va ket qua du doan tu outputs/predictions/<method>/<dataset>/<image_id>.json,
roi ve 2 panel (Ground Truth | <Method>) len cung mot anh, luu vao
visualization/figures/<method>_<dataset>/<image_id>.png

Cai tien:
- Nhom cac box trung khop / gan trung toa do de the hien box da nhan (multi-label) ro rang.
- Hien thi badge dang vien thuoc (pill) co mau tuong ung tung nhan, chu tuong phan cao (contrast-aware).
- Tu dong tranh de chu / chong lap nhan giua cac box gan nhau (collision avoidance).
- Hien thi score tin cay neu co.

Usage:
    python visualization/scripts/visualize_predictions.py --method docling --dataset doclaynet --limit 10
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

from PIL import Image, ImageDraw, ImageFont

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR / "../.."

CATEGORY_NAMES = {
    1: "Caption", 2: "Footnote", 3: "Formula", 4: "List-item",
    5: "Page-footer", 6: "Page-header", 7: "Picture", 8: "Section-header",
    9: "Table", 10: "Text", 11: "Title",
}

# Bang mau co dinh cho 11 nhan DocLayNet
CATEGORY_COLORS = {
    1: (230, 25, 75), 2: (60, 180, 75), 3: (255, 225, 25), 4: (0, 130, 200),
    5: (245, 130, 48), 6: (145, 30, 180), 7: (70, 240, 240), 8: (240, 50, 230),
    9: (210, 245, 60), 10: (0, 128, 128), 11: (170, 110, 40),
}

NAME_TO_CATEGORY_ID = {
    "caption": 1, "footnote": 2, "formula": 3, "list_item": 4, "list-item": 4,
    "page_footer": 5, "page-footer": 5, "footer": 5,
    "page_header": 6, "page-header": 6, "header": 6,
    "picture": 7, "image": 7, "figure": 7,
    "section_header": 8, "section-header": 8, "paragraph_title": 8,
    "table": 9,
    "text": 10,
    "title": 11, "doc_title": 11,
}

UNKNOWN_COLOR = (128, 128, 128)
PAD = 40


def load_font(size: int = 13) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def get_contrast_text_color(rgb_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Chon mau chu (den/trang) dam bao do tuong phan cao nhat tren mau nen."""
    r, g, b = rgb_color[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 145 else (255, 255, 255)


def group_overlapping_boxes(boxes: list, iou_thresh: float = 0.70, dist_thresh: float = 8.0) -> list:
    """Nhom cac bounding box trung khop hoac co do chong lap rat cao (IoU >= 0.7 hoac
    sai lech toa do <= 8px), de bieu dien thanh 1 box mang nhieu nhan (multi-label)."""
    sorted_boxes = sorted(
        boxes,
        key=lambda b: float(b.get("score", 0)) if isinstance(b.get("score"), (int, float)) else 0.0,
        reverse=True
    )
    groups = []
    for b in sorted_boxes:
        bx = b["bbox"]
        bw, bh = bx[2], bx[3]
        matched_group = None
        for g in groups:
            gx = g["bbox"]
            gw, gh = gx[2], gx[3]
            x1 = max(bx[0], gx[0])
            y1 = max(bx[1], gx[1])
            x2 = min(bx[0] + bw, gx[0] + gw)
            y2 = min(bx[1] + bh, gx[1] + gh)
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            union = (bw * bh) + (gw * gh) - inter
            iou = (inter / union) if union > 0 else 0.0
            coord_diff = max(abs(bx[0] - gx[0]), abs(bx[1] - gx[1]), abs(bw - gw), abs(bh - gh))
            if iou >= iou_thresh or coord_diff <= dist_thresh:
                matched_group = g
                break
        if matched_group:
            matched_group["items"].append(b)
        else:
            groups.append({"bbox": list(bx), "items": [b]})
    return groups


def extract_group_labels(items: list) -> list:
    """Trich xuat va loai bo nhan trung trong cung mot nhom box, sap xep theo score giam dan."""
    seen = set()
    labels = []
    for it in items:
        cid = it.get("category_id") or 0
        cname = it.get("category_name")
        if not cname and cid in CATEGORY_NAMES:
            cname = CATEGORY_NAMES[cid]
        if not cname:
            cname = "unknown"

        if cid not in CATEGORY_COLORS:
            cid_fallback = NAME_TO_CATEGORY_ID.get(cname.lower().replace(" ", "_"), 0)
            color = CATEGORY_COLORS.get(cid_fallback, UNKNOWN_COLOR)
        else:
            color = CATEGORY_COLORS[cid]

        score = it.get("score")
        score_val = float(score) if isinstance(score, (int, float)) else None

        norm_key = cname.lower().replace("-", "_").replace(" ", "_")
        if norm_key in seen:
            continue
        seen.add(norm_key)

        score_str = f" {score_val:.2f}" if score_val is not None else ""
        text = f"{cname}{score_str}"
        labels.append({
            "text": text,
            "name": cname,
            "color": color,
            "text_color": get_contrast_text_color(color),
            "score": score_val if score_val is not None else -1.0,
        })
    labels.sort(key=lambda x: x["score"], reverse=True)
    return labels


def draw_boxes(image: Image.Image, boxes: list, font: ImageFont.ImageFont,
               box_only: bool = False, color: Tuple[int, int, int] | None = None) -> Image.Image:
    """Ve cac bounding box len anh voi xu ly chong de chu va ho tro box mang nhieu nhan.
    Neu box_only=True, chi ve vien bounding box va diem tin cay (neu co), bo qua nhan class."""
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    img_w, img_h = image.size

    if box_only:
        default_color = color if color is not None else (37, 99, 235)
        for b in boxes:
            bx = b["bbox"] if isinstance(b, dict) else b
            x, y, w, h = bx
            draw.rectangle([x, y, x + w, y + h], outline=default_color, width=2)
            if isinstance(b, dict) and b.get("score") is not None:
                score_val = float(b["score"])
                text = f"{score_val:.2f}"
                tb = font.getbbox(text)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
                badge_w, badge_h = tw + 6, th + 4
                badge_x = max(2, min(int(x), img_w - badge_w - 2))
                badge_y = max(2, int(y) - badge_h - 2) if int(y) - badge_h - 2 >= 0 else int(y) + 2
                draw.rounded_rectangle([badge_x, badge_y, badge_x + badge_w, badge_y + badge_h], radius=2, fill=default_color)
                draw.text((badge_x + 3, badge_y + 2), text, fill=(255, 255, 255), font=font)
        return out

    groups = group_overlapping_boxes(boxes)
    placed_badges: List[Tuple[float, float, float, float]] = []
    pad_x, pad_y = 4, 2
    pill_spacing = 3

    for g in groups:
        bx, by, bw, bh = g["bbox"]
        labels = extract_group_labels(g["items"])
        if not labels:
            continue

        primary_color = labels[0]["color"]
        draw.rectangle([bx, by, bx + bw, by + bh], outline=primary_color, width=2)

        # Do kich thuoc tung pill nhan
        measured = []
        for l in labels:
            bbox = font.getbbox(l["text"])
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            pw = tw + pad_x * 2
            ph = th + pad_y * 2
            measured.append({**l, "pw": pw, "ph": ph, "tw": tw, "th": th})

        max_row_w = max(bw, 180)
        rows = []
        current_row = []
        current_row_w = 0
        for p in measured:
            if current_row and (current_row_w + pill_spacing + p["pw"] > max_row_w):
                rows.append(current_row)
                current_row = [p]
                current_row_w = p["pw"]
            else:
                if current_row:
                    current_row_w += pill_spacing + p["pw"]
                else:
                    current_row_w = p["pw"]
                current_row.append(p)
        if current_row:
            rows.append(current_row)

        cluster_w = max(sum(p["pw"] for p in r) + (len(r) - 1) * pill_spacing for r in rows)
        row_h = max(p["ph"] for r in rows for p in r)
        cluster_h = len(rows) * row_h + (len(rows) - 1) * 2

        cand_x = max(2, min(int(bx), img_w - cluster_w - 2))

        # Cac vi tri tiem nang de dat badge, tranh de len nhau
        candidates = [
            (cand_x, by - cluster_h - 2),          # 1. Phia tren box
            (cand_x, by + 2),                      # 2. Ben trong tren cung box
            (cand_x, by + bh + 2),                 # 3. Phia duoi box
        ]

        # Them cac vi tri tranh badge da ve gan nhat
        for rx1, ry1, rx2, ry2 in placed_badges[-4:]:
            candidates.append((cand_x, ry1 - cluster_h - 2))
            candidates.append((cand_x, ry2 + 2))
            candidates.append((min(img_w - cluster_w - 2, max(2, rx2 + 2)), max(0, by - cluster_h - 2)))

        best_pos = None
        min_overlap = float("inf")

        for cx, cy in candidates:
            if cy < 0 or cy + cluster_h > img_h:
                continue

            total_overlap = 0
            for rx1, ry1, rx2, ry2 in placed_badges:
                ix1 = max(cx, rx1)
                iy1 = max(cy, ry1)
                ix2 = min(cx + cluster_w, rx2)
                iy2 = min(cy + cluster_h, ry2)
                if ix2 > ix1 and iy2 > iy1:
                    total_overlap += (ix2 - ix1) * (iy2 - iy1)

            if total_overlap == 0:
                best_pos = (cx, cy)
                break
            elif total_overlap < min_overlap:
                min_overlap = total_overlap
                best_pos = (cx, cy)

        if best_pos is None:
            best_pos = (cand_x, max(0, min(img_h - cluster_h, by - cluster_h - 2)))

        start_x, start_y = best_pos

        # Ve cac vien thuoc badge va chu
        curr_y = start_y
        for r in rows:
            curr_x = start_x
            for p in r:
                draw.rounded_rectangle(
                    [curr_x, curr_y, curr_x + p["pw"], curr_y + p["ph"]],
                    radius=3,
                    fill=p["color"],
                    outline=(30, 30, 30),
                    width=1
                )
                draw.text(
                    (curr_x + pad_x, curr_y + pad_y),
                    p["text"],
                    fill=p["text_color"],
                    font=font
                )
                curr_x += p["pw"] + pill_spacing
            curr_y += row_h + 2

        placed_badges.append((start_x, start_y, start_x + cluster_w, start_y + cluster_h))

    return out


def make_side_by_side(left: Image.Image, right: Image.Image, left_title: str, right_title: str) -> Image.Image:
    font = load_font(18)
    w, h = left.size
    canvas = Image.new("RGB", (w * 2 + PAD, h + PAD), (255, 255, 255))
    canvas.paste(left, (0, PAD))
    canvas.paste(right, (w + PAD, PAD))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), left_title, fill=(0, 0, 0), font=font)
    draw.text((w + PAD + 4, 4), right_title, fill=(0, 0, 0), font=font)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="docling", help="Ten method (vd: docling)")
    parser.add_argument("--dataset", default="doclaynet", help="Ten dataset (vd: doclaynet)")
    parser.add_argument("--split", default="train", help="Split cua ground truth (vd: train)")
    parser.add_argument("--limit", type=int, default=10, help="So anh toi da se ve (mac dinh 10)")
    parser.add_argument(
        "--box-only",
        action="store_true",
        help="Chỉ trực quan hóa Bounding Box thuần túy, bỏ qua nhãn class.",
    )
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Thu muc output (mac dinh visualization/figures/<method>_<dataset>)")
    args = parser.parse_args()

    images_dir = ROOT_DIR / "datasets" / args.dataset / "processed" / "images"
    predictions_dir = ROOT_DIR / "outputs" / "predictions" / args.method / args.dataset
    gt_path = ROOT_DIR / "datasets" / args.dataset / "processed" / "annotations" / f"{args.split}.json"
    output_dir = args.output_dir or ROOT_DIR / "visualization" / "figures" / f"{args.method}_{args.dataset}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    gt_by_image = {}
    for ann in gt["annotations"]:
        gt_by_image.setdefault(ann["image_id"], []).append(ann)

    prediction_paths = sorted(predictions_dir.glob("*.json"),
                               key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    if args.limit > 0:
        prediction_paths = prediction_paths[: args.limit]

    font = load_font(13)
    print(f"[viz] Ve {len(prediction_paths)} anh ({args.method} vs ground truth | box_only={args.box_only}) tu {predictions_dir}")
    for pred_path in prediction_paths:
        with open(pred_path, "r", encoding="utf-8") as f:
            pred = json.load(f)
        image_id = pred["image_id"]
        image_path = images_dir / pred["file_name"]
        if not image_path.exists():
            print(f"  [skip] khong tim thay anh {image_path}")
            continue

        base = Image.open(image_path)
        gt_boxes = gt_by_image.get(image_id, [])
        left = draw_boxes(base, gt_boxes, font, box_only=args.box_only, color=(16, 185, 129))
        right = draw_boxes(base, pred["detections"], font, box_only=args.box_only, color=(37, 99, 235))
        combined = make_side_by_side(
            left, right,
            f"Ground Truth ({len(gt_boxes)} boxes)" if args.box_only else f"Ground Truth ({len(gt_boxes)} box)",
            f"{args.method} ({len(pred['detections'])} boxes)" if args.box_only else f"{args.method} ({len(pred['detections'])} box)",
        )

        out_path = output_dir / f"{image_id}.png"
        combined.save(out_path)
        print(f"  [ok] {out_path.name}")

    print(f"[viz] Xong. Anh luu tai {output_dir}")


if __name__ == "__main__":
    main()

