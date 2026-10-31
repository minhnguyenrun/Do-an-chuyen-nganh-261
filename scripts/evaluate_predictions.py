"""Evaluate layout predictions against COCO-style ground-truth annotations."""

import argparse
import json
import sys
from collections import Counter
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.common.metrics import (  # noqa: E402
    BACKGROUND,
    GT_FATES,
    PRED_FATES,
    coco_map,
    equivalent_jitter_px,
    error_analysis,
    error_analysis_multi_iou,
    jitter_ceiling,
    evaluate,
    evaluate_boxes,
    evaluate_boxes_multi_iou,
    coco_box_map,
)
from src.common.profiling import load_runtime_runs, page_time_stats  # noqa: E402
from src.common.splits import SPLITS, normalize_split, resolve, split_of_dataset  # noqa: E402


def load_ground_truth(annotation_path: Path) -> tuple[dict, dict, dict]:
    raw_coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    category_names = {category["id"]: category["name"] for category in raw_coco["categories"]}
    ground_truth = {image["id"]: [] for image in raw_coco["images"]}

    for annotation in raw_coco["annotations"]:
        ground_truth[annotation["image_id"]].append(
            {
                "category_id": annotation["category_id"],
                "bbox": annotation["bbox"],
            }
        )

    return raw_coco, ground_truth, category_names


def load_predictions(predictions_dir: Path) -> tuple[dict, dict]:
    """Tra ve ({image_id: detections}, {image_id: inference_time_s})."""
    predictions = {}
    page_seconds = {}
    for path in predictions_dir.glob("*.json"):
        prediction = json.loads(path.read_text(encoding="utf-8"))
        image_id = prediction["image_id"]
        predictions[image_id] = prediction["detections"]
        if prediction.get("inference_time_s") is not None:
            page_seconds[image_id] = prediction["inference_time_s"]
    return predictions, page_seconds


def macro_scores(result: dict, category_names: dict) -> dict:
    class_count = sum(
        result["per_category"].get(category_id, {}).get("support", 0) > 0
        for category_id in category_names
    )
    return {
        metric: sum(
            result["per_category"].get(category_id, {}).get(metric, 0.0)
            for category_id in category_names
            if result["per_category"].get(category_id, {}).get("support", 0) > 0
        )
        / class_count
        if class_count
        else 0.0
        for metric in ("precision", "recall", "f1")
    }


def print_report(result: dict, category_names: dict, title: str) -> None:
    overall = result["overall"]
    macro = macro_scores(result, category_names)
    macro_class_count = sum(
        result["per_category"].get(category_id, {}).get("support", 0) > 0
        for category_id in category_names
    )
    print(f"\n=== {title} ===")
    print(
        "Micro: "
        f"Precision={overall['precision']:.3f}  Recall={overall['recall']:.3f}  "
        f"F1={overall['f1']:.3f}  "
        f"TP={overall['tp']} FP={overall['fp']} FN={overall['fn']}  "
        f"mean IoU={overall['mean_iou_matched']:.3f}"
    )
    print(
        f"Macro (mean across {macro_class_count} ground-truth classes): "
        f"Precision={macro['precision']:.3f}  Recall={macro['recall']:.3f}  "
        f"F1={macro['f1']:.3f}"
    )
    print(
        f"{'Class':<18}{'GT':>8}{'Pred':>8}{'TP':>7}{'FP':>7}{'FN':>7}"
        f"{'Prec':>9}{'Recall':>9}{'F1':>9}"
    )

    for category_id, name in sorted(
        category_names.items(),
        key=lambda item: (
            -result["per_category"].get(item[0], {}).get("support", 0),
            item[1],
        ),
    ):
        metrics = result["per_category"].get(
            category_id,
            {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "support": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
        )
        pred_count = result["pred_counts"].get(category_id, 0)
        print(
            f"{name:<18}{metrics['support']:>8}{pred_count:>8}{metrics['tp']:>7}"
            f"{metrics['fp']:>7}{metrics['fn']:>7}{metrics['precision']:>9.3f}"
            f"{metrics['recall']:>9.3f}{metrics['f1']:>9.3f}"
        )


GT_FATE_LABELS = {
    "correct": "Đúng", "cls_wrong": "SaiNhãn", "loc": "Lệch", "both": "Lệch+Sai",
    "split": "BịTách", "merged": "BịGộp", "complex": "PhứcTạp", "missed": "BỏSót",
}
PRED_FATE_LABELS = {
    "correct": "Đúng", "cls_wrong": "SaiNhãn", "loc": "Lệch", "both": "Lệch+Sai",
    "duplicate": "Trùng", "split_fragment": "MảnhTách", "merge_box": "BoxGộp",
    "complex": "PhứcTạp", "background": "Thừa",
}


def _pct(x: float) -> str:
    return f"{x * 100:6.2f}%"


def print_error_analysis(result: dict, category_names: dict, title: str) -> None:
    """In báo cáo 2 giai đoạn + tách/gộp + bảng theo lớp + confusion matrix."""
    o = result["overall"]
    print(f"\n{'=' * 100}")
    print(f" {title}")
    print(
        f" IoU >= {o['iou_threshold']} | IoA tách/gộp >= {o['ioa_threshold']} | "
        f"IoU background < {o['bg_iou_threshold']} | phủ tối thiểu khi tách >= {o['min_split_coverage']}"
    )
    print(f" GT={o['n_gt']:,} box | Pred={o['n_pred']:,} box | Pred/GT={o['pred_gt_ratio']:.3f}")
    print(f"{'=' * 100}")

    print("\n[Giai đoạn 1 – Detection (bỏ qua nhãn, khớp 1-1)]")
    print(
        f"  Det-TP={o['det_tp']:,}  Det-FP={o['det_fp']:,}  Det-FN={o['det_fn']:,}  |  "
        f"Det-P={_pct(o['det_precision'])}  Det-R={_pct(o['det_recall'])}  Det-F1={o['det_f1']:.4f}"
    )
    print(
        f"  IoU các cặp khớp: mean={o['iou_mean']:.3f}  median={o['iou_median']:.3f}  "
        f"P10={o['iou_p10']:.3f}  ≥0.75: {_pct(o['iou_ge_075'])}  ≥0.95: {_pct(o['iou_ge_095'])}"
    )

    print("\n[Giai đoạn 2 – Classification (trên các box Det-TP)]")
    print(
        f"  Gán nhãn đúng={o['cls_correct']:,}  Gán nhãn sai={o['cls_wrong']:,}  "
        f"Cls-Accuracy={_pct(o['cls_accuracy'])}"
    )

    print("\n[End-to-end (đúng vị trí và đúng nhãn)]")
    print(
        f"  E2E-P={_pct(o['e2e_precision'])}  E2E-R={_pct(o['e2e_recall'])}  E2E-F1={o['e2e_f1']:.4f}  "
        f"(E2E-R = Det-R × Cls-Acc = {o['det_recall']:.4f} × {o['cls_accuracy']:.4f})"
    )

    s, m, c = o["split"], o["merge"], o["complex"]
    print("\n[Tách / Gộp]")
    print(
        f"  Tách : {s['gt']:,} GT bị tách thành {s['fragments']:,} mảnh "
        f"(TB {s['mean_fragments']:.2f} mảnh/GT) | Split rate={_pct(s['rate'])}"
    )
    print(
        f"  Gộp  : {m['boxes']:,} box gộp chứa {m['gt']:,} GT "
        f"(TB {m['mean_gt_per_box']:.2f} GT/box) | Merge rate={_pct(m['rate'])} | "
        f"cùng lớp={m['same_class']:,}  khác lớp={m['cross_class']:,}"
    )
    print(f"  Phức tạp (m GT – n pred): {c['groups']:,} nhóm, {c['gt']:,} GT, {c['pred']:,} pred")
    if result["merge_combos"]:
        print("  Các tổ hợp lớp bị gộp nhiều nhất:")
        for combo, count in result["merge_combos"].most_common(8):
            print(f"    {' + '.join(combo):<45}{count:>7,}")

    print("\n[Bảng A – Số phận GT theo lớp (mỗi hàng cộng lại = #GT)]")
    header = f"{'Lớp GT':<18}{'#GT':>8}" + "".join(f"{GT_FATE_LABELS[f]:>10}" for f in GT_FATES)
    print(header + f"{'Mảnh':>8}{'Det-R':>9}{'Cls-Acc':>9}{'E2E-R':>9}")
    rows = sorted(result["per_gt_class"].values(), key=lambda r: -r["gt"])
    for r in rows:
        print(
            f"{r['name']:<18}{r['gt']:>8,}" + "".join(f"{r[f]:>10,}" for f in GT_FATES)
            + f"{r['split_fragments']:>8,}{r['det_recall']:>9.3f}{r['cls_accuracy']:>9.3f}{r['e2e_recall']:>9.3f}"
        )
    gf = o["gt_fates"]
    print(
        f"{'TỔNG':<18}{o['n_gt']:>8,}" + "".join(f"{gf[f]:>10,}" for f in GT_FATES)
        + f"{s['fragments']:>8,}{o['det_recall']:>9.3f}{o['cls_accuracy']:>9.3f}{o['e2e_recall']:>9.3f}"
    )
    print("  (Mảnh = tổng số pred mà các GT bị tách của lớp đó bị chia thành)")

    print("\n[Bảng B – Nguồn gốc Pred theo lớp dự đoán (mỗi hàng cộng lại = #Pred)]")
    header = f"{'Lớp Pred':<18}{'#Pred':>8}" + "".join(f"{PRED_FATE_LABELS[f]:>10}" for f in PRED_FATES)
    print(header + f"{'GTgộp':>8}{'Det-P':>9}{'E2E-P':>9}")
    rows = sorted(result["per_pred_class"].values(), key=lambda r: -r["pred"])
    for r in rows:
        print(
            f"{r['name']:<18}{r['pred']:>8,}" + "".join(f"{r[f]:>10,}" for f in PRED_FATES)
            + f"{r['merged_gt']:>8,}{r['det_precision']:>9.3f}{r['e2e_precision']:>9.3f}"
        )
    pf = o["pred_fates"]
    print(
        f"{'TỔNG':<18}{o['n_pred']:>8,}" + "".join(f"{pf[f]:>10,}" for f in PRED_FATES)
        + f"{m['gt']:>8,}{o['det_precision']:>9.3f}{o['e2e_precision']:>9.3f}"
    )
    print("  (GTgộp = số GT bị nuốt vào các box gộp mang nhãn này)")

    print_confusion(result, category_names)

    pages = o["pages"]
    print("\n[Theo trang]")
    print(
        f"  Trang hoàn hảo (mọi GT và pred đều 'Đúng'): {pages['perfect']:,}/{pages['images']:,} "
        f"= {_pct(pages['perfect_rate'])}"
    )
    print(
        f"  TB mỗi trang: {pages['mean_gt_errors']:.2f} GT không đúng, "
        f"{pages['mean_pred_errors']:.2f} pred không đúng"
    )


def print_confusion(result: dict, category_names: dict) -> None:
    confusion = result["confusion"]
    cats = sorted(category_names)
    extra = sorted(
        {p for (_, p) in confusion if p != BACKGROUND and p not in category_names},
        key=str,
    )
    cols = cats + extra + [BACKGROUND]
    rows = cats + [BACKGROUND]
    names = {**category_names, BACKGROUND: "background", None: "(outside taxonomy)"}
    label = {c: (str(i) if c != BACKGROUND else "BG") for i, c in enumerate(cats + extra)}
    label[BACKGROUND] = "BG"

    print("\n[Bảng C – Confusion matrix (hàng = nhãn GT, cột = nhãn pred; chỉ trên cặp Det-TP;")
    print("  cột BG = GT bị bỏ sót, hàng BG = pred thừa; các fate khác không nằm trong bảng)]")
    print("  " + "  ".join(f"{label[c]}={names.get(c, str(c))}" for c in cats + extra))
    width = max(7, max((len(f"{v:,}") for v in confusion.values()), default=0) + 2)
    corner = "GT \\ Pred"
    print(f"{corner:<20}" + "".join(f"{label[c]:>{width}}" for c in cols))
    for r in rows:
        name = f"{label[r]} {names.get(r, str(r))}"[:19]
        print(f"{name:<20}" + "".join(
            f"{confusion.get((r, c), 0):>{width},}" if (r, c) != (BACKGROUND, BACKGROUND) else f"{'-':>{width}}"
            for c in cols
        ))

    wrong = [((g, p), n) for (g, p), n in confusion.items()
             if BACKGROUND not in (g, p) and g != p]
    if wrong:
        print("  Các cặp nhầm nhãn nhiều nhất (GT → Pred):")
        for (g, p), n in sorted(wrong, key=lambda item: -item[1])[:10]:
            print(f"    {names.get(g, str(g)):<18} → {names.get(p, str(p)):<18}{n:>7,}")


def _fmt_px(value: float) -> str:
    if value == float("inf"):
        return "rất lớn"
    return f"{value:.1f}px" if value > 0 else "<0.5px"


def print_localization_reference(result: dict, ceiling: dict) -> None:
    """In thang đo tham chiếu: kết quả ở ngưỡng IoU này là cao hay thấp."""
    o = result["overall"]
    pixels = ceiling["pixels"]
    show = [k for k in (1, 2, 3, 5) if k in pixels]
    tau = o["iou_threshold"]

    print(f"\n[Thang đo tham chiếu độ khít box ở IoU ≥ {tau}]")
    print("  Dung sai   = số px mỗi cạnh được lệch mà IoU vẫn ≥ ngưỡng (trung vị trên các GT của lớp)")
    print("  Lệch biên  = sai lệch trung bình 4 cạnh giữa pred và GT (px), trên các box đã tìm thấy (IoU ≥ 0.5)")
    print(f"  Đạt        = trong các box đã tìm thấy, tỉ lệ đạt IoU ≥ {tau} (chỉ đo độ khít, không tính gộp/tách/sót)")
    print(f"  Trần ±k    = tỉ lệ đạt IoU ≥ {tau} nếu lấy chính GT rồi lệch ngẫu nhiên mỗi cạnh trong ±k px")
    print("  Tương đương = model khít ngang với GT bị lệch ±k px mỗi cạnh (càng nhỏ càng tốt)")
    print(
        f"{'Lớp GT':<18}{'#Cặp':>8}{'Dung sai':>10}{'Lệch biên':>11}{'P90':>8}{'≤1px':>8}{'≤2px':>8}{'Đạt':>8}"
        + "".join(f"{'Trần±' + str(k):>9}" for k in show) + f"{'Tương đương':>13}"
    )

    def row(name, stats, curve):
        equivalent = equivalent_jitter_px(stats["loc_pass_rate"], pixels, curve)
        print(
            f"{name:<18}{stats['loc_pairs']:>8,}{stats['tolerance_median']:>9.2f}px"
            f"{stats['edge_err_median']:>9.2f}px{stats['edge_err_p90']:>6.1f}px"
            f"{stats['edge_err_le_1px'] * 100:>7.1f}%{stats['edge_err_le_2px'] * 100:>7.1f}%"
            f"{stats['loc_pass_rate'] * 100:>7.1f}%"
            + "".join(f"{curve[pixels.index(k)] * 100:>8.1f}%" for k in show)
            + f"{_fmt_px(equivalent):>13}"
        )

    for cat_id, stats in sorted(result["per_gt_class"].items(), key=lambda item: -item[1]["gt"]):
        if cat_id in ceiling["per_class"]:
            row(stats["name"], stats, ceiling["per_class"][cat_id])
    row("TỔNG", o, ceiling["overall"])


def print_error_analysis_multi_iou(results: list[dict]) -> None:
    print("\n[Tóm tắt theo ngưỡng IoU]")
    print(
        f"{'IoU':<7}{'Det-P':>9}{'Det-R':>9}{'Det-F1':>9}{'Cls-Acc':>9}{'E2E-P':>9}{'E2E-R':>9}"
        f"{'E2E-F1':>9}{'Lệch':>8}{'BịTách':>8}{'BịGộp':>8}{'BỏSót':>8}{'Thừa':>8}{'Hoàn hảo':>10}"
    )
    for r in results:
        o = r["overall"]
        print(
            f"≥{o['iou_threshold']:<6.2f}{o['det_precision']:>9.3f}{o['det_recall']:>9.3f}{o['det_f1']:>9.3f}"
            f"{o['cls_accuracy']:>9.3f}{o['e2e_precision']:>9.3f}{o['e2e_recall']:>9.3f}{o['e2e_f1']:>9.3f}"
            f"{o['gt_fates']['loc'] + o['gt_fates']['both']:>8,}{o['gt_fates']['split']:>8,}"
            f"{o['gt_fates']['merged']:>8,}{o['gt_fates']['missed']:>8,}{o['pred_fates']['background']:>8,}"
            f"{o['pages']['perfect_rate'] * 100:>9.2f}%"
        )


def run_coco_map(gt_coco: dict, predictions_by_image: dict, evaluated_image_ids: set) -> None:
    """Tính và in bảng mAP kiểu COCO (maxDets=100) trên tập ảnh evaluated_image_ids."""
    all_detections = [
        d for img_id in evaluated_image_ids for d in predictions_by_image.get(img_id, [])
    ]
    if not all_detections:
        print("\n[mAP] Khong co du doan nao de tinh mAP.")
        return

    no_score = sum(d.get("score") is None for d in all_detections)
    if no_score:
        print(
            f"\n[mAP] Bo qua: {no_score:,}/{len(all_detections):,} box khong co truong 'score' "
            "(chay lai inference de luu confidence/score)."
        )
        return

    cat_ids = {c["id"] for c in gt_coco["categories"]}
    gt_subset = {
        "images": [img for img in gt_coco["images"] if img["id"] in evaluated_image_ids],
        "annotations": [
            ann for ann in gt_coco["annotations"] if ann["image_id"] in evaluated_image_ids
        ],
        "categories": gt_coco["categories"],
    }

    coco_predictions = [
        {
            "image_id": img_id,
            "category_id": d["category_id"],
            "bbox": d["bbox"],
            "score": float(d["score"]),
        }
        for img_id in evaluated_image_ids
        for d in predictions_by_image.get(img_id, [])
        if d.get("category_id") in cat_ids
    ]

    try:
        result = coco_map(gt_subset, coco_predictions)
    except Exception as exc:
        print(f"\n[mAP] Khong the tinh COCO mAP: {exc}")
        return

    def fmt(x: float) -> str:
        return f"{x:.3f}" if x >= 0 else "    -  "

    o = result["overall"]
    print("\n=== COCO-style mAP (maxDets=100) ===")
    print(
        f"mAP@[.5:.95]={fmt(o['map'])}  mAP@.5={fmt(o['map50'])}  mAP@.75={fmt(o['map75'])}"
    )
    print(
        f"mAP small/medium/large={fmt(o['map_small'])}/{fmt(o['map_medium'])}/{fmt(o['map_large'])}  "
        f"AR@1/10/100={fmt(o['ar1'])}/{fmt(o['ar10'])}/{fmt(o['ar100'])}  "
        f"AR small/medium/large={fmt(o['ar_small'])}/{fmt(o['ar_medium'])}/{fmt(o['ar_large'])}"
    )
    print(f"{'Category':<18}{'AP':>9}{'AP50':>9}")
    for c in sorted(
        result["per_category"].values(), key=lambda item: item["ap"], reverse=True
    ):
        print(f"{c['name']:<18}{fmt(c['ap']):>9}{fmt(c['ap50']):>9}")


def print_runtime(method: str, dataset: str, page_seconds: dict, evaluated_image_ids: set) -> None:
    """In thoi gian xu ly moi trang (tren tap anh duoc danh gia) va peak RAM/VRAM."""
    seconds = [page_seconds[i] for i in evaluated_image_ids if i in page_seconds]
    runs = load_runtime_runs(method, dataset)
    print("\n=== Runtime ===")
    if seconds:
        stats = page_time_stats(seconds)
        missing = len(evaluated_image_ids) - len(seconds)
        print(
            f"Time/page ({len(seconds)} pages): mean={stats['mean_s_per_page']:.3f}s  "
            f"median={stats['median_s_per_page']:.3f}s  p95={stats['p95_s_per_page']:.3f}s"
            + (f"  ({missing} pages without timing)" if missing else "")
        )
    else:
        print("Time/page: n/a (prediction files have no 'inference_time_s'; rerun inference).")

    if not runs:
        print("Peak RAM/VRAM: n/a (no runtime summary in outputs/runtime/).")
        return

    def peak(key: str) -> str:
        values = [run[key] for run in runs if run.get(key) is not None]
        return f"{max(values):,.0f} MB" if values else "n/a"

    devices = sorted({run.get("device", "?") for run in runs})
    loads = [run["model_load_s"] for run in runs if run.get("model_load_s") is not None]
    print(
        f"Peak RAM={peak('peak_ram_mb')}  Peak VRAM={peak('peak_vram_mb')}  "
        f"model load={f'{statistics.median(loads):.1f}s' if loads else 'n/a'}  "
        f"device={', '.join(devices)}  ({len(runs)} run(s))"
    )
    if len(devices) > 1:
        print("[warn] Inference ran on several devices; time/page is not comparable.")


def plot_evaluation_results(
    strict: dict,
    category_names: dict,
    loose: dict | None = None,
    title: str = "Evaluation Metrics",
    output_path: Path | str | None = None,
) -> plt.Figure:
    """Vẽ biểu đồ so sánh Precision, Recall, F1 theo từng class và so sánh Strict vs Loose."""
    sorted_items = sorted(
        category_names.items(),
        key=lambda item: -strict["per_category"].get(item[0], {}).get("support", 0)
    )
    labels = [name for _, name in sorted_items]
    y = np.arange(len(labels))
    height = 0.25

    precisions = [strict["per_category"].get(cid, {}).get("precision", 0.0) for cid, _ in sorted_items]
    recalls = [strict["per_category"].get(cid, {}).get("recall", 0.0) for cid, _ in sorted_items]
    f1s = [strict["per_category"].get(cid, {}).get("f1", 0.0) for cid, _ in sorted_items]

    if loose is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    else:
        fig, ax1 = plt.subplots(figsize=(9, 6))
        ax2 = None

    # Subplot 1: Strict Precision, Recall, F1
    ax1.barh(y - height, precisions, height, label="Precision", color="#3b82f6")
    ax1.barh(y, recalls, height, label="Recall", color="#f97316")
    ax1.barh(y + height, f1s, height, label="F1-Score", color="#10b981")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 1.05)
    ax1.set_xlabel("Score", fontsize=11)
    ax1.set_title("Strict Metrics (Location & Class)", fontsize=12, fontweight="bold")
    ax1.grid(axis="x", linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    ov = strict["overall"]
    ax1.text(
        0.02, 0.02,
        f"Micro: P={ov['precision']:.3f} | R={ov['recall']:.3f} | F1={ov['f1']:.3f}",
        transform=ax1.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#ccc")
    )

    # Subplot 2: Strict F1 vs Loose F1 comparison
    if ax2 is not None:
        loose_f1s = [loose["per_category"].get(cid, {}).get("f1", 0.0) for cid, _ in sorted_items]
        h2 = 0.35
        ax2.barh(y - h2 / 2, f1s, h2, label="Strict F1 (Class & Location)", color="#10b981")
        ax2.barh(y + h2 / 2, loose_f1s, h2, label="Loose F1 (Location only)", color="#8b5cf6")
        ax2.set_xlim(0, 1.05)
        ax2.set_xlabel("F1-Score", fontsize=11)
        ax2.set_title("F1 Comparison: Strict vs Loose (Localization)", fontsize=12, fontweight="bold")
        ax2.grid(axis="x", linestyle="--", alpha=0.5)
        ax2.legend(loc="lower right")

        lov = loose["overall"]
        ax2.text(
            0.02, 0.02,
            f"Loose Micro: P={lov['precision']:.3f} | R={lov['recall']:.3f} | F1={lov['f1']:.3f}",
            transform=ax2.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#ccc")
        )

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=200, bbox_inches="tight")
        print(f"[eval] Evaluation chart saved to: {out_p}")

    return fig


def plot_box_evaluation_results(
    multi_iou_results: list[dict],
    single_res: dict,
    title: str = "Bounding Box Evaluation",
    output_path: Path | str | None = None,
) -> plt.Figure:
    """Ve bieu do phan tich chat luong Bounding Box thuan tuy theo cac nguong IoU."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    # Subplot 1: IoU vs Metrics Curve
    ious = [r["iou_threshold"] for r in multi_iou_results]
    recalls = [r["recall"] for r in multi_iou_results]
    precisions = [r["precision"] for r in multi_iou_results]
    f1s = [r["f1"] for r in multi_iou_results]
    mean_ious = [r["mean_iou_matched"] for r in multi_iou_results]

    ax1.plot(ious, recalls, marker="o", color="#f97316", linewidth=2.5, label="Recall (Tỉ lệ nhận diện)")
    ax1.plot(ious, precisions, marker="s", color="#3b82f6", linewidth=2.5, label="Precision (Độ chính xác)")
    ax1.plot(ious, f1s, marker="^", color="#10b981", linewidth=2.5, label="F1-Score")
    ax1.plot(ious, mean_ious, marker="d", color="#8b5cf6", linewidth=1.8, linestyle="--", label="Mean IoU (độ khít)")

    idx_50 = 0
    ax1.annotate(
        f"Recall@0.5: {recalls[idx_50]*100:.1f}%\nPrec@0.5: {precisions[idx_50]*100:.1f}%",
        xy=(ious[idx_50], recalls[idx_50]),
        xytext=(ious[idx_50] + 0.05, recalls[idx_50] - 0.12),
        arrowprops=dict(arrowstyle="->", color="#f97316", lw=1.5),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff7ed", edgecolor="#f97316", alpha=0.9),
    )

    ax1.set_xlabel("IoU Threshold", fontsize=11)
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_xlim(0.48, 0.97)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_title("Đường cong BBox Metrics theo ngưỡng IoU", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left", fontsize=10)

    # Subplot 2: Phan bo hop TP, FP, FN tai nguong kiem thu
    tp = single_res["tp"]
    fp = single_res["fp"]
    fn = single_res["fn"]
    bar_labels = ["True Positive\n(Khớp đúng)", "False Positive\n(Dư / sai lệch)", "False Negative\n(Bị bỏ sót)"]
    bar_vals = [tp, fp, fn]
    bar_colors = ["#10b981", "#ef4444", "#f59e0b"]

    bars = ax2.bar(bar_labels, bar_vals, color=bar_colors, width=0.55, edgecolor="#333", linewidth=0.8)
    for b in bars:
        h = b.get_height()
        ax2.annotate(f"{h:,}", xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                     fontweight="bold", fontsize=11)

    ax2.set_ylabel("Số lượng Bounding Box", fontsize=11)
    ax2.set_title(f"Phân bố Hộp tại ngưỡng IoU >= {single_res['iou_threshold']:.2f}", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    summary_text = (
        f"Tổng GT: {single_res['total_gt']:,} hộp\n"
        f"Tổng Pred: {single_res['total_pred']:,} hộp\n"
        f"Recall: {single_res['recall']*100:.2f}%\n"
        f"Precision: {single_res['precision']*100:.2f}%\n"
        f"F1: {single_res['f1']:.4f}\n"
        f"Mean IoU: {single_res['mean_iou_matched']:.4f}"
    )
    ax2.text(
        0.97, 0.95, summary_text,
        transform=ax2.transAxes,
        fontsize=10,
        va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.95)
    )

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=200, bbox_inches="tight")
        print(f"[eval] Box evaluation chart saved to: {out_p}")

    return fig


def print_box_report(
    single_res: dict,
    multi_res: list[dict],
    method: str,
    dataset: str,
    split: str,
    num_images: int,
) -> None:
    print(f"\n{'=' * 88}")
    print(f" ĐÁNH GIÁ CHẤT LƯỢNG BOUNDING BOX (CLASS-AGNOSTIC / LOCALIZATION ONLY)")
    print(f" Method: {method} | Dataset: {dataset}/{split} | Số lượng ảnh đánh giá: {num_images:,}")
    print(f" Tổng Ground Truth: {single_res['total_gt']:,} hộp | Tổng Dự đoán: {single_res['total_pred']:,} hộp")
    print(f"{'=' * 88}")

    print(f"\n[Kết quả tại ngưỡng IoU >= {single_res['iou_threshold']:.2f}]:")
    print(f"  • Tỉ lệ BBox được nhận diện (Recall) : {single_res['recall']*100:.2f}%  ({single_res['tp']:,} / {single_res['total_gt']:,} hộp)")
    print(f"  • Độ chính xác BBox (Precision)       : {single_res['precision']*100:.2f}%  ({single_res['tp']:,} / {single_res['total_pred']:,} hộp)")
    print(f"  • F1-Score                           : {single_res['f1']:.4f}")
    print(f"  • Độ khít viền trung bình (Mean IoU) : {single_res['mean_iou_matched']:.4f}")
    print(f"  • Thống kê chi tiết                  : TP={single_res['tp']:,} | FP={single_res['fp']:,} | FN={single_res['fn']:,}")

    print(f"\n--- Bảng độ đo chi tiết theo các ngưỡng IoU ---")
    print(f"{'IoU Thresh':<12}{'GT':>8}{'Pred':>8}{'TP':>8}{'FP':>8}{'FN':>8}{'Recall':>12}{'Prec':>12}{'F1':>10}{'Mean IoU':>12}")
    for r in multi_res:
        print(f">={r['iou_threshold']:<10.2f}{r['total_gt']:>8}{r['total_pred']:>8}{r['tp']:>8}{r['fp']:>8}{r['fn']:>8}{r['recall']*100:>11.2f}%{r['precision']*100:>11.2f}%{r['f1']:>10.4f}{r['mean_iou_matched']:>12.4f}")


def run_coco_box_map(raw_coco: dict, predictions: dict, evaluated_image_ids: set) -> None:
    subset_coco = {
        **raw_coco,
        "images": [img for img in raw_coco["images"] if img["id"] in evaluated_image_ids],
        "annotations": [
            ann for ann in raw_coco["annotations"] if ann["image_id"] in evaluated_image_ids
        ],
    }
    pred_list = [
        {**detection, "image_id": image_id}
        for image_id, detections in predictions.items()
        if image_id in evaluated_image_ids
        for detection in detections
    ]
    try:
        res = coco_box_map(subset_coco, pred_list)
        ov = res["overall"]
        print(f"\n=== Class-Agnostic Box mAP (COCO-style) ===")
        print(f"Box mAP@[.5:.95]={ov['map']:.3f}  Box mAP@.5={ov['map50']:.3f}  Box mAP@.75={ov['map75']:.3f}")
        print(f"Box AR@100={ov['ar100']:.3f}  AR small/medium/large={ov['ar_small']:.3f}/{ov['ar_medium']:.3f}/{ov['ar_large']:.3f}")
    except Exception as e:
        print(f"[warn] Không thể tính Class-Agnostic COCO mAP: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="docling")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Ten dataset (doclaynet / doclaynet_val / doclaynet_test / docbank ...). "
             "Bo trong de suy ra tu --split.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Tap danh gia: train / valid (val, validation) / test. Voi dataset DocLayNet, "
             "ten dataset va file annotation duoc suy ra tu day; voi dataset khac la ten file annotation.",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--loose",
        action="store_true",
        help="Also report localization-only scores, ignoring class labels.",
    )
    parser.add_argument(
        "--box-only",
        action="store_true",
        help="Chỉ đánh giá chất lượng Bounding Box (Class-agnostic), bỏ qua hoàn toàn nhãn class.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Cho phép đánh giá trên tập ảnh hiện có dự đoán (hữu ích khi test với --limit).",
    )
    parser.add_argument(
        "--no-map",
        action="store_true",
        help="Không tính mAP kiểu COCO.",
    )
    parser.add_argument(
        "--no-error-analysis",
        action="store_true",
        help="Không in phân tích lỗi 2 giai đoạn (detection/classification, tách/gộp, bảng theo lớp).",
    )
    parser.add_argument(
        "--error-iou-threshold",
        type=float,
        default=0.95,
        help="Ngưỡng IoU để một box được tính là 'Đúng' trong phân tích lỗi (độc lập với --iou-threshold "
             "của báo cáo Strict). IoU giữa --bg-iou-threshold và ngưỡng này được tính là 'Lệch'.",
    )
    parser.add_argument(
        "--ioa-threshold",
        type=float,
        default=0.8,
        help="Ngưỡng IoA (intersection / diện tích box nhỏ) để coi một box nằm trong box kia khi xét tách/gộp.",
    )
    parser.add_argument(
        "--bg-iou-threshold",
        type=float,
        default=0.1,
        help="Pred có IoU nhỏ hơn ngưỡng này với mọi GT được tính là background (thừa).",
    )
    parser.add_argument(
        "--min-split-coverage",
        type=float,
        default=0.5,
        help="Các mảnh phải phủ ít nhất tỉ lệ này diện tích GT mới tính là GT bị tách.",
    )
    parser.add_argument(
        "--error-iou-thresholds",
        default="0.5,0.75,0.95",
        help="Các ngưỡng IoU cho bảng tóm tắt phân tích lỗi, phân tách bằng dấu phẩy.",
    )
    parser.add_argument(
        "--plot",
        nargs="?",
        const="default",
        default=None,
        help="Lưu biểu đồ đánh giá ra file ảnh (mặc định: outputs/reports/<method>_<dataset>/metrics_chart.png).",
    )
    args = parser.parse_args()

    # Chon tap train / valid / test mot cach tuong minh va chan viec tron lan giua cac tap.
    if args.dataset is None or split_of_dataset(args.dataset):
        try:
            info = resolve(args.split)
        except ValueError as exc:
            raise SystemExit(str(exc))
        if args.dataset is not None and args.dataset != info.dataset:
            raise SystemExit(
                f"--dataset {args.dataset} thuoc tap '{split_of_dataset(args.dataset)}' nhung --split la '{info.name}'. "
                f"Dung --split {split_of_dataset(args.dataset)} (hoac bo --dataset) de khong tron lan cac tap."
            )
        args.dataset, args.split = info.dataset, info.annotation
    split_name = split_of_dataset(args.dataset)
    if split_name:
        banner = f"TAP DANH GIA: {split_name.upper()} ({args.dataset}) - {SPLITS[split_name].purpose}"
        print(banner)
        if split_name == "train":
            print("[warn] Dang danh gia tren tap TRAIN: model co the da hoc du lieu nay, khong dung de bao cao ket qua.")
        elif split_name == "test":
            print("[note] Tap TEST chi de bao cao cuoi cung; chon nguong / cach map nhan tren tap valid.")

    annotation_path = (
        ROOT_DIR
        / "datasets"
        / args.dataset
        / "processed"
        / "annotations"
        / f"{args.split}.json"
    )
    predictions_dir = ROOT_DIR / "outputs" / "predictions" / args.method / args.dataset

    raw_coco, ground_truth, category_names = load_ground_truth(annotation_path)
    predictions, page_seconds = load_predictions(predictions_dir)
    missing_ids = set(ground_truth) - set(predictions)
    if missing_ids:
        if args.allow_incomplete:
            ground_truth = {k: v for k, v in ground_truth.items() if k in predictions}
        else:
            examples = ", ".join(map(str, sorted(missing_ids, key=str)[:10]))
            raise SystemExit(
                f"Incomplete predictions: {len(missing_ids)} of {len(ground_truth)} "
                f"ground-truth images are missing (e.g. {examples}). Run inference first (hoặc dùng --allow-incomplete)."
            )

    if not ground_truth:
        raise SystemExit(
            f"Chua co prediction nao cua {args.method} cho tap {args.dataset}. "
            f"Chay inference truoc: python methods/{args.method}/src/run_inference.py --split {split_name or args.split}"
        )

    # Chi danh gia du doan tren dung tap anh cua GT (cung split); du doan tren
    # anh ngoai tap nay khong thuoc phep do nen bo ra, kem canh bao.
    extra_ids = set(predictions) - set(ground_truth)
    if extra_ids:
        examples = ", ".join(map(str, sorted(extra_ids, key=str)[:10]))
        print(
            f"[warn] {len(extra_ids)} prediction file(s) have image_id not in "
            f"{annotation_path.name} and are excluded (e.g. {examples})."
        )
        predictions = {k: v for k, v in predictions.items() if k in ground_truth}
    detections = [d for dets in predictions.values() for d in dets]

    if args.box_only:
        evaluated_ids = set(ground_truth.keys())
        single_box_eval = evaluate_boxes(ground_truth, predictions, iou_threshold=args.iou_threshold)
        multi_box_eval = evaluate_boxes_multi_iou(ground_truth, predictions)
        print_box_report(single_box_eval, multi_box_eval, args.method, args.dataset, args.split, len(ground_truth))

        if args.plot:
            if args.plot == "default":
                plot_path = ROOT_DIR / "outputs" / "reports" / f"{args.method}_{args.dataset}" / "metrics_chart.png"
            else:
                plot_path = Path(args.plot)
            plot_box_evaluation_results(
                multi_box_eval,
                single_box_eval,
                title=f"{args.method.capitalize()} Bounding Box Evaluation ({args.dataset}/{args.split})",
                output_path=plot_path,
            )

        if not args.no_map:
            run_coco_box_map(raw_coco, predictions, evaluated_ids)

        print_runtime(args.method, args.dataset, page_seconds, evaluated_ids)
        return

    pred_counts = Counter(detection["category_id"] for detection in detections)
    strict = evaluate(
        ground_truth,
        predictions,
        category_names,
        args.iou_threshold,
        require_same_category=True,
    )
    strict["pred_counts"] = pred_counts

    ground_truth_count = sum(map(len, ground_truth.values()))
    print(
        f"Evaluation: {args.method} on {args.dataset}/{args.split} | "
        f"{len(ground_truth)} images | GT={ground_truth_count:,} boxes | "
        f"Predictions={len(detections):,} boxes | IoU >= {args.iou_threshold}"
    )
    print_report(strict, category_names, "Strict (location and class)")

    unsupported_labels = Counter(
        detection["category_name"]
        for detection in detections
        if detection.get("category_id") not in category_names
    )
    if unsupported_labels:
        print(f"\n{args.method.capitalize()} labels outside the {args.dataset.capitalize()} taxonomy (counted as strict FP):")
        for label, count in sorted(unsupported_labels.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {label}: {count}")

    loose = None
    if args.loose:
        loose = evaluate(
            ground_truth,
            predictions,
            category_names,
            args.iou_threshold,
            require_same_category=False,
        )
        loose["pred_counts"] = pred_counts
        print_report(loose, category_names, "Localization-only (class ignored)")

    if not args.no_error_analysis:
        analysis_kwargs = {
            "ioa_threshold": args.ioa_threshold,
            "bg_iou_threshold": args.bg_iou_threshold,
            "min_split_coverage": args.min_split_coverage,
        }
        analysis = error_analysis(
            ground_truth, predictions, category_names, args.error_iou_threshold, **analysis_kwargs
        )
        print_error_analysis(
            analysis,
            category_names,
            "PHÂN TÍCH LỖI 2 GIAI ĐOẠN: DETECTION → CLASSIFICATION, TÁCH / GỘP BOX",
        )
        print_localization_reference(
            analysis, jitter_ceiling(ground_truth, args.error_iou_threshold)
        )
        thresholds = [float(t) for t in args.error_iou_thresholds.split(",") if t.strip()]
        if thresholds:
            print_error_analysis_multi_iou(
                error_analysis_multi_iou(
                    ground_truth, predictions, category_names, thresholds, **analysis_kwargs
                )
            )

    if args.plot:
        if args.plot == "default":
            plot_path = ROOT_DIR / "outputs" / "reports" / f"{args.method}_{args.dataset}" / "metrics_chart.png"
        else:
            plot_path = Path(args.plot)
        plot_evaluation_results(
            strict,
            category_names,
            loose=loose,
            title=f"{args.method.capitalize()} Layout Evaluation ({args.dataset}/{args.split}, IoU >= {args.iou_threshold})",
            output_path=plot_path,
        )

    if not args.no_map:
        run_coco_map(raw_coco, predictions, set(ground_truth.keys()))

    print_runtime(args.method, args.dataset, page_seconds, set(ground_truth.keys()))


if __name__ == "__main__":
    main()
