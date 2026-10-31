"""Tinh Precision/Recall/F1 cho bai toan Document Layout Analysis bang cach
so khop bounding box du doan voi ground truth theo IoU.

Dung chung cho moi method (docling, paddleocr_vl, ...): moi method chi can
xuat du doan dung dinh dang {"category_id", "bbox": [x, y, w, h]} la co the
danh gia bang module nay.
"""

from collections import Counter, defaultdict
import contextlib
import io

import numpy as np


def iou(box_a: list, box_b: list) -> float:
    """IoU giua 2 box dang [x, y, w, h]."""
    ax1, ay1, ax2, ay2 = box_a[0], box_a[1], box_a[0] + box_a[2], box_a[1] + box_a[3]
    bx1, by1, bx2, by2 = box_b[0], box_b[1], box_b[0] + box_b[2], box_b[1] + box_b[3]

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def match_image(gt_boxes: list, pred_boxes: list, iou_threshold: float = 0.5,
                 require_same_category: bool = True):
    """Ghep du doan voi ground truth cho MOT anh bang greedy-IoU.

    Tra ve (matches, unmatched_gt_idx, unmatched_pred_idx):
      - matches: list (gt_idx, pred_idx, iou)
      - unmatched_gt_idx / unmatched_pred_idx: index con lai khong duoc ghep
    """
    candidates = []
    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            if require_same_category and gt["category_id"] != pred["category_id"]:
                continue
            score = iou(gt["bbox"], pred["bbox"])
            if score >= iou_threshold:
                candidates.append((score, gi, pi))

    candidates.sort(key=lambda c: c[0], reverse=True)

    matched_gt, matched_pred = set(), set()
    matches = []
    for score, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        matches.append((gi, pi, score))

    unmatched_gt = [i for i in range(len(gt_boxes)) if i not in matched_gt]
    unmatched_pred = [i for i in range(len(pred_boxes)) if i not in matched_pred]
    return matches, unmatched_gt, unmatched_pred


def evaluate(gt_by_image: dict, pred_by_image: dict, category_names: dict,
             iou_threshold: float = 0.5, require_same_category: bool = True) -> dict:
    """gt_by_image / pred_by_image: {image_id: [{"category_id", "bbox"}, ...]}.

    Tra ve dict {"overall": {...}, "per_category": {category_id: {...}}}
    voi precision/recall/f1/support/tp/fp/fn.
    """
    per_cat_tp = defaultdict(int)
    per_cat_fp = defaultdict(int)
    per_cat_fn = defaultdict(int)
    iou_sum, iou_count = 0.0, 0

    # Duyet hop 2 tap ID: du doan tren anh khong co trong GT van bi tinh FP,
    # thay vi bi bo qua am tham. Noi goi can tu loc pred_by_image ve dung tap
    # anh can danh gia (vd. cung split) truoc khi goi ham nay.
    for image_id in set(gt_by_image) | set(pred_by_image):
        gt_boxes = gt_by_image.get(image_id, [])
        pred_boxes = pred_by_image.get(image_id, [])
        matches, unmatched_gt, unmatched_pred = match_image(
            gt_boxes, pred_boxes, iou_threshold, require_same_category)

        for gi, pi, score in matches:
            cat_id = gt_boxes[gi]["category_id"]
            per_cat_tp[cat_id] += 1
            iou_sum += score
            iou_count += 1
        for gi in unmatched_gt:
            per_cat_fn[gt_boxes[gi]["category_id"]] += 1
        for pi in unmatched_pred:
            per_cat_fp[pred_boxes[pi]["category_id"]] += 1

    all_cats = set(per_cat_tp) | set(per_cat_fp) | set(per_cat_fn)
    per_category = {}
    for cat_id in sorted(all_cats, key=lambda c: (c is None, c)):
        tp, fp, fn = per_cat_tp[cat_id], per_cat_fp[cat_id], per_cat_fn[cat_id]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_category[cat_id] = {
            "name": category_names.get(cat_id, str(cat_id)),
            "tp": tp, "fp": fp, "fn": fn, "support": tp + fn,
            "precision": precision, "recall": recall, "f1": f1,
        }

    total_tp = sum(per_cat_tp.values())
    total_fp = sum(per_cat_fp.values())
    total_fn = sum(per_cat_fn.values())
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "overall": {
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "precision": precision, "recall": recall, "f1": f1,
            "mean_iou_matched": iou_sum / iou_count if iou_count > 0 else 0.0,
        },
        "per_category": per_category,
    }


def coco_map(gt_coco: dict, predictions: list, iou_type: str = "bbox") -> dict:
    """Tinh mAP/AR kieu COCO bang pycocotools.

    gt_coco: dict COCO ({"images", "annotations", "categories"}), chi nen chua
        cac anh da co du doan.
    predictions: list {"image_id", "category_id", "bbox": [x, y, w, h], "score"}.
        Box co category_id khong thuoc gt_coco["categories"] phai loai truoc.

    Tra ve dict {"overall": {...}, "per_category": {category_id: {"name", "ap", "ap50"}}}.
    """

    gt_coco = dict(gt_coco)
    gt_coco["annotations"] = [
        {**ann, "iscrowd": ann.get("iscrowd", 0),
         "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3])}
        for ann in gt_coco["annotations"]
    ]

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        raise ImportError(
            "Cần cài đặt thư viện 'pycocotools' để tính mAP. Vui lòng cài đặt: pip install pycocotools"
        )

    # pycocotools in rat nhieu log, gom lai de khong lan voi bao cao.
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        coco_gt = COCO()
        coco_gt.dataset = gt_coco
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(predictions) if predictions else COCO()
        coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    keys = ["map", "map50", "map75", "map_small", "map_medium", "map_large",
            "ar1", "ar10", "ar100", "ar_small", "ar_medium", "ar_large"]
    overall = {k: float(v) for k, v in zip(keys, coco_eval.stats)}

    # precision: [T iou thresholds, R recall points, K categories, A areas, M maxDets]
    precision = coco_eval.eval["precision"]
    names = {c["id"]: c["name"] for c in gt_coco["categories"]}
    per_category = {}
    for k, cat_id in enumerate(coco_eval.params.catIds):
        p_all = precision[:, :, k, 0, -1]
        p_50 = precision[0, :, k, 0, -1]
        per_category[cat_id] = {
            "name": names.get(cat_id, str(cat_id)),
            "ap": float(np.mean(p_all[p_all > -1])) if (p_all > -1).any() else -1.0,
            "ap50": float(np.mean(p_50[p_50 > -1])) if (p_50 > -1).any() else -1.0,
        }
    return {"overall": overall, "per_category": per_category}


def evaluate_boxes(
    gt_by_image: dict,
    pred_by_image: dict,
    iou_threshold: float = 0.5,
) -> dict:
    """Danh gia chat luong Bounding Box thuan tuy (Class-agnostic / Localization-only),
    hoan toan bo qua nhan class.

    gt_by_image / pred_by_image: {image_id: [{"bbox": [x, y, w, h], ...}, ...] hoac [[x, y, w, h], ...]}.

    Tra ve dict:
        total_gt, total_pred, tp, fp, fn, precision, recall, f1, mean_iou_matched, iou_threshold
    """
    total_tp = 0
    total_gt = 0
    total_pred = 0
    iou_sum = 0.0
    iou_count = 0

    all_image_ids = set(gt_by_image) | set(pred_by_image)
    for image_id in all_image_ids:
        raw_gts = gt_by_image.get(image_id, [])
        raw_preds = pred_by_image.get(image_id, [])

        gt_boxes = [{"bbox": b["bbox"] if isinstance(b, dict) else b} for b in raw_gts]
        pred_boxes = [{"bbox": b["bbox"] if isinstance(b, dict) else b} for b in raw_preds]

        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)

        matches, _, _ = match_image(
            gt_boxes, pred_boxes, iou_threshold=iou_threshold, require_same_category=False
        )
        total_tp += len(matches)
        for _, _, score in matches:
            iou_sum += score
            iou_count += 1

    total_fp = total_pred - total_tp
    total_fn = total_gt - total_tp
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = iou_sum / iou_count if iou_count > 0 else 0.0

    return {
        "iou_threshold": iou_threshold,
        "total_gt": total_gt,
        "total_pred": total_pred,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou_matched": mean_iou,
    }


def evaluate_boxes_multi_iou(
    gt_by_image: dict,
    pred_by_image: dict,
    iou_thresholds: list[float] | None = None,
) -> list[dict]:
    """Danh gia Bounding Box tren danh sach nhieu nguong IoU (mac dinh: 0.50 den 0.95 buoc 0.05)."""
    if iou_thresholds is None:
        iou_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    return [
        evaluate_boxes(gt_by_image, pred_by_image, iou_threshold=thresh)
        for thresh in iou_thresholds
    ]


def coco_box_map(gt_coco: dict, predictions: list, iou_type: str = "bbox") -> dict:
    """Tinh COCO mAP thuan tuy cho Bounding Box (Class-agnostic Box mAP), bo qua class.

    Quy doi toan bo ground truth va predictions ve mot class duy nhat: ID=1 ('BBox').
    """
    dummy_gt_coco = dict(gt_coco)
    dummy_gt_coco["categories"] = [{"id": 1, "name": "BBox"}]
    dummy_gt_coco["annotations"] = [
        {**ann, "category_id": 1} for ann in dummy_gt_coco.get("annotations", [])
    ]
    dummy_predictions = [
        {**pred, "category_id": 1} for pred in predictions
    ]
    return coco_map(dummy_gt_coco, dummy_predictions, iou_type=iou_type)


# ---------------------------------------------------------------------------
# Phan tich loi 2 giai doan (Detection -> Classification) + tach/gop box
# ---------------------------------------------------------------------------
#
# Moi GT co dung MOT "so phan" (GT_FATES), moi pred co dung MOT "nguon goc"
# (PRED_FATES), nen tong moi hang cua bang theo lop luon bang #GT / #Pred.
#
# Thu tu xet tren moi anh:
#   1. Tach / gop (dua tren IoA = intersection / area cua mot box):
#      - pred p "nam trong" GT g neu inter/area_p >= ioa_threshold
#      - GT g "nam trong" pred p neu inter/area_g >= ioa_threshold
#      - GT g bi TACH neu >= 2 pred nam trong g, cac manh phu >= min_split_coverage
#        dien tich g, va khong co pred nao khop 1-1 voi g (IoU >= iou_threshold).
#      - pred p GOP neu >= 2 GT nam trong p (chi tinh GT khong co pred khac
#        khop 1-1 voi no, de box "bao ngoai" thua khong bi coi la gop).
#      Cac canh tach/gop duoc gom thanh nhom lien thong: 1 GT - k pred = split,
#      k GT - 1 pred = merge, m GT - n pred (m, n >= 2) = complex.
#   2. Khop 1-1 khong xet nhan (Hungarian, IoU >= iou_threshold) tren phan con
#      lai -> Det-TP. Cung nhan = correct, khac nhan = cls_wrong.
#   3. Pred con lai trung (IoU >= iou_threshold hoac nam trong) mot GT da duoc
#      giai thich o buoc 1-2 -> duplicate.
#   4. Ghep 1-1 GT/pred con lai co bg_iou_threshold <= IoU < iou_threshold
#      -> loc (cung nhan) hoac both (lech vi tri + sai nhan).
#   5. Pred con lai con cham GT (IoU >= bg_iou_threshold) -> duplicate,
#      khong cham GT nao -> background. GT con lai -> missed.

GT_FATES = ("correct", "cls_wrong", "loc", "both", "split", "merged", "complex", "missed")
PRED_FATES = ("correct", "cls_wrong", "loc", "both", "duplicate", "split_fragment",
              "merge_box", "complex", "background")
BACKGROUND = "background"


def _xyxy(boxes: list) -> np.ndarray:
    if not boxes:
        return np.zeros((0, 4), dtype=float)
    arr = np.asarray([b["bbox"] for b in boxes], dtype=float).reshape(-1, 4)
    return np.stack([arr[:, 0], arr[:, 1], arr[:, 0] + arr[:, 2], arr[:, 1] + arr[:, 3]], axis=1)


def overlap_matrices(gt_boxes: list, pred_boxes: list):
    """Tra ve (iou, ioa_gt, ioa_pred), moi ma tran shape (G, P).

    ioa_gt[g, p] = inter / area_g (GT nam trong pred),
    ioa_pred[g, p] = inter / area_p (pred nam trong GT).
    """
    g, p = _xyxy(gt_boxes), _xyxy(pred_boxes)
    inter_w = np.clip(np.minimum(g[:, None, 2], p[None, :, 2]) - np.maximum(g[:, None, 0], p[None, :, 0]), 0, None)
    inter_h = np.clip(np.minimum(g[:, None, 3], p[None, :, 3]) - np.maximum(g[:, None, 1], p[None, :, 1]), 0, None)
    inter = inter_w * inter_h
    area_g = np.clip(g[:, 2] - g[:, 0], 0, None) * np.clip(g[:, 3] - g[:, 1], 0, None)
    area_p = np.clip(p[:, 2] - p[:, 0], 0, None) * np.clip(p[:, 3] - p[:, 1], 0, None)
    union = area_g[:, None] + area_p[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou_mat = np.where(union > 0, inter / union, 0.0)
        ioa_gt = np.where(area_g[:, None] > 0, inter / area_g[:, None], 0.0)
        ioa_pred = np.where(area_p[None, :] > 0, inter / area_p[None, :], 0.0)
    return iou_mat, ioa_gt, ioa_pred


def _assign(score: np.ndarray, rows: list, cols: list, threshold: float) -> list:
    """Ghep 1-1 toi da tong IoU giua rows (GT) va cols (pred), chi giu cap >= threshold.

    Dung Hungarian (scipy) neu co, nguoc lai greedy theo IoU giam dan.
    Tra ve list (gi, pi, iou).
    """
    if not rows or not cols:
        return []
    sub = score[np.ix_(rows, cols)]
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        linear_sum_assignment = None

    pairs = []
    if linear_sum_assignment is not None:
        # Cap duoi nguong bi dua ve 0 de Hungarian khong "hy sinh" cap tot.
        weights = np.where(sub >= threshold, sub, 0.0)
        r_idx, c_idx = linear_sum_assignment(weights, maximize=True)
        pairs = [(rows[r], cols[c], float(sub[r, c])) for r, c in zip(r_idx, c_idx)
                 if sub[r, c] >= threshold]
    else:
        used_r, used_c = set(), set()
        for flat in np.argsort(-sub, axis=None):
            r, c = divmod(int(flat), sub.shape[1])
            if sub[r, c] < threshold:
                break
            if r in used_r or c in used_c:
                continue
            used_r.add(r)
            used_c.add(c)
            pairs.append((rows[r], cols[c], float(sub[r, c])))
    return pairs


def analyze_image(gt_boxes: list, pred_boxes: list, iou_threshold: float = 0.5,
                  ioa_threshold: float = 0.8, bg_iou_threshold: float = 0.1,
                  min_split_coverage: float = 0.5) -> dict:
    """Gan so phan cho tung GT va nguon goc cho tung pred tren MOT anh.

    Tra ve dict:
      gt_fates / pred_fates: list ten fate theo index box
      matches: list (gi, pi, iou) cua cac cap khop 1-1 (Det-TP)
      groups: list {"type": split|merge|complex, "gt": [...], "pred": [...]}
    """
    n_gt, n_pred = len(gt_boxes), len(pred_boxes)
    gt_fates = [None] * n_gt
    pred_fates = [None] * n_pred
    result = {"gt_fates": gt_fates, "pred_fates": pred_fates, "matches": [], "loc_pairs": [], "groups": []}
    if n_gt == 0 or n_pred == 0:
        gt_fates[:] = ["missed"] * n_gt
        pred_fates[:] = ["background"] * n_pred
        return result

    iou_mat, ioa_gt, ioa_pred = overlap_matrices(gt_boxes, pred_boxes)
    good = iou_mat >= iou_threshold
    good_count = good.sum(axis=1)

    # --- 1. Canh tach / gop ---
    edges = []
    # Moi pred chi duoc tinh la manh cua GT ma no nam trong nhieu nhat.
    host = np.argmax(ioa_pred, axis=0)
    for gi in range(n_gt):
        if good_count[gi]:
            continue
        frags = [pi for pi in range(n_pred) if host[pi] == gi and ioa_pred[gi, pi] >= ioa_threshold]
        if len(frags) >= 2 and min(1.0, float(ioa_gt[gi, frags].sum())) >= min_split_coverage:
            edges.extend((gi, pi) for pi in frags)
    for pi in range(n_pred):
        # GT da co pred KHAC khop tot thi box nay chi la box bao ngoai thua.
        inside = [gi for gi in range(n_gt)
                  if ioa_gt[gi, pi] >= ioa_threshold and good_count[gi] - good[gi, pi] == 0]
        if len(inside) >= 2:
            edges.extend((gi, pi) for gi in inside)

    if edges:
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for gi, pi in edges:
            parent[find(("g", gi))] = find(("p", pi))
        components = defaultdict(lambda: {"gt": set(), "pred": set()})
        for gi, pi in edges:
            comp = components[find(("g", gi))]
            comp["gt"].add(gi)
            comp["pred"].add(pi)
        for comp in components.values():
            gts, preds = sorted(comp["gt"]), sorted(comp["pred"])
            if len(gts) == 1:
                kind, gt_fate, pred_fate = "split", "split", "split_fragment"
            elif len(preds) == 1:
                kind, gt_fate, pred_fate = "merge", "merged", "merge_box"
            else:
                kind, gt_fate, pred_fate = "complex", "complex", "complex"
            for gi in gts:
                gt_fates[gi] = gt_fate
            for pi in preds:
                pred_fates[pi] = pred_fate
            result["groups"].append({"type": kind, "gt": gts, "pred": preds})

    # --- 2. Khop 1-1 khong xet nhan ---
    free_gt = [gi for gi in range(n_gt) if gt_fates[gi] is None]
    free_pred = [pi for pi in range(n_pred) if pred_fates[pi] is None]
    for gi, pi, score in _assign(iou_mat, free_gt, free_pred, iou_threshold):
        same = gt_boxes[gi]["category_id"] == pred_boxes[pi].get("category_id")
        gt_fates[gi] = pred_fates[pi] = "correct" if same else "cls_wrong"
        result["matches"].append((gi, pi, score))

    # --- 3. Trung lap voi GT da duoc giai thich ---
    explained = np.array([f is not None for f in gt_fates])
    if explained.any():
        for pi in range(n_pred):
            if pred_fates[pi] is None:
                hit = (iou_mat[:, pi] >= iou_threshold) | (ioa_pred[:, pi] >= ioa_threshold)
                if (hit & explained).any():
                    pred_fates[pi] = "duplicate"

    # --- 4. Lech vi tri ---
    free_gt = [gi for gi in range(n_gt) if gt_fates[gi] is None]
    free_pred = [pi for pi in range(n_pred) if pred_fates[pi] is None]
    for gi, pi, score in _assign(iou_mat, free_gt, free_pred, bg_iou_threshold):
        same = gt_boxes[gi]["category_id"] == pred_boxes[pi].get("category_id")
        gt_fates[gi] = pred_fates[pi] = "loc" if same else "both"
        result["loc_pairs"].append((gi, pi, score))

    # --- 5. Phan con lai ---
    for pi in range(n_pred):
        if pred_fates[pi] is None:
            pred_fates[pi] = "duplicate" if (iou_mat[:, pi] >= bg_iou_threshold).any() else "background"
    for gi in range(n_gt):
        if gt_fates[gi] is None:
            gt_fates[gi] = "missed"
    return result


def _prf(tp: int, n_pred: int, n_gt: int) -> tuple:
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def iou_tolerance_px(widths, heights, iou_threshold: float) -> np.ndarray:
    """So pixel moi canh co the lech (no deu ra ngoai) ma IoU van >= iou_threshold.

    Giai wh / ((w + 2d)(h + 2d)) = t theo d. Box cang nho thi dung sai cang nho:
    vd. box 30x10 px o IoU 0.95 chi cho phep lech ~0.12 px moi canh.
    """
    w = np.asarray(widths, dtype=float)
    h = np.asarray(heights, dtype=float)
    s = w + h
    return (-s + np.sqrt(s * s + 4 * w * h * (1 / iou_threshold - 1))) / 4


def _edge_errors(gt_boxes: list, pred_boxes: list, pairs: list) -> np.ndarray:
    """Sai lech trung binh |canh GT - canh pred| (px) cua 4 canh, cho moi cap (gi, pi, iou)."""
    if not pairs:
        return np.zeros(0)
    g = _xyxy([gt_boxes[gi] for gi, _, _ in pairs])
    p = _xyxy([pred_boxes[pi] for _, pi, _ in pairs])
    return np.abs(g - p).mean(axis=1)


def jitter_ceiling(gt_by_image: dict, iou_threshold: float,
                   pixels: tuple = (0.5, 1, 1.5, 2, 3, 4, 5, 7, 10), seed: int = 0) -> dict:
    """Tran tham chieu do nhieu gan nhan: dich ngau nhien moi canh cua GT trong
    [-k, k] px roi tinh ti le box con dat IoU >= iou_threshold voi GT goc.

    Ket qua cho biet mot "model hoan hao nhung box lech +-k px" dat bao nhieu o
    nguong nay -> dung de quy doi ket qua cua model ve "do lech tuong duong (px)".

    Tra ve {"pixels": [...], "overall": [ti le theo k], "per_class": {cat_id: [ti le theo k]}}.
    """
    boxes = [b for boxes in gt_by_image.values() for b in boxes]
    cats = np.asarray([b["category_id"] for b in boxes], dtype=object)
    g = _xyxy(boxes)
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-1.0, 1.0, size=g.shape)
    area_g = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])

    overall, per_class = [], defaultdict(list)
    for k in pixels:
        p = g + noise * k
        inter = (np.clip(np.minimum(g[:, 2], p[:, 2]) - np.maximum(g[:, 0], p[:, 0]), 0, None)
                 * np.clip(np.minimum(g[:, 3], p[:, 3]) - np.maximum(g[:, 1], p[:, 1]), 0, None))
        area_p = np.clip(p[:, 2] - p[:, 0], 0, None) * np.clip(p[:, 3] - p[:, 1], 0, None)
        union = area_g + area_p - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            passed = np.where(union > 0, inter / union, 0.0) >= iou_threshold
        overall.append(float(passed.mean()) if passed.size else 0.0)
        for cat_id in set(cats.tolist()):
            per_class[cat_id].append(float(passed[cats == cat_id].mean()))
    return {"pixels": list(pixels), "overall": overall, "per_class": dict(per_class)}


def equivalent_jitter_px(pass_rate: float, pixels: list, ceiling: list) -> float:
    """Noi suy k (px) sao cho tran nhieu +-k px bang pass_rate cua model.

    Tra ve 0 neu model tot hon nhieu +-pixels[0] px, inf neu kem hon nhieu +-pixels[-1] px.
    """
    if pass_rate >= ceiling[0]:
        return 0.0
    if pass_rate < ceiling[-1]:
        return float("inf")
    # ceiling giam dan theo k; np.interp can x tang dan.
    return float(np.interp(pass_rate, ceiling[::-1], pixels[::-1]))


def _class_sort_key(cat_id):
    # category_id thuong la int; nhan ngoai taxonomy co the la None hoac str.
    return (cat_id is None, not isinstance(cat_id, int), cat_id if isinstance(cat_id, int) else 0, str(cat_id))


def error_analysis(gt_by_image: dict, pred_by_image: dict, category_names: dict,
                   iou_threshold: float = 0.5, ioa_threshold: float = 0.8,
                   bg_iou_threshold: float = 0.1, min_split_coverage: float = 0.5) -> dict:
    """Danh gia 2 giai doan + phan loai loi + tach/gop, tong hop tren toan bo anh.

    gt_by_image / pred_by_image: {image_id: [{"category_id", "bbox": [x, y, w, h]}, ...]}.

    Tra ve dict:
      overall: Det-P/R/F1, Cls-Acc, E2E-P/R/F1, thong ke IoU, dem fate,
               thong ke split/merge/complex, chi so theo trang
      per_gt_class: {cat_id: {"name", "gt", <GT_FATES>, det_recall, cls_accuracy, e2e_recall}}
      per_pred_class: {cat_id: {"name", "pred", <PRED_FATES>, e2e_precision}}
      confusion: Counter {(gt_cat | BACKGROUND, pred_cat | BACKGROUND): count}
        tren cac cap Det-TP, cong GT bi bo sot (cot BACKGROUND) va pred
        background (hang BACKGROUND).
      merge_combos: Counter {tuple ten lop GT bi gop chung: so box gop}
    """
    gt_counts = defaultdict(Counter)
    pred_counts = defaultdict(Counter)
    confusion = Counter()
    merge_combos = Counter()
    split_fragments_by_class = Counter()
    merged_into_by_class = Counter()
    matched_ious = []
    # Cac cap 1-1 co IoU >= 0.5 (da "tim thay" vung): dung de tach rieng do khit
    # cua box khoi cac loi gop/tach/bo sot.
    pair_ious = defaultdict(list)
    pair_edge_err = defaultdict(list)
    tolerance = defaultdict(list)
    split = {"gt": 0, "fragments": 0}
    merge = {"boxes": 0, "gt": 0, "same_class": 0, "cross_class": 0}
    complex_ = {"groups": 0, "gt": 0, "pred": 0}
    pages = {"images": 0, "perfect": 0, "gt_errors": 0, "pred_errors": 0}

    for image_id in set(gt_by_image) | set(pred_by_image):
        gt_boxes = gt_by_image.get(image_id, [])
        pred_boxes = pred_by_image.get(image_id, [])
        res = analyze_image(gt_boxes, pred_boxes, iou_threshold, ioa_threshold,
                            bg_iou_threshold, min_split_coverage)

        for gi, fate in enumerate(res["gt_fates"]):
            gt_counts[gt_boxes[gi]["category_id"]][fate] += 1
            if fate == "missed":
                confusion[(gt_boxes[gi]["category_id"], BACKGROUND)] += 1
        for pi, fate in enumerate(res["pred_fates"]):
            pred_counts[pred_boxes[pi].get("category_id")][fate] += 1
            if fate == "background":
                confusion[(BACKGROUND, pred_boxes[pi].get("category_id"))] += 1
        for gi, pi, score in res["matches"]:
            confusion[(gt_boxes[gi]["category_id"], pred_boxes[pi].get("category_id"))] += 1
            matched_ious.append(score)

        pairs = [pair for pair in res["matches"] + res["loc_pairs"] if pair[2] >= 0.5]
        for (gi, _, score), err in zip(pairs, _edge_errors(gt_boxes, pred_boxes, pairs)):
            cat_id = gt_boxes[gi]["category_id"]
            pair_ious[cat_id].append(score)
            pair_edge_err[cat_id].append(float(err))
        if gt_boxes:
            wh = np.asarray([b["bbox"][2:4] for b in gt_boxes], dtype=float)
            for gt, tol in zip(gt_boxes, iou_tolerance_px(wh[:, 0], wh[:, 1], iou_threshold)):
                tolerance[gt["category_id"]].append(float(tol))

        for group in res["groups"]:
            if group["type"] == "split":
                split["gt"] += 1
                split["fragments"] += len(group["pred"])
                split_fragments_by_class[gt_boxes[group["gt"][0]]["category_id"]] += len(group["pred"])
            elif group["type"] == "merge":
                cats = [gt_boxes[gi]["category_id"] for gi in group["gt"]]
                merge["boxes"] += 1
                merge["gt"] += len(cats)
                merge["same_class" if len(set(cats)) == 1 else "cross_class"] += 1
                merge_combos[tuple(sorted(category_names.get(c, str(c)) for c in set(cats)))] += 1
                merged_into_by_class[pred_boxes[group["pred"][0]].get("category_id")] += len(cats)
            else:
                complex_["groups"] += 1
                complex_["gt"] += len(group["gt"])
                complex_["pred"] += len(group["pred"])

        gt_err = sum(f != "correct" for f in res["gt_fates"])
        pred_err = sum(f != "correct" for f in res["pred_fates"])
        pages["images"] += 1
        pages["gt_errors"] += gt_err
        pages["pred_errors"] += pred_err
        pages["perfect"] += gt_err == 0 and pred_err == 0

    def localization_stats(ious: list, errs: list, tols: list) -> dict:
        ious, errs = np.asarray(ious, dtype=float), np.asarray(errs, dtype=float)
        return {
            # Trong cac box da tim thay (IoU >= 0.5), ti le dat nguong iou_threshold.
            "loc_pairs": int(ious.size),
            "loc_pass_rate": float((ious >= iou_threshold).mean()) if ious.size else 0.0,
            "edge_err_median": float(np.median(errs)) if errs.size else 0.0,
            "edge_err_p90": float(np.percentile(errs, 90)) if errs.size else 0.0,
            "edge_err_le_1px": float((errs <= 1).mean()) if errs.size else 0.0,
            "edge_err_le_2px": float((errs <= 2).mean()) if errs.size else 0.0,
            "tolerance_median": float(np.median(tols)) if len(tols) else 0.0,
        }

    per_gt_class = {}
    for cat_id in sorted(gt_counts, key=_class_sort_key):
        counts = gt_counts[cat_id]
        n = sum(counts.values())
        det_tp = counts["correct"] + counts["cls_wrong"]
        per_gt_class[cat_id] = {
            "name": category_names.get(cat_id, str(cat_id)),
            "gt": n,
            **{fate: counts[fate] for fate in GT_FATES},
            "split_fragments": split_fragments_by_class[cat_id],
            "det_recall": det_tp / n if n else 0.0,
            "cls_accuracy": counts["correct"] / det_tp if det_tp else 0.0,
            "e2e_recall": counts["correct"] / n if n else 0.0,
            **localization_stats(pair_ious[cat_id], pair_edge_err[cat_id], tolerance[cat_id]),
        }

    per_pred_class = {}
    for cat_id in sorted(pred_counts, key=_class_sort_key):
        counts = pred_counts[cat_id]
        n = sum(counts.values())
        det_tp = counts["correct"] + counts["cls_wrong"]
        per_pred_class[cat_id] = {
            "name": category_names.get(cat_id, "(outside taxonomy)" if cat_id is None else str(cat_id)),
            "pred": n,
            **{fate: counts[fate] for fate in PRED_FATES},
            "merged_gt": merged_into_by_class[cat_id],
            "det_precision": det_tp / n if n else 0.0,
            "e2e_precision": counts["correct"] / n if n else 0.0,
        }

    gt_fates = Counter({f: sum(c[f] for c in gt_counts.values()) for f in GT_FATES})
    pred_fates = Counter({f: sum(c[f] for c in pred_counts.values()) for f in PRED_FATES})
    n_gt, n_pred = sum(gt_fates.values()), sum(pred_fates.values())
    det_tp = gt_fates["correct"] + gt_fates["cls_wrong"]
    det_p, det_r, det_f1 = _prf(det_tp, n_pred, n_gt)
    e2e_p, e2e_r, e2e_f1 = _prf(gt_fates["correct"], n_pred, n_gt)
    ious = np.asarray(matched_ious, dtype=float)
    n_pages = pages["images"]

    overall = {
        "iou_threshold": iou_threshold,
        "ioa_threshold": ioa_threshold,
        "bg_iou_threshold": bg_iou_threshold,
        "min_split_coverage": min_split_coverage,
        "n_gt": n_gt,
        "n_pred": n_pred,
        "pred_gt_ratio": n_pred / n_gt if n_gt else 0.0,
        # Giai doan 1: Detection (khong xet nhan, khop 1-1)
        "det_tp": det_tp,
        "det_fp": n_pred - det_tp,
        "det_fn": n_gt - det_tp,
        "det_precision": det_p,
        "det_recall": det_r,
        "det_f1": det_f1,
        "iou_mean": float(ious.mean()) if ious.size else 0.0,
        "iou_median": float(np.median(ious)) if ious.size else 0.0,
        "iou_p10": float(np.percentile(ious, 10)) if ious.size else 0.0,
        "iou_ge_075": float((ious >= 0.75).mean()) if ious.size else 0.0,
        "iou_ge_095": float((ious >= 0.95).mean()) if ious.size else 0.0,
        **localization_stats(
            [v for c in pair_ious.values() for v in c],
            [v for c in pair_edge_err.values() for v in c],
            [v for c in tolerance.values() for v in c],
        ),
        # Giai doan 2: Classification tren cac box Det-TP
        "cls_correct": gt_fates["correct"],
        "cls_wrong": gt_fates["cls_wrong"],
        "cls_accuracy": gt_fates["correct"] / det_tp if det_tp else 0.0,
        # End-to-end: E2E-Recall = Det-Recall x Cls-Accuracy
        "e2e_precision": e2e_p,
        "e2e_recall": e2e_r,
        "e2e_f1": e2e_f1,
        "gt_fates": gt_fates,
        "pred_fates": pred_fates,
        "split": {
            **split,
            "rate": split["gt"] / n_gt if n_gt else 0.0,
            "mean_fragments": split["fragments"] / split["gt"] if split["gt"] else 0.0,
        },
        "merge": {
            **merge,
            "rate": merge["gt"] / n_gt if n_gt else 0.0,
            "mean_gt_per_box": merge["gt"] / merge["boxes"] if merge["boxes"] else 0.0,
        },
        "complex": complex_,
        "pages": {
            **pages,
            "perfect_rate": pages["perfect"] / n_pages if n_pages else 0.0,
            "mean_gt_errors": pages["gt_errors"] / n_pages if n_pages else 0.0,
            "mean_pred_errors": pages["pred_errors"] / n_pages if n_pages else 0.0,
        },
    }
    return {
        "overall": overall,
        "per_gt_class": per_gt_class,
        "per_pred_class": per_pred_class,
        "confusion": confusion,
        "merge_combos": merge_combos,
    }


def error_analysis_multi_iou(gt_by_image: dict, pred_by_image: dict, category_names: dict,
                             iou_thresholds: list[float] | None = None, **kwargs) -> list[dict]:
    """Chay error_analysis tren nhieu nguong IoU (mac dinh 0.5, 0.75, 0.95)."""
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75, 0.95]
    return [
        error_analysis(gt_by_image, pred_by_image, category_names, iou_threshold=t, **kwargs)
        for t in iou_thresholds
    ]

