r"""Training script for Method 3: Book-Level Context Encoder & Context Fusion with DINO + M2Doc.

Thực hiện quy trình huấn luyện và đánh giá theo chuẩn Machine Learning:
1. Train Loop: Huấn luyện trên tập Train (toàn bộ DocLayNet: 10,604 ảnh / 328 sách / ~2,673 cửa sổ liên trang).
2. Validation Loop (sau mỗi epoch): Kiểm tra độ tổng quát hóa trên tập Validation / Test (doclaynet_test),
   tính toán val_loss và val_acc mà không cập nhật gradient.
3. Checkpoint Tracking: Tự động lưu Best Model Checkpoint dựa trên val_acc để ngăn ngừa overfitting.
4. History Logging: Tự động lưu lịch sử chi tiết ra cả 2 định dạng:
   - methods/method3/report/training_history.json
   - methods/method3/report/training_history.csv
5. Báo cáo trực quan: Tự động cập nhật đồ thị đối chiếu Train vs Val tại:
   - methods/method3/report/training_report.png
6. Tùy chọn --eval-after-train: Tự động chạy suy luận và đánh giá metric chuẩn COCO (Strict, Loose, mAP)
   sau khi hoàn thành toàn bộ epochs.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parent
METHOD_DIR = SRC_DIR.parent
ROOT_DIR = METHOD_DIR.parents[1]

for p in (SRC_DIR, ROOT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from book_context import ContextAwareBoxRefiner, HierarchicalBookEncoder
from dataset import BookContextWindowDataset
from pipeline1_docling import DoclingPageScout


class BookContextTrainerModule(nn.Module):
    """Module kết hợp HierarchicalBookEncoder và ContextAwareBoxRefiner phục vụ huấn luyện."""

    def __init__(self, num_classes: int = 11, embed_dim: int = 256, num_summary_tokens: int = 4):
        super().__init__()
        self.book_encoder = HierarchicalBookEncoder(
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_summary_tokens=num_summary_tokens,
        )
        self.box_refiner = ContextAwareBoxRefiner(
            num_classes=num_classes,
            embed_dim=embed_dim,
        )

    def forward(self, sample: dict, device: str = "cpu") -> dict:
        spatial = sample["spatial"].to(device)  # (W, M, 8)
        init_labels = sample["init_labels"].to(device)  # (W, M)
        init_scores = sample["init_scores"].to(device)  # (W, M)
        pad_mask = sample["pad_mask"].to(device)  # (W, M)
        page_indices = sample["page_indices"].to(device)  # (W,)
        total_pages = sample["total_pages"].to(device)  # (W,)

        c_book, c_pages = self.book_encoder(
            spatial_features=spatial,
            labels=init_labels,
            scores=init_scores,
            page_indices=page_indices,
            padding_mask=pad_mask,
            total_pages=total_pages,
        )
        out = self.box_refiner(
            spatial_features=spatial,
            init_labels=init_labels,
            init_scores=init_scores,
            book_context=c_book,
            page_context_vec=c_pages,
        )
        return out


def compute_multitask_loss(
    pred_out: dict,
    gt_labels: torch.Tensor,
    gt_deltas: torch.Tensor,
    lambda_cls: float = 2.0,
    lambda_box: float = 5.0,
    lambda_giou: float = 2.0,
) -> Tuple[torch.Tensor, dict]:
    r"""Tính hàm loss đa nhiệm có kiểm tra an toàn số học (tránh chia cho 0 / NaN):
        L = \lambda_{cls} L_{cls} + \lambda_{box} L_{box} + \lambda_{giou} L_{giou}
    """
    cls_logits = pred_out["cls_logits"]  # (W, M, C)
    box_deltas = pred_out["box_deltas"]  # (W, M, 4)

    W, M, C = cls_logits.shape
    valid_mask = gt_labels != -100

    if valid_mask.any():
        loss_cls = F.cross_entropy(cls_logits.view(W * M, C), gt_labels.view(W * M), ignore_index=-100)
        loss_box = F.l1_loss(box_deltas[valid_mask], gt_deltas[valid_mask])
        loss_giou = F.smooth_l1_loss(box_deltas[valid_mask], gt_deltas[valid_mask], beta=0.01)
    else:
        loss_cls = torch.tensor(0.0, device=cls_logits.device, requires_grad=True)
        loss_box = torch.tensor(0.0, device=cls_logits.device, requires_grad=True)
        loss_giou = torch.tensor(0.0, device=cls_logits.device, requires_grad=True)

    # Đảm bảo triệt để không bao giờ trả về NaN
    loss_cls = torch.nan_to_num(loss_cls, nan=0.0, posinf=10.0, neginf=0.0)
    loss_box = torch.nan_to_num(loss_box, nan=0.0, posinf=1.0, neginf=0.0)
    loss_giou = torch.nan_to_num(loss_giou, nan=0.0, posinf=1.0, neginf=0.0)

    total_loss = lambda_cls * loss_cls + lambda_box * loss_box + lambda_giou * loss_giou
    return total_loss, {
        "loss_cls": float(loss_cls.item()),
        "loss_box": float(loss_box.item()),
        "loss_giou": float(loss_giou.item()),
    }


def save_history_files(history: dict, report_dir: Path):
    """Lưu lịch sử huấn luyện ra JSON và CSV."""
    report_dir.mkdir(parents=True, exist_ok=True)

    # 1. Lưu file JSON
    json_path = report_dir / "training_history.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # 2. Lưu file CSV
    csv_path = report_dir / "training_history.csv"
    keys = list(history.keys())
    num_rows = len(history["epoch"])
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(num_rows):
            writer.writerow([history[k][i] for k in keys])


def plot_training_metrics(history: dict, plot_path: Path):
    """Vẽ biểu đồ theo dõi đa chỉ số trực quan 4 bảng đối chiếu Train vs Validation."""
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10), dpi=150)
        epochs_x = history["epoch"]
        has_val = len(history.get("val_loss", [])) == len(epochs_x)

        # 1. Các thành phần Loss: Train vs Val
        ax1.plot(epochs_x, history["train_loss"], "r-o", linewidth=2.0, label="Train Total Loss")
        if has_val:
            ax1.plot(epochs_x, history["val_loss"], "r--s", linewidth=2.0, alpha=0.85, label="Val Total Loss")
        ax1.plot(epochs_x, history["train_cls"], "b:^", linewidth=1.2, alpha=0.7, label="Train Cls Loss")
        ax1.plot(epochs_x, history["train_box"], "g:v", linewidth=1.2, alpha=0.7, label="Train Box Loss (L1)")
        ax1.set_title("Multi-Task Loss over Epochs (Train vs Val)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Epoch", fontsize=9)
        ax1.set_ylabel("Loss Value", fontsize=9)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper right", frameon=True)

        # 2. Box Classification Accuracy: Train vs Val
        ax2.plot(epochs_x, history["train_acc"], "g-o", linewidth=2.2, label="Train Box Cls Acc (%)")
        if has_val:
            ax2.plot(epochs_x, history["val_acc"], "b--s", linewidth=2.2, label="Val Box Cls Acc (%)")
        ax2.set_title("Box Classification Accuracy over Epochs", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Epoch", fontsize=9)
        ax2.set_ylabel("Accuracy (%)", fontsize=9)
        all_acc = history["train_acc"] + (history["val_acc"] if has_val else [])
        ax2.set_ylim(min(50.0, min(all_acc) - 5 if all_acc else 0.0), 100.5)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="lower right", frameon=True)

        # 3. Learning Rate Schedule
        if "lr" in history and history["lr"]:
            ax3.plot(epochs_x, history["lr"], "orange", marker="o", linewidth=2.0, label="Learning Rate")
            ax3.set_title("Learning Rate Schedule (Cosine Annealing)", fontsize=11, fontweight="bold")
            ax3.set_xlabel("Epoch", fontsize=9)
            ax3.set_ylabel("LR", fontsize=9)
            ax3.grid(True, linestyle="--", alpha=0.5)
            ax3.legend(loc="upper right", frameon=True)
            ax3.ticklabel_format(style="sci", scilimits=(0, 0), axis="y")

        # 4. Thời gian Epoch & Bộ nhớ GPU VRAM
        if "epoch_time" in history and history["epoch_time"]:
            ax4_color = "#1f77b4"
            ax4.set_xlabel("Epoch", fontsize=9)
            ax4.set_ylabel("Thời gian (giây)", color=ax4_color, fontsize=9)
            line1 = ax4.plot(epochs_x, history["epoch_time"], color=ax4_color, marker="s", label="Epoch Time (s)")
            ax4.tick_params(axis="y", labelcolor=ax4_color)
            ax4.grid(True, linestyle="--", alpha=0.4)

            if "vram_gb" in history and any(v > 0 for v in history["vram_gb"]):
                ax4_twin = ax4.twinx()
                ax4_twin_color = "#d62728"
                ax4_twin.set_ylabel("GPU VRAM đỉnh (GB)", color=ax4_twin_color, fontsize=9)
                line2 = ax4_twin.plot(epochs_x, history["vram_gb"], color=ax4_twin_color, marker="^", linestyle="--", label="Peak VRAM (GB)")
                ax4_twin.tick_params(axis="y", labelcolor=ax4_twin_color)
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax4.legend(lines, labels, loc="upper right")
            else:
                ax4.legend(loc="upper right")
            ax4.set_title("Hiệu năng Huấn luyện (Thời gian & VRAM)", fontsize=11, fontweight="bold")

        plt.suptitle("Method 3 — Hierarchical Book-Level Context Training Report", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(str(plot_path), bbox_inches="tight")
        plt.close()
        print(f"[Method 3] Đã cập nhật đồ thị báo cáo huấn luyện tại: {plot_path}")
    except Exception as e:
        print(f"[Method 3] Cảnh báo khi xuất đồ thị báo cáo: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train Method 3 Book-Level Context & Context Fusion with Validation")
    parser.add_argument("--dataset", default="doclaynet", help="Train dataset name (mặc định: doclaynet)")
    parser.add_argument("--val-dataset", default="doclaynet_test", help="Validation dataset name (mặc định: doclaynet_test)")
    parser.add_argument("--epochs", type=int, default=12, help="Số epochs huấn luyện (mặc định: 12)")
    parser.add_argument("--window-size", type=int, default=8, help="Số trang liên tiếp trong 1 cửa sổ sách (mặc định: 8)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate ban đầu (mặc định: 2e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (mặc định: 1e-4)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--limit-pages",
        type=int,
        default=0,
        help="Giới hạn số trang nạp cho tập train (0 = toàn bộ dataset, không giới hạn)",
    )
    parser.add_argument(
        "--val-limit-windows",
        type=int,
        default=100,
        help="Số cửa sổ sách kiểm tra trong tập validation sau mỗi epoch (0 = toàn bộ tập val)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=str(METHOD_DIR / "weights" / "method3_book_context_best.pth"),
        help="Đường dẫn lưu trọng số tốt nhất",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Không lưu lại file trọng số (.pth) sau khi huấn luyện",
    )
    parser.add_argument(
        "--no-val",
        action="store_true",
        help="Bỏ qua bước validation trong lúc huấn luyện",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=str(METHOD_DIR / "report"),
        help="Thư mục lưu báo cáo lịch sử (JSON, CSV, PNG)",
    )
    parser.add_argument(
        "--eval-after-train",
        action="store_true",
        help="Tự động chạy suy luận và tính COCO mAP trên tập validation/test sau khi hoàn thành huấn luyện",
    )
    parser.add_argument(
        "--viz-samples",
        type=int,
        default=5,
        help="Số lượng trang tập val xuất ảnh trực quan so sánh Ground Truth vs Dự đoán (mặc định: 5)",
    )
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    plot_report_path = report_dir / "training_report.png"

    print("=" * 85)
    print(" HUẤN LUYỆN METHOD 3: HIERARCHICAL BOOK-LEVEL CONTEXT ENCODER & CONTEXT FUSION")
    print("=" * 85)
    print(f"✓ Thiết bị tính toán : {args.device.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"✓ Tập Train          : {args.dataset} (Limit: {'Toàn bộ (10,604 ảnh)' if args.limit_pages <= 0 else args.limit_pages})")
    print(f"✓ Tập Validation     : {'TẮT' if args.no_val else f'{args.val_dataset} (Limit: {args.val_limit_windows} windows)'}")
    print(f"✓ Số Epochs          : {args.epochs}")
    print(f"✓ Book Window Size   : {args.window_size} trang / cửa sổ")
    print(f"✓ Learning Rate      : {args.lr} (Cosine Annealing Scheduler)")
    print(f"✓ Checkpoint Save    : {'TẮT (--no-save)' if args.no_save else args.save_path}")
    print(f"✓ Báo cáo & Lịch sử  : {report_dir} (JSON, CSV & PNG)")
    print("=" * 85)

    # 1. Chuẩn bị tập Train
    if args.dataset == "doclaynet":
        train_anno = ROOT_DIR / "datasets" / "doclaynet" / "processed" / "annotations" / "train.json"
        train_img_dir = ROOT_DIR / "datasets" / "doclaynet" / "processed" / "images"
    else:
        train_anno = ROOT_DIR / "datasets" / args.dataset / "processed" / "annotations" / "test.json"
        train_img_dir = ROOT_DIR / "datasets" / args.dataset / "processed" / "images"
        if not train_anno.is_file():
            train_anno = ROOT_DIR / "datasets" / args.dataset / "processed" / "annotations" / "train.json"

    train_scout = DoclingPageScout(
        cache_dir=ROOT_DIR / "outputs" / "cache" / "method3_docling" / args.dataset,
        existing_predictions_dir=(ROOT_DIR / "outputs" / "predictions" / "docling" / args.dataset),
        lazy_init=True,
    )

    print(f"\n[Dataset] Đang nạp tập Train từ: {train_anno.name}...")
    train_dataset = BookContextWindowDataset(
        annotation_file=train_anno,
        images_dir=train_img_dir,
        scout=train_scout,
        window_size=args.window_size,
        stride=max(1, args.window_size // 2),
        limit_pages=args.limit_pages if args.limit_pages > 0 else None,
    )
    num_train_windows = len(train_dataset)
    print(f"[Dataset] Train Windows: {num_train_windows:,} cửa sổ sách.")

    # 2. Chuẩn bị tập Validation (nếu không tắt)
    val_dataset = None
    if not args.no_val:
        val_anno = ROOT_DIR / "datasets" / args.val_dataset / "processed" / "annotations" / "test.json"
        if not val_anno.is_file():
            val_anno = ROOT_DIR / "datasets" / args.val_dataset / "processed" / "annotations" / "train.json"
        val_img_dir = ROOT_DIR / "datasets" / args.val_dataset / "processed" / "images"

        val_scout = DoclingPageScout(
            cache_dir=ROOT_DIR / "outputs" / "cache" / "method3_docling" / args.val_dataset,
            existing_predictions_dir=(ROOT_DIR / "outputs" / "predictions" / "docling" / args.val_dataset),
            lazy_init=True,
        )
        val_dataset = BookContextWindowDataset(
            annotation_file=val_anno,
            images_dir=val_img_dir,
            scout=val_scout,
            window_size=args.window_size,
            stride=max(1, args.window_size // 2),
            limit_pages=None,
        )
        val_len = len(val_dataset) if args.val_limit_windows <= 0 else min(args.val_limit_windows, len(val_dataset))
        print(f"[Dataset] Validation Windows: {val_len:,} cửa sổ sách từ {args.val_dataset}.")

    # Khởi tạo mô hình
    model = BookContextTrainerModule(num_classes=11, embed_dim=256).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_cls": [],
        "train_box": [],
        "train_giou": [],
        "train_acc": [],
        "val_loss": [],
        "val_cls": [],
        "val_box": [],
        "val_giou": [],
        "val_acc": [],
        "lr": [],
        "epoch_time": [],
        "vram_gb": [],
    }

    best_val_acc = 0.0
    save_path = Path(args.save_path)

    print("\nBẮT ĐẦU VÒNG LẶP HUẤN LUYỆN & KIỂM TRA:")
    print("-" * 85)

    try:
        for epoch in range(1, args.epochs + 1):
            # ======================== TRAIN LOOP ========================
            model.train()
            t0 = time.time()
            train_loss = 0.0
            train_cls = 0.0
            train_box = 0.0
            train_giou = 0.0
            train_correct = 0
            train_total = 0

            current_lr = optimizer.param_groups[0]["lr"]

            train_pbar = tqdm(
                range(num_train_windows),
                desc=f"Epoch [{epoch:02d}/{args.epochs:02d}] Train",
                unit="win",
                dynamic_ncols=True,
                file=sys.stdout,
            )

            train_steps = 0
            for idx in train_pbar:
                sample = train_dataset[idx]
                gt_labels = sample["gt_labels"].to(args.device)
                gt_deltas = sample["gt_deltas"].to(args.device)

                valid = gt_labels != -100
                if not valid.any():
                    continue

                optimizer.zero_grad()
                pred_out = model(sample, device=args.device)
                loss, loss_dict = compute_multitask_loss(pred_out, gt_labels, gt_deltas)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                l_val = float(loss.item())
                train_loss += l_val
                train_cls += loss_dict["loss_cls"]
                train_box += loss_dict["loss_box"]
                train_giou += loss_dict["loss_giou"]
                train_steps += 1

                preds = pred_out["cls_logits"].argmax(dim=-1)
                train_correct += int((preds[valid] == gt_labels[valid]).sum().item())
                train_total += int(valid.sum().item())

                cur_acc = (train_correct / max(train_total, 1)) * 100
                cur_mean_loss = train_loss / max(train_steps, 1)

                train_pbar.set_postfix({
                    "loss": f"{cur_mean_loss:.4f}",
                    "cls": f"{train_cls/max(train_steps, 1):.3f}",
                    "box": f"{train_box/max(train_steps, 1):.3f}",
                    "acc": f"{cur_acc:.1f}%",
                    "lr": f"{current_lr:.1e}",
                })

            scheduler.step()

            mean_train_loss = train_loss / max(train_steps, 1)
            mean_train_cls = train_cls / max(train_steps, 1)
            mean_train_box = train_box / max(train_steps, 1)
            mean_train_giou = train_giou / max(train_steps, 1)
            epoch_train_acc = (train_correct / max(train_total, 1)) * 100

            # ===================== VALIDATION LOOP =====================
            mean_val_loss = 0.0
            mean_val_cls = 0.0
            mean_val_box = 0.0
            mean_val_giou = 0.0
            epoch_val_acc = 0.0

            if val_dataset is not None:
                model.eval()
                val_loss_acc = 0.0
                val_cls_acc = 0.0
                val_box_acc = 0.0
                val_giou_acc = 0.0
                val_correct = 0
                val_total = 0
                val_steps = 0

                num_eval_windows = len(val_dataset) if args.val_limit_windows <= 0 else min(args.val_limit_windows, len(val_dataset))

                val_pbar = tqdm(
                    range(num_eval_windows),
                    desc=f"Epoch [{epoch:02d}/{args.epochs:02d}] Val  ",
                    unit="win",
                    dynamic_ncols=True,
                    file=sys.stdout,
                )

                with torch.no_grad():
                    for v_idx in val_pbar:
                        v_sample = val_dataset[v_idx]
                        v_gt_labels = v_sample["gt_labels"].to(args.device)
                        v_gt_deltas = v_sample["gt_deltas"].to(args.device)

                        v_valid = v_gt_labels != -100
                        if not v_valid.any():
                            continue

                        v_out = model(v_sample, device=args.device)
                        v_l, v_ldict = compute_multitask_loss(v_out, v_gt_labels, v_gt_deltas)

                        if torch.isnan(v_l) or torch.isinf(v_l):
                            continue

                        val_loss_acc += float(v_l.item())
                        val_cls_acc += v_ldict["loss_cls"]
                        val_box_acc += v_ldict["loss_box"]
                        val_giou_acc += v_ldict["loss_giou"]
                        val_steps += 1

                        v_preds = v_out["cls_logits"].argmax(dim=-1)
                        val_correct += int((v_preds[v_valid] == v_gt_labels[v_valid]).sum().item())
                        val_total += int(v_valid.sum().item())

                        val_pbar.set_postfix({
                            "val_loss": f"{val_loss_acc/max(val_steps, 1):.4f}",
                            "val_acc": f"{(val_correct/max(val_total, 1))*100:.1f}%",
                        })

                mean_val_loss = val_loss_acc / max(val_steps, 1)
                mean_val_cls = val_cls_acc / max(val_steps, 1)
                mean_val_box = val_box_acc / max(val_steps, 1)
                mean_val_giou = val_giou_acc / max(val_steps, 1)
                epoch_val_acc = (val_correct / max(val_total, 1)) * 100

            dt = time.time() - t0
            speed = num_train_windows / max(dt, 1e-4)
            peak_vram = 0.0
            if torch.cuda.is_available():
                peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)

            # Cập nhật lịch sử
            history["epoch"].append(epoch)
            history["train_loss"].append(round(mean_train_loss, 4))
            history["train_cls"].append(round(mean_train_cls, 4))
            history["train_box"].append(round(mean_train_box, 4))
            history["train_giou"].append(round(mean_train_giou, 4))
            history["train_acc"].append(round(epoch_train_acc, 2))
            history["val_loss"].append(round(mean_val_loss, 4))
            history["val_cls"].append(round(mean_val_cls, 4))
            history["val_box"].append(round(mean_val_box, 4))
            history["val_giou"].append(round(mean_val_giou, 4))
            history["val_acc"].append(round(epoch_val_acc, 2))
            history["lr"].append(current_lr)
            history["epoch_time"].append(round(dt, 1))
            history["vram_gb"].append(round(peak_vram, 2))

            # In bảng tổng kết rõ ràng cuối mỗi Epoch
            print(
                f"--> Epoch {epoch:02d} Summary: "
                f"Train Loss: {mean_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
                f"Val Loss: {mean_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}% | "
                f"Thời gian: {dt:.1f}s ({speed:.1f} win/s) | Peak VRAM: {peak_vram:.2f} GB"
            )

            # Cập nhật checkpoint tốt nhất dựa trên Val Accuracy (hoặc Train Accuracy nếu không có val)
            score_to_check = epoch_val_acc if val_dataset is not None else epoch_train_acc
            if not args.no_save and score_to_check > best_val_acc:
                best_val_acc = score_to_check
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "best_acc": best_val_acc,
                        "best_val_acc": epoch_val_acc,
                        "book_encoder": model.book_encoder.state_dict(),
                        "box_context_refiner": model.box_refiner.state_dict(),
                        "history": history,
                        "config": vars(args),
                    },
                    str(save_path),
                )
                print(f"    ★ [NEW BEST] Đã lưu checkpoint đạt độ chính xác cao nhất ({best_val_acc:.2f}%) tại: {save_path.name}")

            # Lưu file lịch sử JSON, CSV và cập nhật biểu đồ PNG sau mỗi epoch
            save_history_files(history, report_dir)
            plot_training_metrics(history, plot_report_path)

    except KeyboardInterrupt:
        print("\n\n[Method 3] Người dùng đã dừng huấn luyện sớm (KeyboardInterrupt).")
        if len(history["epoch"]) > 0:
            save_history_files(history, report_dir)
            plot_training_metrics(history, plot_report_path)

    print("\n" + "=" * 85)
    print("HOÀN TẤT QUÁ TRÌNH HUẤN LUYỆN METHOD 3!")
    if history["epoch"]:
        print(f"✓ Độ chính xác Val cao nhất     : {max(history['val_acc']):.2f}%")
        print(f"✓ Độ chính xác Train cuối cùng  : {history['train_acc'][-1]:.2f}%")
    print(f"✓ Lịch sử huấn luyện JSON       : {report_dir / 'training_history.json'}")
    print(f"✓ Lịch sử huấn luyện CSV        : {report_dir / 'training_history.csv'}")
    print(f"✓ Đồ thị trực quan Train vs Val : {plot_report_path}")
    if not args.no_save and save_path.is_file():
        print(f"✓ Trọng số tốt nhất đã lưu tại  : {save_path}")
    print("=" * 85)

    # 3. Tự động chạy Final Evaluation trên tập Validation/Test nếu được yêu cầu
    if args.eval_after_train:
        print("\n[Method 3] Đang tiến hành Final Evaluation trên tập đánh giá...")
        eval_ds = args.val_dataset
        split_name = "test" if eval_ds == "doclaynet_test" else "val"

        # Chạy inference
        infer_cmd = [
            sys.executable,
            str(SRC_DIR / "run_inference.py"),
            "--dataset", eval_ds,
            "--fusion-mode", "feature",
            "--window-size", str(args.window_size),
            "--weights", str(METHOD_DIR / "weights" / "dino_m2doc_r50.pth"),
            "--context-weights", str(save_path),
            "--limit", "50",  # Đánh giá trên tập mẫu 50 trang
        ]
        print("Running inference:", " ".join(infer_cmd))
        subprocess.run(infer_cmd, cwd=ROOT_DIR, check=True)

        # Chạy evaluate_predictions.py
        eval_cmd = [
            sys.executable,
            "scripts/evaluate_predictions.py",
            "--method", "method3",
            "--dataset", eval_ds,
            "--split", split_name,
            "--allow-incomplete",
            "--loose",
            "--plot",
        ]
        print("Running evaluation:", " ".join(eval_cmd))
        subprocess.run(eval_cmd, cwd=ROOT_DIR, check=True)

        # Chạy trực quan hóa một vài ảnh tập validation (Side-by-side: Ground Truth vs Method 3)
        if args.viz_samples > 0:
            viz_out_dir = report_dir / "val_visualizations"
            viz_out_dir.mkdir(parents=True, exist_ok=True)
            viz_cmd = [
                sys.executable,
                "visualization/scripts/visualize_predictions.py",
                "--method", "method3",
                "--dataset", eval_ds,
                "--split", split_name,
                "--limit", str(args.viz_samples),
                "--output-dir", str(viz_out_dir),
            ]
            print("\nRunning visualization:", " ".join(viz_cmd))
            subprocess.run(viz_cmd, cwd=ROOT_DIR, check=True)
            print(f"✓ Đã xuất {args.viz_samples} ảnh trực quan đối chiếu GT vs Method 3 tại: {viz_out_dir}")


if __name__ == "__main__":
    main()
