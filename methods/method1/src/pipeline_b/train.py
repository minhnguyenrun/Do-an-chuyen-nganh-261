"""Training script for BBox Multimodal Transformer (Pipeline B)."""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import DocLayNetBBoxDataset, pad_collate_fn
from .model import BBoxMultimodalTransformer

ROOT = Path(__file__).resolve().parents[4]


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        crops = batch["crops"].to(device)
        spatial = batch["spatial"].to(device)
        text_embeds = batch["text_embeds"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()
        logits = model(crops, text_embeds, spatial, mask=mask)

        # Tính loss chỉ trên các hộp hợp lệ (bỏ qua padding mask với ignore_index=-100)
        B, N, C = logits.shape
        loss = criterion(logits.view(B * N, C), labels.view(B * N))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B
        valid_mask = labels != -100
        preds = torch.argmax(logits, dim=-1)
        correct += (preds[valid_mask] == labels[valid_mask]).sum().item()
        total += valid_mask.sum().item()

    return total_loss / len(loader.dataset), correct / max(total, 1)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            crops = batch["crops"].to(device)
            spatial = batch["spatial"].to(device)
            text_embeds = batch["text_embeds"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)

            logits = model(crops, text_embeds, spatial, mask=mask)
            B, N, C = logits.shape
            loss = criterion(logits.view(B * N, C), labels.view(B * N))

            total_loss += loss.item() * B
            valid_mask = labels != -100
            preds = torch.argmax(logits, dim=-1)
            correct += (preds[valid_mask] == labels[valid_mask]).sum().item()
            total += valid_mask.sum().item()

    return total_loss / len(loader.dataset), correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Train Pipeline B: BBox Multimodal Transformer")
    parser.add_argument("--epochs", type=int, default=10, help="Số epochs huấn luyện")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (số trang / batch)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--name", default="pipeline_b_transformer")
    args = parser.parse_args()

    print("=" * 80)
    print("HUẤN LUYỆN PIPELINE B: BBOX MULTIMODAL TRANSFORMER")
    print(f"Device: {args.device.upper()} | Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print("=" * 80)

    train_anno = ROOT / "datasets" / "doclaynet" / "processed" / "annotations" / "train.json"
    train_img_dir = ROOT / "datasets" / "doclaynet" / "processed" / "images"

    val_anno = ROOT / "datasets" / "doclaynet_test" / "processed" / "annotations" / "test.json"
    val_img_dir = ROOT / "datasets" / "doclaynet_test" / "processed" / "images"

    train_dataset = DocLayNetBBoxDataset(train_anno, train_img_dir)
    val_dataset = DocLayNetBBoxDataset(val_anno, val_img_dir, limit_samples=500)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate_fn)

    model = BBoxMultimodalTransformer(num_classes=11).to(args.device)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_acc = 0.0
    weights_dir = ROOT / "methods" / "method1" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    save_path = weights_dir / f"{args.name}_best.pth"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, args.device)
        val_loss, val_acc = validate(model, val_loader, criterion, args.device)
        dt = time.time() - t0

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({dt:.1f}s) | "
              f"Train Loss: {train_loss:.4f} - Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} - Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(save_path))
            print(f"  -> Lưu checkpoint tốt nhất mới ({val_acc*100:.2f}%) tại: {save_path.name}")

    print(f"\n✓ Hoàn tất huấn luyện Pipeline B! Trọng số tối ưu: {save_path}")


if __name__ == "__main__":
    main()
