from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import CSVDataset
from src.transforms import get_transforms
from src.utils.seed import seed_everything


# =========================
# Utils
# =========================
def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_from_logits(logits, y) * bs
        n += bs
    return total_loss / max(n, 1), total_acc / max(n, 1)


def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_from_logits(logits, y) * bs
        n += bs
    return total_loss / max(n, 1), total_acc / max(n, 1)


def save_best_checkpoint(best_path: Path, model, args, train_ds, best_val_acc: float):
    payload = {
        "model": args.model,
        "state_dict": model.state_dict(),
        "label_to_idx": getattr(train_ds, "label_to_idx", None),
        "img_size": args.img_size,
        "best_val_acc": best_val_acc,
        "vit_name": getattr(args, "vit_name", None),
        "freeze_backbone": getattr(args, "freeze_backbone", False),
    }
    torch.save(payload, best_path)


# =========================
# Models (self-contained)
# =========================
class SimpleCNN(nn.Module):
    """CNN baseline gọn, ổn định (không hardcode flatten)."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def build_model(args, num_classes: int) -> nn.Module:
    if args.model == "cnn_baseline":
        return SimpleCNN(num_classes)

    if args.model == "cnn_transfer":
        from torchvision import models
        try:
            m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception:
            m = models.resnet18(pretrained=True)

        if args.freeze_backbone:
            for p in m.parameters():
                p.requires_grad = False

        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        for p in m.fc.parameters():
            p.requires_grad = True
        return m

    if args.model == "vit":
        import timm
        # timm will create correct classifier head with num_classes
        return timm.create_model(args.vit_name, pretrained=True, num_classes=num_classes)

    raise ValueError(f"Unknown model: {args.model}")


# =========================
# Plot
# =========================
def try_plot_curves(out_dir: Path, history_rows: list[dict]):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    epochs = [r["epoch"] for r in history_rows]
    tr_loss = [r["train_loss"] for r in history_rows]
    va_loss = [r["val_loss"] for r in history_rows]
    tr_acc = [r["train_acc"] for r in history_rows]
    va_acc = [r["val_acc"] for r in history_rows]

    plt.figure()
    plt.plot(epochs, tr_loss, label="train_loss")
    plt.plot(epochs, va_loss, label="val_loss")
    plt.plot(epochs, tr_acc, label="train_acc")
    plt.plot(epochs, va_acc, label="val_acc")
    plt.legend()
    plt.title("Training Curves")
    plt.savefig(out_dir / "curves.png", dpi=150, bbox_inches="tight")
    plt.close()


# =========================
# Main
# =========================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="cnn_baseline",
                    choices=["cnn_baseline", "cnn_transfer", "vit"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--vit_name", type=str, default="vit_base_patch16_224")

    # paths
    ap.add_argument("--splits_dir", type=str, default="splits")
    ap.add_argument("--model_dir", type=str, default="models")
    ap.add_argument("--output_dir", type=str, default="outputs")

    # dataloader
    ap.add_argument("--num_workers", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # CSV paths
    splits_dir = Path(args.splits_dir)
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(f"Missing splits CSVs: {train_csv} / {val_csv}")

    # data
    train_tf, eval_tf = get_transforms(img_size=args.img_size)
    train_ds = CSVDataset(str(train_csv), transform=train_tf)
    val_ds = CSVDataset(str(val_csv), transform=eval_tf, label_to_idx=train_ds.label_to_idx)

    num_classes = len(train_ds.label_to_idx)
    print("num_classes =", num_classes)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # model
    model = build_model(args, num_classes).to(device)

    # optimizer (only trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # output dirs
    model_dir = Path(args.model_dir) / args.model
    out_dir = Path(args.output_dir) / args.model
    model_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    history_path = out_dir / "history.csv"
    best_path = model_dir / "best.pt"
    best_val_acc = -1.0
    history_rows: list[dict] = []

    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "seconds"])

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            tr_loss, tr_acc = train_one_epoch(model, train_loader, device, optimizer, criterion)
            va_loss, va_acc = evaluate(model, val_loader, device, criterion)

            sec = time.time() - t0

            writer.writerow([epoch, tr_loss, tr_acc, va_loss, va_acc, sec])
            f.flush()

            history_rows.append({
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": va_loss,
                "val_acc": va_acc,
                "seconds": sec,
            })

            print(f"[{args.model}] epoch {epoch}/{args.epochs} | "
                  f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.4f} | {sec:.1f}s")

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                save_best_checkpoint(best_path, model, args, train_ds, best_val_acc)
                print(f"✅ Saved best to {best_path} (val_acc={best_val_acc:.4f})")

    try_plot_curves(out_dir, history_rows)
    if (out_dir / "curves.png").exists():
        print("Saved curves:", out_dir / "curves.png")

    print("DONE. Best val acc:", best_val_acc)


if _name_ == "_main_":
    main()