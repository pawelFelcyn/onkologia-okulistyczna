"""
train_kermany.py
================
Stage A – training loop for the Kermany OCT classifier (U-Net encoder pretraining).

Imports:
  kermany_dataset  – DataLoaders
  kermany_model    – KermanyClassifier / UNetEncoder

Saves ONLY the encoder → <output_dir>/encoder_kermany_pretrained.pth
Test evaluation → eval_kermany.py

Usage:
    python train_kermany.py --data_dir ./OCT2018 --epochs 25 --batch_size 32

Download dataset before training:
    python train_kermany.py --download --data_dir ./OCT2018

TensorBoard:
    tensorboard --logdir <output_dir>/tensorboard
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from kermany_dataset import download_kermany, build_dataloaders, NUM_CLASSES
from kermany_model import KermanyClassifier

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def build_metrics(device: torch.device):
    acc = MulticlassAccuracy(num_classes=NUM_CLASSES, average="macro").to(device)
    f1  = MulticlassF1Score(num_classes=NUM_CLASSES,  average="macro").to(device)
    return acc, f1


# ---------------------------------------------------------------------------
# Single epoch
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, device,
              acc_fn, f1_fn, is_train: bool):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    acc_fn.reset()
    f1_fn.reset()

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader, leave=False,
                                 desc="train" if is_train else "val/test"):
            imgs   = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            acc_fn.update(preds, labels)
            f1_fn.update(preds, labels)

    return (
        total_loss / len(loader),
        acc_fn.compute().item(),
        f1_fn.compute().item(),
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    data_dir:       str   = "./OCT2018",
    epochs:         int   = 25,
    batch_size:     int   = 32,
    lr:             float = 1e-4,
    weight_decay:   float = 1e-4,
    early_stopping: int   = 5,
    base_channels:  int   = 64,
    num_workers:    int   = 4,
    output_dir:     str   = "./runs_kermany",
    val_split:      float = 0.1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # DataLoaders (test loader not needed here – evaluation is done in eval_kermany.py)
    train_loader, val_loader, _ = build_dataloaders(
        data_dir, batch_size, val_split=val_split, num_workers=num_workers
    )

    # Model
    model = KermanyClassifier(
        in_channels=3, base=base_channels, num_classes=NUM_CLASSES
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable parameters: {n_params:,}")

    # Loss / Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)

    # Metrics
    train_acc, train_f1 = build_metrics(device)
    val_acc,   val_f1   = build_metrics(device)

    # Output directory + TensorBoard
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out / "tensorboard"))

    best_val_f1 = -1.0
    no_improve  = 0
    history     = []

    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")

        tr_loss, tr_acc, tr_f1 = run_epoch(
            model, train_loader, criterion, optimizer, device,
            train_acc, train_f1, is_train=True,
        )
        vl_loss, vl_acc, vl_f1 = run_epoch(
            model, val_loader, criterion, None, device,
            val_acc, val_f1, is_train=False,
        )

        print(f"  Train  loss={tr_loss:.4f}  acc={tr_acc:.4f}  F1={tr_f1:.4f}")
        print(f"  Val    loss={vl_loss:.4f}  acc={vl_acc:.4f}  F1={vl_f1:.4f}")

        writer.add_scalars("loss",     {"train": tr_loss, "val": vl_loss}, epoch)
        writer.add_scalars("accuracy", {"train": tr_acc,  "val": vl_acc},  epoch)
        writer.add_scalars("f1_macro", {"train": tr_f1,   "val": vl_f1},   epoch)

        history.append(dict(
            epoch=epoch,
            train_loss=tr_loss, train_acc=tr_acc, train_f1=tr_f1,
            val_loss=vl_loss,   val_acc=vl_acc,   val_f1=vl_f1,
        ))

        # Early stopping + save encoder checkpoint only
        if vl_f1 > best_val_f1:
            best_val_f1  = vl_f1
            no_improve   = 0
            encoder_path = out / "encoder_kermany_pretrained.pth"
            torch.save(model.encoder.state_dict(), str(encoder_path))
            print(f"  ✓ Best F1={best_val_f1:.4f} → {encoder_path}")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{early_stopping})")
            if no_improve >= early_stopping:
                print("[INFO] Early stopping triggered.")
                break

    writer.close()

    with open(out / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[DONE] Encoder: {encoder_path}")
    print(f"[DONE] TensorBoard: tensorboard --logdir {out / 'tensorboard'}")
    print(f"[INFO] Ewaluacja: python eval_kermany.py --weights {encoder_path}")
    return model.encoder


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Kermany OCT – U-Net encoder pretraining (classification)"
    )
    p.add_argument("--download",     action="store_true",
                   help="Download dataset from Kaggle before training")
    p.add_argument("--data_dir",     default="./OCT2018")
    p.add_argument("--epochs",       type=int,   default=25)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--early_stop",   type=int,   default=5)
    p.add_argument("--base",         type=int,   default=64,
                   help="Base channels of the U-Net encoder (default: 64)")
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--output_dir",   default="./runs_kermany")
    p.add_argument("--val_split",    type=float, default=0.1)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.download:
        download_kermany(dest_dir=args.data_dir)

    train(
        data_dir       = args.data_dir,
        epochs         = args.epochs,
        batch_size     = args.batch_size,
        lr             = args.lr,
        weight_decay   = args.weight_decay,
        early_stopping = args.early_stop,
        base_channels  = args.base,
        num_workers    = args.num_workers,
        output_dir     = args.output_dir,
        val_split      = args.val_split,
    )
