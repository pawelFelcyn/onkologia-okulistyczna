"""
eval_kermany.py
===============
Evaluate a trained Kermany OCT classifier.

Loads encoder weights from a .pth file, runs inference on the test set and reports:
  - Accuracy (macro)
  - F1 (macro + per-class)
  - Confusion matrix

Results are saved to <output_dir>/eval_results.json.

Usage:
    python eval_kermany.py --weights ./runs_kermany/encoder_kermany_pretrained.pth \\
                           --data_dir ./OCT2018
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
)
from tqdm import tqdm

from kermany_dataset import build_dataloaders, CLASSES, NUM_CLASSES
from kermany_model import KermanyClassifier

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    weights_path: str,
    data_dir:     str   = "./OCT2018",
    batch_size:   int   = 32,
    base_channels:int   = 64,
    num_workers:  int   = 4,
    output_dir:   str   = "./runs_kermany",
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # DataLoader – test split only
    _, _, test_loader = build_dataloaders(
        data_dir, batch_size, val_split=0.0, num_workers=num_workers
    )

    # Model
    model = KermanyClassifier(
        in_channels=3, base=base_channels, num_classes=NUM_CLASSES
    ).to(device)

    weights = Path(weights_path)
    if not weights.exists():
        raise FileNotFoundError(f"Nie znaleziono wag: {weights}")

    # Accept both a full model checkpoint and an encoder-only checkpoint
    state = torch.load(weights, map_location=device)
    try:
        model.load_state_dict(state)
        print(f"[INFO] Loaded full model from {weights}")
    except RuntimeError:
        model.encoder.load_state_dict(state)
        print(f"[INFO] Loaded encoder from {weights}")

    # Metrics
    criterion  = nn.CrossEntropyLoss()
    acc_fn     = MulticlassAccuracy(num_classes=NUM_CLASSES, average="macro").to(device)
    f1_macro   = MulticlassF1Score(num_classes=NUM_CLASSES,  average="macro").to(device)
    f1_per_cls = MulticlassF1Score(num_classes=NUM_CLASSES,  average="none").to(device)
    cm_fn      = MulticlassConfusionMatrix(num_classes=NUM_CLASSES).to(device)

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating"):
            imgs   = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            total_loss += criterion(logits, labels).item()

            preds = logits.argmax(dim=1)
            acc_fn.update(preds, labels)
            f1_macro.update(preds, labels)
            f1_per_cls.update(preds, labels)
            cm_fn.update(preds, labels)

    avg_loss    = total_loss / len(test_loader)
    accuracy    = acc_fn.compute().item()
    f1_score    = f1_macro.compute().item()
    f1_classes  = f1_per_cls.compute().tolist()
    conf_matrix = cm_fn.compute().cpu().tolist()

    # Print results
    print(f"\n{'='*45}")
    print(f"  Test Loss : {avg_loss:.4f}")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  F1 macro  : {f1_score:.4f}")
    print(f"\n  F1 per class:")
    for cls, f1 in zip(CLASSES, f1_classes):
        print(f"    {cls:<8}: {f1:.4f}")
    print(f"\n  Confusion matrix (rows = true, cols = predicted):")
    header = "         " + "  ".join(f"{c:>7}" for c in CLASSES)
    print(header)
    for cls, row in zip(CLASSES, conf_matrix):
        print(f"  {cls:<7}  " + "  ".join(f"{v:>7}" for v in row))
    print(f"{'='*45}\n")

    results = {
        "test_loss": avg_loss,
        "accuracy":  accuracy,
        "f1_macro":  f1_score,
        "f1_per_class": dict(zip(CLASSES, f1_classes)),
        "confusion_matrix": {
            "classes": CLASSES,
            "matrix":  conf_matrix,
        },
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Results saved to: {results_path}")

    return results


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate Kermany OCT classifier"
    )
    p.add_argument("--weights",      required=True,
                   help="Path to .pth file (encoder-only or full model)")
    p.add_argument("--data_dir",     default="./OCT2018")
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--base",         type=int, default=64,
                   help="Base channels of the encoder (default: 64)")
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--output_dir",   default="./runs_kermany")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        weights_path  = args.weights,
        data_dir      = args.data_dir,
        batch_size    = args.batch_size,
        base_channels = args.base,
        num_workers   = args.num_workers,
        output_dir    = args.output_dir,
    )
