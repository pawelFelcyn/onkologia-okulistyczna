"""
kermany_dataset.py
==================
KerMany OCT Dataset utilities:
  - download_kermany()  – download from Kaggle
  - get_transforms()    – augmentation pipeline (train / val / test)
  - build_dataloaders() – DataLoaders for train / val / test
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASSES     = ["CNV", "DME", "DRUSEN", "NORMAL"]
NUM_CLASSES = 4

# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------

def download_kermany(dest_dir: str = "./OCT2018") -> None:
    """
    Download the Kermany OCT Dataset from Kaggle and unzip it to dest_dir.

    Requirements:
        pip install kaggle
        ~/.kaggle/kaggle.json  (token from https://www.kaggle.com/settings)

    Dataset: https://www.kaggle.com/datasets/paultimothymooney/kermany2018

    Expected layout after extraction:
        <dest_dir>/
            train/  CNV/  DME/  DRUSEN/  NORMAL/
            test/   CNV/  DME/  DRUSEN/  NORMAL/
    """
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("[ERROR] Package 'kaggle' not found. Install it: pip install kaggle")
        sys.exit(1)

    dest = Path(dest_dir)
    if dest.exists() and any(dest.iterdir()):
        print(f"[INFO] '{dest}' already exists and is not empty – skipping download.")
        return

    dest.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading Kermany OCT dataset to '{dest}' …"))

    import subprocess
    subprocess.run(
        [
            sys.executable, "-m", "kaggle", "datasets", "download",
            "-d", "paultimothymooney/kermany2018",
            "--unzip", "-p", str(dest),
        ],
        check=True,
    )
    print("[INFO] Download complete.")


# ---------------------------------------------------------------------------
# Transforms / Augmentation
# ---------------------------------------------------------------------------

def get_transforms(split: str) -> T.Compose:
    """
    Return the transform pipeline for the given split.

    Train:
        Resize(256) → RandomCrop(224) → HorizontalFlip → Rotation(±10°)
        → ColorJitter → ToTensor → Normalize(0.5, 0.5)

    Val / Test:
        Resize(224) → CenterCrop(224) → ToTensor → Normalize(0.5, 0.5)

    Note: VerticalFlip is intentionally disabled – OCT images have a fixed
    anatomical top-to-bottom orientation.
    """
    normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    if split == "train":
        return T.Compose([
            T.Resize(256),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            normalize,
        ])
    else:
        return T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize,
        ])


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def build_dataloaders(
    data_dir:    str,
    batch_size:  int,
    val_split:   float = 0.1,
    num_workers: int   = 4,
) -> tuple:
    """
    Build DataLoaders for train / val / test.

    Expected data_dir layout:
        data_dir/
            train/  CNV/  DME/  DRUSEN/  NORMAL/
            test/   CNV/  DME/  DRUSEN/  NORMAL/

    Splits val_split fraction from the training set (no augmentation on val).

    Returns:
        (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_dir)

    train_full   = ImageFolder(str(data_dir / "train"), transform=get_transforms("train"))
    test_dataset = ImageFolder(str(data_dir / "test"),  transform=get_transforms("test"))

    n_total = len(train_full)
    n_val   = int(n_total * val_split)
    n_train = n_total - n_val

    train_subset, val_subset = torch.utils.data.random_split(
        train_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # Validation split uses val/test transforms (no augmentation)
    val_subset.dataset = ImageFolder(
        str(data_dir / "train"), transform=get_transforms("val")
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"[INFO] Train: {n_train}  |  Val: {n_val}  |  Test: {len(test_dataset)}")
    print(f"[INFO] Classes: {train_full.classes}")

    return train_loader, val_loader, test_loader
