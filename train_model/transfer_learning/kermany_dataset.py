"""
kermany_dataset.py
==================
Kermany OCT Dataset utilities:
    - download_kermany()          – download from Kaggle
    - validate_kermany_structure()– validate expected directory layout
    - summarize_kermany_dataset() – print and return per-split statistics
    - get_transforms()            – augmentation pipeline (train / val / test)
    - build_dataloaders()         – DataLoaders for train / val / test
"""

import sys
import shutil
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASSES     = ["CNV", "DME", "DRUSEN", "NORMAL"]
NUM_CLASSES = 4


def _expected_split_dirs(data_dir: Path) -> dict[str, dict[str, Path]]:
    return {
        split: {class_name: data_dir / split / class_name for class_name in CLASSES}
        for split in ("train", "test")
    }


def _find_kermany_root(search_root: str | Path) -> Path | None:
    """Find the directory that contains the expected Kermany train/test layout."""
    root = Path(search_root)

    if has_kermany_structure(root):
        return root

    for candidate in sorted((path for path in root.rglob("*") if path.is_dir()), key=lambda p: len(p.parts)):
        if has_kermany_structure(candidate):
            return candidate

    return None


def _copy_kermany_dataset(source_root: Path, dest_root: Path) -> None:
    """Copy the prepared dataset tree into the requested destination directory."""
    dest_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        shutil.copytree(source_root / split, dest_root / split, dirs_exist_ok=True)


def has_kermany_structure(data_dir: str | Path) -> bool:
    """Return True if the dataset root contains the expected split/class folders."""
    root = Path(data_dir)
    expected = _expected_split_dirs(root)
    return all(path.is_dir() for split_dirs in expected.values() for path in split_dirs.values())


def validate_kermany_structure(data_dir: str | Path) -> Path:
    """Validate the Kermany directory layout and return the normalized root path."""
    root = Path(data_dir)
    missing: list[Path] = []

    for split, class_dirs in _expected_split_dirs(root).items():
        split_dir = root / split
        if not split_dir.is_dir():
            missing.append(split_dir)
            continue
        for class_dir in class_dirs.values():
            if not class_dir.is_dir():
                missing.append(class_dir)

    if missing:
        missing_lines = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Invalid Kermany dataset structure. Missing:\n"
            f"{missing_lines}\n\n"
            "Expected layout:\n"
            f"{root}/train/{{CNV,DME,DRUSEN,NORMAL}}\n"
            f"{root}/test/{{CNV,DME,DRUSEN,NORMAL}}"
        )

    return root


def summarize_kermany_dataset(data_dir: str | Path) -> dict[str, Any]:
    """Return and print dataset image counts per split and class."""
    root = validate_kermany_structure(data_dir)

    summary: dict[str, Any] = {"root": str(root), "splits": {}, "total": 0}
    print(f"[INFO] Dataset root: {root}")

    for split in ("train", "test"):
        split_counts: dict[str, int] = {}
        split_total = 0
        for class_name in CLASSES:
            class_dir = root / split / class_name
            count = sum(1 for path in class_dir.iterdir() if path.is_file())
            split_counts[class_name] = count
            split_total += count

        summary["splits"][split] = {
            "classes": split_counts,
            "total": split_total,
        }
        summary["total"] += split_total

        counts_str = ", ".join(f"{cls}={count}" for cls, count in split_counts.items())
        print(f"[INFO] {split:<5} total={split_total} ({counts_str})")

    print(f"[INFO] All images: {summary['total']}")
    return summary


def prepare_kermany_dataset(data_dir: str = "./OCT2018", download: bool = False) -> dict[str, Any]:
    """Optionally download the dataset and then validate/summary the prepared data."""
    if download:
        download_kermany(dest_dir=data_dir)

    summary = summarize_kermany_dataset(data_dir)
    print("[DONE] Dataset is ready for training.")
    return summary

# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------

def download_kermany(dest_dir: str = "./OCT2018") -> None:
    """
    Download the Kermany OCT Dataset to dest_dir.

    Requirements:
        Preferred: pip install kagglehub
        Fallback:  pip install kaggle and configure ~/.kaggle/kaggle.json

    Dataset: https://www.kaggle.com/datasets/paultimothymooney/kermany2018

    Expected layout after preparation:
        <dest_dir>/
            train/  CNV/  DME/  DRUSEN/  NORMAL/
            test/   CNV/  DME/  DRUSEN/  NORMAL/
    """
    dest = Path(dest_dir)
    if has_kermany_structure(dest):
        print(f"[INFO] '{dest}' already contains a valid Kermany dataset – skipping download.")
        return

    if dest.exists() and any(dest.iterdir()):
        raise RuntimeError(
            f"Folder '{dest}' already exists and is not empty, but does not contain a valid Kermany structure. "
            "Use an empty directory or remove the invalid files first."
        )

    dataset_root: Path | None = None

    try:
        import kagglehub

        print("[INFO] Downloading Kermany OCT dataset with kagglehub …")
        cached_path = Path(kagglehub.dataset_download("paultimothymooney/kermany2018"))
        dataset_root = _find_kermany_root(cached_path)
        if dataset_root is None:
            raise FileNotFoundError(
                f"Downloaded dataset was not found under '{cached_path}'."
            )
        print(f"[INFO] kagglehub cache: {cached_path}")
    except ImportError:
        print("[WARN] Package 'kagglehub' not found. Falling back to Kaggle API.")
    except Exception as exc:
        print(f"[WARN] kagglehub download failed: {exc}")
        print("[WARN] Falling back to Kaggle API.")

    if dataset_root is None:
        try:
            import kaggle  # noqa: F401
        except ImportError:
            print(
                "[ERROR] Neither 'kagglehub' nor 'kaggle' is available. "
                "Install 'kagglehub' (preferred) or 'kaggle' as a fallback."
            )
            sys.exit(1)

        dest.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Downloading Kermany OCT dataset to '{dest}' with Kaggle API …")

        import subprocess
        subprocess.run(
            [
                sys.executable, "-m", "kaggle", "datasets", "download",
                "-d", "paultimothymooney/kermany2018",
                "--unzip", "-p", str(dest),
            ],
            check=True,
        )
        dataset_root = _find_kermany_root(dest)

    if dataset_root is None:
        raise FileNotFoundError(
            "Download completed, but the expected train/test/class directory layout was not found."
        )

    if dataset_root.resolve() != dest.resolve():
        print(f"[INFO] Copying prepared dataset to '{dest}' …")
        _copy_kermany_dataset(dataset_root, dest)

    validate_kermany_structure(dest)
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
    data_dir = validate_kermany_structure(data_dir)

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
