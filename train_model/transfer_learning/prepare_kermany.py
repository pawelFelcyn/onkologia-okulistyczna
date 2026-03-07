"""
prepare_kermany.py
==================
Prepare the Kermany OCT 2018 dataset for encoder pretraining.

Responsibilities:
    - optionally download the dataset from Kaggle,
  - validate the expected train/test folder structure,
  - print a short summary of image counts.

Usage:
    python prepare_kermany.py --download --data_dir ./OCT2018
    python prepare_kermany.py --data_dir ./OCT2018
"""

import argparse

from kermany_dataset import prepare_kermany_dataset


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare the Kermany OCT dataset for transfer learning"
    )
    p.add_argument(
        "--download",
        action="store_true",
        help="Download dataset from Kaggle before validation",
    )
    p.add_argument(
        "--data_dir",
        default="./OCT2018",
        help="Path to the dataset root",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_kermany_dataset(data_dir=args.data_dir, download=args.download)