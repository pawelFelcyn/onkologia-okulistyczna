import unet_utils
import os
import random
import numpy as np
from torch.utils.data import DataLoader
from utils import get_unique_path
from dotenv import load_dotenv
import argparse
import torch
import re
from pathlib import Path


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

load_dotenv(dotenv_path='train_model/.env')


def get_last_run_model():
    unet_runs = Path('runs_unet')
    max_run = max(
        int(m.group(1))
        for p in unet_runs.iterdir()
        if p.is_dir() and (m := re.fullmatch(r"run(\d+)", p.name))
    )
    max_run_dir = Path(os.path.join(unet_runs, f'run{max_run}'))
    max_epoch = max(
        int(m.group(1))
        for p in max_run_dir.iterdir()
        if p.is_dir() and any(p.iterdir()) and (m := re.fullmatch(r"epoch_(\d+)", p.name))
    )
    return os.path.join(max_run_dir, "weights", "last.pth"), max_epoch


def main(train_csv, val_csv, save_path=None, epochs=50, imgsz=512, batch=16, unet_continue_last_run=False, seed=42):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | seed: {seed}")

    root_dir = os.path.join("Ophthalmic_Scans")
    train_dataset = unet_utils.UNetDataset(train_csv, root_dir, imgsz=imgsz)
    val_dataset = unet_utils.UNetDataset(val_csv, root_dir, imgsz=imgsz)

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset,   batch_size=batch, shuffle=False)
    model = unet_utils.UNet(3, 2)

    if unet_continue_last_run:
        print("Continue training from last run...")
        last_weights, trained_epochs = get_last_run_model()
        print(
            f"Last run will be used to continue training from epoch {trained_epochs + 1}.")
        model.load_state_dict(torch.load(last_weights, map_location=device))
        epochs = epochs - trained_epochs

    if epochs < 0:
        print("No more epochs to train")
        return

    model.train_model(train_loader, val_loader, epochs, device=device)

    default_dir = "models/unet"
    os.makedirs(default_dir, exist_ok=True)

    if save_path is None:
        save_path = os.path.join(default_dir, "weights.pth")

    save_path = get_unique_path(save_path)
    model.save(save_path)
    print(f"\n✅ Saved model in: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train UNET model from CSV splits with explicit labels.")

    default_split = os.getenv(
        'SPLIT', 'Ophthalmic_Scans/splits/tumor_and_fluid_segmentation_oct')
    default_epochs = int(os.getenv('EPOCHS', '50'))
    default_batch = int(os.getenv('BATCH', '16'))

    parser.add_argument(
        "--train_csv",
        type=str,
        default=os.path.join(default_split, "train.csv"),
        help="Path to train.csv",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default=os.path.join(default_split, "val.csv"),
        help="Path to val.csv",
    )
    parser.add_argument(
        "--unet_continue_last_run",
        action="store_true",
        default=bool(os.getenv('UNET_CONTINUE_LAST_RUN', False)),
        help="Continue training from last run checkpoint"
    )
    parser.add_argument("--save_path", type=str, default=None,
                        help="Optional model save path")
    parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=default_batch)
    parser.add_argument("--seed", type=int, default=int(os.getenv('SEED', '42')))

    args = parser.parse_args()
    main(**vars(args))
