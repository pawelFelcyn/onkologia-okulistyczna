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


def get_last_run_model() -> tuple[str, int]:
    """Find the latest resumable checkpoint in runs_unet/.

    Iterates over run<N> directories from newest to oldest and returns
    the path to last.pth and the highest completed epoch number for the
    first run that has both a completed epoch directory and a checkpoint.

    Returns:
        (weights_path, trained_epochs): path to last.pth and epoch count.

    Raises:
        FileNotFoundError: if no resumable run exists.
    """
    unet_runs = Path('runs_unet')
    # Find all run<N> directories sorted in descending order (newest first)
    runs = sorted(
        [(int(m.group(1)), p) for p in unet_runs.iterdir()
         if p.is_dir() and (m := re.fullmatch(r"run(\d+)", p.name))],
        reverse=True
    )
    if not runs:
        raise FileNotFoundError("No runs_unet/run* directories found — nothing to resume.")
    for _, run_dir in runs:
        epochs_done = [
            int(m.group(1))
            for p in run_dir.iterdir()
            if p.is_dir() and any(p.iterdir()) and (m := re.fullmatch(r"epoch_(\d+)", p.name))
        ]
        if epochs_done:
            max_epoch = max(epochs_done)
            weights = run_dir / "weights" / "last.pth"
            if weights.exists():
                print(f"Resuming from: {run_dir.name}, epoch {max_epoch}")
                return str(weights), max_epoch
    raise FileNotFoundError(
        "Found run* directories but none contain a completed epoch and a last.pth checkpoint."
    )


def main(train_csv, val_csv, save_path=None, epochs=50, imgsz=512, batch=16,
         unet_continue_last_run=False, seed=42, encoder_weights=None,
         freeze_encoder=False):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | seed: {seed}")

    root_dir = os.path.join("Ophthalmic_Scans")
    train_dataset = unet_utils.UNetDataset(train_csv, root_dir, imgsz=imgsz)
    val_dataset = unet_utils.UNetDataset(val_csv, root_dir, imgsz=imgsz)

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset,   batch_size=batch, shuffle=False)
    model = unet_utils.UNet(3, 2)

    if encoder_weights:
        state = torch.load(encoder_weights, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[INFO] Loaded encoder weights from: {encoder_weights}")
        if missing:
            print(f"[INFO] Decoder/output layers will train from scratch: {missing}")

    if freeze_encoder:
        encoder_blocks = [model.conv1, model.conv2, model.conv3, model.conv4, model.conv5]
        for block in encoder_blocks:
            for param in block.parameters():
                param.requires_grad = False
        print("[INFO] Encoder blocks conv1-conv5 are frozen.")

    if unet_continue_last_run:
        print("Resuming training from last checkpoint...")
        last_weights, trained_epochs = get_last_run_model()
        print(f"Resuming from epoch {trained_epochs + 1} (checkpoint: {last_weights})")
        model.load_state_dict(torch.load(last_weights, map_location=device))
        epochs = epochs - trained_epochs

    if epochs <= 0:
        print("No remaining epochs to train — target already reached.")
        return

    model.train_model(train_loader, val_loader, epochs, device=device,
                      freeze_encoder=freeze_encoder)

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
    _env_continue = os.getenv('UNET_CONTINUE_LAST_RUN', 'false').strip().lower()
    parser.add_argument(
        "--unet_continue_last_run",
        action="store_true",
        default=_env_continue == 'true',
        help="Resume training from the latest checkpoint (set UNET_CONTINUE_LAST_RUN=true to enable via .env)"
    )
    parser.add_argument("--save_path", type=str, default=None,
                        help="Optional model save path")
    parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=default_batch)
    parser.add_argument("--seed", type=int, default=int(os.getenv('SEED', '42')))
    parser.add_argument("--encoder_weights", type=str, default=None,
                        help="Path to pretrained encoder checkpoint (e.g. from train_kermany.py)")
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Freeze encoder blocks conv1-conv5 during segmentation training"
    )

    args = parser.parse_args()
    main(**vars(args))
