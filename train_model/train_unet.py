import unet_utils
import os
import random
import numpy as np
import csv
import json
import subprocess
from torch.utils.data import DataLoader
from utils import get_unique_path
from dotenv import load_dotenv
import argparse
import torch
import re
from pathlib import Path
from datetime import datetime, timezone


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

load_dotenv(dotenv_path='train_model/.env')


def _extract_completed_epochs(run_dir: Path) -> list[int]:
    return [
        int(m.group(1))
        for p in run_dir.iterdir()
        if p.is_dir() and any(p.iterdir()) and (m := re.fullmatch(r"epoch_(\d+)", p.name))
    ]


def _checkpoint_and_epoch_for_run(run_dir: Path) -> tuple[str, int] | None:
    epochs_done = _extract_completed_epochs(run_dir)
    if not epochs_done:
        return None
    weights = run_dir / "weights" / "last.pth"
    if not weights.exists():
        return None
    max_epoch = max(epochs_done)
    return str(weights), max_epoch


def get_last_run_model() -> tuple[str, int, str]:
    """Find the latest resumable checkpoint in runs_unet based on mtime.

    Iterates over run directories from newest to oldest and returns
    the path to last.pth and the highest completed epoch number for the
    first run that has both a completed epoch directory and a checkpoint.

    Returns:
        (weights_path, trained_epochs, run_name): checkpoint path, epoch count and run folder name.

    Raises:
        FileNotFoundError: if no resumable run exists.
    """
    unet_runs = Path('runs_unet')
    if not unet_runs.exists() or not unet_runs.is_dir():
        raise FileNotFoundError("No runs_unet directory found — nothing to resume.")

    runs = sorted(
        [p for p in unet_runs.iterdir() if p.is_dir() and not p.name.startswith(("test_run", "unet_eval"))],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError("No runs_unet/* directories found — nothing to resume.")
    for run_dir in runs:
        result = _checkpoint_and_epoch_for_run(run_dir)
        if result is not None:
            weights, max_epoch = result
            print(f"Resuming from: {run_dir.name}, epoch {max_epoch}")
            return weights, max_epoch, run_dir.name
    raise FileNotFoundError(
        "Found run* directories but none contain a completed epoch and a last.pth checkpoint."
    )


def get_run_model_by_name(run_name: str) -> tuple[str, int, str]:
    run_dir = Path('runs_unet') / run_name
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    result = _checkpoint_and_epoch_for_run(run_dir)
    if result is None:
        raise FileNotFoundError(
            f"Run {run_name} does not contain a completed epoch and weights/last.pth"
        )
    weights, max_epoch = result
    print(f"Resuming from explicit run: {run_name}, epoch {max_epoch}")
    return weights, max_epoch, run_name


def infer_approach(encoder_weights: str | None, freeze_encoder: bool) -> str:
    if encoder_weights and freeze_encoder:
        return "transfer_freeze"
    if encoder_weights:
        return "transfer"
    return "baseline"


def make_run_name(approach: str, seed: int, imgsz: int, batch: int, started_at: datetime) -> str:
    safe_approach = re.sub(r"[^a-zA-Z0-9_-]+", "-", approach.strip()).strip("-") or "unknown"
    stamp = started_at.strftime("%Y%m%d-%H%M%S")
    return f"unet_{safe_approach}__seed{seed}__img{imgsz}__bs{batch}__{stamp}"


def get_git_commit_short() -> str:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
        return commit.strip()
    except Exception:
        return "unknown"


def append_experiment_index(row: dict):
    runs_root = Path("runs_unet")
    runs_root.mkdir(exist_ok=True)
    index_path = runs_root / "experiments_index.csv"

    fieldnames = [
        "run_name", "approach", "seed", "imgsz", "batch", "epochs_requested", "epochs_trained",
        "freeze_encoder", "encoder_weights", "save_path", "best_val_dice", "best_tumor_dice",
        "best_fluid_dice", "started_at", "finished_at", "resume_mode", "resume_from", "git_commit", "slurm_job_id"
    ]
    write_header = not index_path.exists()
    with open(index_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def main(train_csv, val_csv, save_path=None, epochs=50, imgsz=512, batch=16,
         unet_continue_last_run=False, seed=42, encoder_weights=None,
        freeze_encoder=False, run_name=None, resume_from=None,
        resume_run_name=None, approach=None):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | seed: {seed}")

    started_at = datetime.now(timezone.utc)
    approach = approach or infer_approach(encoder_weights, freeze_encoder)
    run_name = run_name or make_run_name(approach, seed, imgsz, batch, started_at)

    root_dir = os.path.join("Ophthalmic_Scans")
    train_dataset = unet_utils.UNetDataset(train_csv, root_dir, imgsz=imgsz)
    val_dataset = unet_utils.UNetDataset(val_csv, root_dir, imgsz=imgsz)

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset,   batch_size=batch, shuffle=False)
    model = unet_utils.UNet(3, 2)

    if encoder_weights:
        state = torch.load(encoder_weights, map_location="cpu", weights_only=True)
        missing, _unexpected = model.load_state_dict(state, strict=False)
        print(f"[INFO] Loaded encoder weights from: {encoder_weights}")
        if missing:
            print(f"[INFO] Decoder/output layers will train from scratch: {missing}")

    if freeze_encoder:
        encoder_blocks = [model.conv1, model.conv2, model.conv3, model.conv4, model.conv5]
        for block in encoder_blocks:
            for param in block.parameters():
                param.requires_grad = False
        print("[INFO] Encoder blocks conv1-conv5 are frozen.")

    trained_epochs = 0
    resume_mode = "none"
    resume_source = ""

    if resume_from:
        resume_mode = "resume_from"
        resume_source = resume_from
        print(f"Resuming from explicit checkpoint path: {resume_from}")
        model.load_state_dict(torch.load(resume_from, map_location=device))
    elif resume_run_name:
        resume_mode = "resume_run_name"
        last_weights, trained_epochs, resumed_run_name = get_run_model_by_name(resume_run_name)
        resume_source = f"{resumed_run_name}:{last_weights}"
        print(f"Resuming from epoch {trained_epochs + 1} (checkpoint: {last_weights})")
        model.load_state_dict(torch.load(last_weights, map_location=device))
        epochs = epochs - trained_epochs
    elif unet_continue_last_run:
        resume_mode = "continue_last_run"
        print("Resuming training from latest checkpoint...")
        last_weights, trained_epochs, resumed_run_name = get_last_run_model()
        resume_source = f"{resumed_run_name}:{last_weights}"
        print(f"Resuming from epoch {trained_epochs + 1} (checkpoint: {last_weights})")
        model.load_state_dict(torch.load(last_weights, map_location=device))
        epochs = epochs - trained_epochs

    if epochs <= 0:
        print("No remaining epochs to train — target already reached.")
        return

    run_meta = {
        "run_name": run_name,
        "approach": approach,
        "seed": seed,
        "train_csv": train_csv,
        "val_csv": val_csv,
        "epochs_requested": epochs + trained_epochs,
        "epochs_trained": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "freeze_encoder": freeze_encoder,
        "encoder_weights": encoder_weights,
        "resume_mode": resume_mode,
        "resume_source": resume_source,
        "save_path_requested": save_path,
        "started_at": started_at.isoformat(),
        "git_commit": get_git_commit_short(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID", ""),
    }

    train_result = model.train_model(
        train_loader,
        val_loader,
        epochs,
        device=device,
        freeze_encoder=freeze_encoder,
        run_name=run_name,
        run_meta=run_meta,
    )
    weights_dir = train_result["weights_dir"]
    run_dir = train_result["run_dir"]

    default_dir = "models/unet"
    os.makedirs(default_dir, exist_ok=True)

    if save_path is None:
        save_path = os.path.join(default_dir, "weights.pth")

    save_path = get_unique_path(save_path)

    best_ckpt = os.path.join(weights_dir, "best.pth")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    print(f"Loaded best checkpoint from: {best_ckpt}")

    model.save(save_path)
    print(f"\n✅ Saved best model in: {save_path}")

    finished_at = datetime.now(timezone.utc)
    run_summary = {
        **run_meta,
        "run_dir": run_dir,
        "weights_dir": weights_dir,
        "best_checkpoint": best_ckpt,
        "save_path_final": save_path,
        "best_val_dice": train_result["best_val_dice"],
        "best_tumor_dice": train_result["best_tumor_dice"],
        "best_fluid_dice": train_result["best_fluid_dice"],
        "finished_at": finished_at.isoformat(),
    }
    with open(os.path.join(run_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    append_experiment_index({
        "run_name": run_name,
        "approach": approach,
        "seed": seed,
        "imgsz": imgsz,
        "batch": batch,
        "epochs_requested": run_meta["epochs_requested"],
        "epochs_trained": run_meta["epochs_trained"],
        "freeze_encoder": freeze_encoder,
        "encoder_weights": encoder_weights or "",
        "save_path": save_path,
        "best_val_dice": train_result["best_val_dice"],
        "best_tumor_dice": train_result["best_tumor_dice"],
        "best_fluid_dice": train_result["best_fluid_dice"],
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "resume_mode": resume_mode,
        "resume_from": resume_source,
        "git_commit": run_meta["git_commit"],
        "slurm_job_id": run_meta["slurm_job_id"],
    })


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
    parser.add_argument("--approach", type=str, default=None,
                        help="Optional run label, e.g. baseline|transfer|transfer_freeze")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Optional explicit run directory name under runs_unet/")
    parser.add_argument("--resume_run_name", type=str, default=None,
                        help="Resume from a specific run directory under runs_unet/")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from an explicit checkpoint path to last.pth")
    parser.add_argument("--encoder_weights", type=str, default=None,
                        help="Path to pretrained encoder checkpoint (e.g. from train_kermany.py)")
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Freeze encoder blocks conv1-conv5 during segmentation training"
    )

    args = parser.parse_args()
    main(**vars(args))
