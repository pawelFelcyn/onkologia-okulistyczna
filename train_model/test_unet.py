import unet_utils
import torch
import os
import csv
from torch.utils.data import DataLoader
import argparse
from dotenv import load_dotenv
import re
import uuid
from pathlib import Path
from datetime import datetime, timezone

load_dotenv(dotenv_path='train_model/.env')


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip()).strip("-") or "unknown"


def make_eval_id() -> str:
    return uuid.uuid4().hex[:8]


def make_eval_run_name(split: str, model_to_test: str, imgsz: int, batch: int, eval_id: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    split_tag = _slug(Path(split).name)
    model_tag = _slug(Path(model_to_test).stem)
    return f"unet_eval__{stamp}__id{eval_id}__model{model_tag}__split{split_tag}__img{imgsz}__bs{batch}"


def extract_seed(model_path: str) -> str:
    match = re.search(r"(?:^|[^a-zA-Z])seed[_-]?(\d+)", Path(model_path).stem, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def flatten_metrics(prefix: str, metrics: dict, cm) -> dict:
    row = {f"{prefix}_{key}": float(value) for key, value in metrics.items()}
    row.update({
        f"{prefix}_tp": int(cm[1][1]),
        f"{prefix}_tn": int(cm[0][0]),
        f"{prefix}_fp": int(cm[0][1]),
        f"{prefix}_fn": int(cm[1][0]),
    })
    return row


def append_metrics_csv(csv_path: str, row: dict) -> None:
    output_path = Path(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def resolve_models(model_to_test: str, models_dir: str | None, model_glob: str) -> list[str]:
    if models_dir:
        return [str(path) for path in sorted(Path(models_dir).glob(model_glob)) if path.is_file()]
    return [model_to_test]


def evaluate_one_model(split: str, model_to_test: str, batch: int, imgsz: int, metrics_csv: str | None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    root_dir = os.path.join("Ophthalmic_Scans")
    test_csv = os.path.join(split, 'test.csv')
    test_dataset = unet_utils.UNetDataset(test_csv, root_dir, imgsz=imgsz)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

    model = unet_utils.UNet(3, 2)
    model.load_state_dict(torch.load(model_to_test, map_location=device))
    eval_id = make_eval_id()
    run_name = make_eval_run_name(
        split=split,
        model_to_test=model_to_test,
        imgsz=imgsz,
        batch=batch,
        eval_id=eval_id,
    )
    run_dir = os.path.join('runs_unet', run_name)
    print(f"Saving evaluation results to: {run_dir}")
    fluid_metrics, fluid_cm, tumor_metrics, tumor_cm = model.test_model(test_loader, device=device, run_name=run_name)

    if metrics_csv:
        fluid_dice = float(fluid_metrics["dice"])
        tumor_dice = float(tumor_metrics["dice"])
        row = {
            "eval_id": eval_id,
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_path": model_to_test,
            "model_name": Path(model_to_test).name,
            "seed": extract_seed(model_to_test),
            "split": split,
            "test_csv": test_csv,
            "imgsz": imgsz,
            "batch": batch,
            "run_name": run_name,
            "run_dir": run_dir,
            "dice_macro": (fluid_dice + tumor_dice) / 2.0,
            "iou_macro": (float(fluid_metrics["iou"]) + float(tumor_metrics["iou"])) / 2.0,
            "f1_macro": (float(fluid_metrics["f1"]) + float(tumor_metrics["f1"])) / 2.0,
        }
        row.update(flatten_metrics("fluid", fluid_metrics, fluid_cm))
        row.update(flatten_metrics("tumor", tumor_metrics, tumor_cm))
        append_metrics_csv(metrics_csv, row)
        print(f"Saved evaluation summary row to: {metrics_csv}")

    print(
        "Summary | "
        f"fluid Dice: {float(fluid_metrics['dice']):.4f} | "
        f"tumor Dice: {float(tumor_metrics['dice']):.4f} | "
        f"macro Dice: {(float(fluid_metrics['dice']) + float(tumor_metrics['dice'])) / 2.0:.4f}"
    )

    print("\nModel evaluation complete.")


def main(split: str, model_to_test: str, batch: int, imgsz: int,
         metrics_csv: str | None = None, models_dir: str | None = None,
         model_glob: str = "*.pth") -> None:
    model_paths = resolve_models(model_to_test=model_to_test, models_dir=models_dir, model_glob=model_glob)
    if not model_paths:
        raise FileNotFoundError(f"No models found in {models_dir!r} matching {model_glob!r}")

    for model_path in model_paths:
        evaluate_one_model(
            split=split,
            model_to_test=model_path,
            batch=batch,
            imgsz=imgsz,
            metrics_csv=metrics_csv,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained UNet model on the test split.")

    default_split = os.getenv('SPLIT', 'Ophthalmic_Scans/splits/tumor_and_fluid_segmentation_oct')
    default_batch = int(os.getenv('BATCH', '16'))

    parser.add_argument(
        "--split",
        type=str,
        default=default_split,
        help="Directory containing test.csv",
    )
    parser.add_argument(
        "--model_to_test",
        type=str,
        default="models/unet/weights.pth",
        help="Path to the .pth checkpoint to evaluate",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default=None,
        help="Optional directory with .pth checkpoints to evaluate in one run",
    )
    parser.add_argument(
        "--model_glob",
        type=str,
        default="*.pth",
        help="Glob used with --models_dir, e.g. 'kermany_transfer_seed*.pth'",
    )
    parser.add_argument(
        "--metrics_csv",
        type=str,
        default=None,
        help="Optional CSV path where one summary row per evaluated model will be appended",
    )
    parser.add_argument("--batch", type=int, default=default_batch)
    parser.add_argument("--imgsz", type=int, default=512,
                        help="Resize images to this size before inference (must match training imgsz)")

    args = parser.parse_args()
    main(**vars(args))
