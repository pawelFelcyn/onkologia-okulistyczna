import unet_utils
import torch
import os
from torch.utils.data import DataLoader
import argparse
from dotenv import load_dotenv
import re
from pathlib import Path
from datetime import datetime, timezone

load_dotenv(dotenv_path='train_model/.env')


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip()).strip("-") or "unknown"


def make_eval_run_name(split: str, model_to_test: str, imgsz: int, batch: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    split_tag = _slug(Path(split).name)
    model_tag = _slug(Path(model_to_test).stem)
    return f"unet_eval__model{model_tag}__split{split_tag}__img{imgsz}__bs{batch}__{stamp}"


def main(split: str, model_to_test: str, batch: int, imgsz: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    root_dir = os.path.join("Ophthalmic_Scans")
    test_csv = os.path.join(split, 'test.csv')
    test_dataset = unet_utils.UNetDataset(test_csv, root_dir, imgsz=imgsz)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

    model = unet_utils.UNet(3, 2)
    model.load_state_dict(torch.load(model_to_test, map_location=device))
    run_name = make_eval_run_name(split=split, model_to_test=model_to_test, imgsz=imgsz, batch=batch)
    print(f"Saving evaluation results to: {os.path.join('runs_unet', run_name)}")
    model.test_model(test_loader, device=device, run_name=run_name)

    print("\nModel evaluation complete.")


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
    parser.add_argument("--batch", type=int, default=default_batch)
    parser.add_argument("--imgsz", type=int, default=512,
                        help="Resize images to this size before inference (must match training imgsz)")

    args = parser.parse_args()
    main(**vars(args))