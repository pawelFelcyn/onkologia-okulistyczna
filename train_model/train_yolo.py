import argparse
import os
from ultralytics import YOLO
from utils import make_yolo_split, get_unique_path
from dotenv import load_dotenv

load_dotenv(dotenv_path='train_model/.env')

def get_model_path(should_continue: bool):
    if not should_continue:
        return "base_models/yolov8n-seg.pt"
    runs_dir = os.path.join("runs", "segment")
    all_runs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
    latest_run = max(all_runs, key=lambda d: os.path.getmtime(os.path.join(runs_dir, d)))
    return os.path.join(runs_dir, latest_run, "weights", "last.pt")

def main(train_csv, val_csv, save_path=None, epochs=50, imgsz=512, batch=16, continue_from_epoch=None):
    make_yolo_split(train_csv, "train")
    make_yolo_split(val_csv, "val")
    print(f"Last run will be used to continue training from epoch {continue_from_epoch}." if default_continue_from_epoch else "Training will start from base model.")
    actual_epochs = epochs if continue_from_epoch is None else epochs - (continue_from_epoch - 1)
    print(f"Training for {actual_epochs} epochs left.")
    model_name = YOLO(get_model_path(continue_from_epoch is not None))
    print("Actual base model for training:", model_name)
    model = YOLO(model_name)
    model.train(
        data="data.yaml",
        epochs=actual_epochs,
        imgsz=imgsz,
        batch=batch,
        name="exp_from_csv",
    )

    default_dir = "models/yolo"
    os.makedirs(default_dir, exist_ok=True)

    if save_path is None:
        save_path = os.path.join(default_dir, "weights.pt")

    save_path = get_unique_path(save_path)
    model.save(save_path)
    print(f"\n✅ Saved model in: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO model from CSV splits with explicit labels.")
    
    default_split = os.getenv('SPLIT', 'Ophthalmic_Scans/splits/tumor_and_fluid_segmentation_oct')
    default_epochs = int(os.getenv('EPOCHS', '50'))
    default_batch = int(os.getenv('BATCH', '16'))
    default_continue_from_epoch = os.getenv('CONTINUE_FROM_EPOCH', None)
    parser.add_argument(
        "--continue-from-epoch",
        type=int,
        default=default_continue_from_epoch,
        help="Epoch to continue training from (if applicable)"
    )
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
    parser.add_argument("--save_path", type=str, default=None, help="Opctional model save path")
    parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=default_batch)

    args = parser.parse_args()
    main(**vars(args))
