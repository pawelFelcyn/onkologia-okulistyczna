import argparse
import os
from ultralytics import YOLO
from utils import make_yolo_split
from dotenv import load_dotenv

load_dotenv(dotenv_path='train_model/.env')

def get_unique_path(base_path):
    """Returns a unique path if the file already exists."""
    if not os.path.exists(base_path):
        return base_path
    root, ext = os.path.splitext(base_path)
    i = 1
    while os.path.exists(f"{root}({i}){ext}"):
        i += 1
    return f"{root}({i}){ext}"


def main(train_csv, val_csv, save_path=None, epochs=50, imgsz=640, batch=16):
    make_yolo_split(train_csv, "train")
    make_yolo_split(val_csv, "val")

    model = YOLO("base_models/yolov8n-seg.pt")
    model.train(
        data="data.yaml",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name="exp_from_csv",
    )

    default_dir = "models"
    os.makedirs(default_dir, exist_ok=True)

    if save_path is None:
        save_path = os.path.join(default_dir, "weights.pt")

    save_path = get_unique_path(save_path)
    model.save(save_path)
    print(f"\n✅ Saved model in: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO model from CSV splits with explicit labels.")
    
    default_split = os.getenv('SPLIT', 'Ophthalmic_Scans/splits/tumor_and_fluid_segmentation')
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
    parser.add_argument("--save_path", type=str, default=None, help="Opctional model save path")
    parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=default_batch)

    args = parser.parse_args()
    main(**vars(args))
