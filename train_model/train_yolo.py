import argparse
import os
from ultralytics import YOLO
from utils import make_yolo_split, get_unique_path
from dotenv import load_dotenv

load_dotenv(dotenv_path='train_model/.env')

def main(train_csv, val_csv, save_path=None, epochs=50, imgsz=512, batch=16):
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
