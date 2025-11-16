import argparse
import os
import pandas as pd
from ultralytics import YOLO


def make_yolo_split(csv_path, split_name):
    """
    Creates dataset structure in YOLO-style with simlink,
    so that YOLO can see image <-> label pairs.
    """
    df = pd.read_csv(csv_path)
    img_dir = os.path.join('yolo_dataset', 'images', split_name)
    lbl_dir = os.path.join('yolo_dataset', 'labels', split_name)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for i, row in df.iterrows():
        img_src = os.path.join('Ophthalmic_Scans', row["image_path"])
        lbl_src = os.path.join('Ophthalmic_Scans', row["label_path"])
        img_name = os.path.basename(img_src)
        img_extension = os.path.splitext(img_name)[1]
        img_dst = os.path.join(img_dir, f'{i}{img_extension}')
        lbl_dst = os.path.join(lbl_dir, f'{i}.txt')

        if os.path.lexists(img_dst):
            os.remove(img_dst)
        os.link(img_src, img_dst)
        if os.path.lexists(lbl_dst):
            os.remove(lbl_dst)
        os.link(lbl_src, lbl_dst)


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

    model = YOLO("yolov8n.pt")
    results = model.train(
        data="train_model/data.yaml",
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

    parser.add_argument(
        "--train_csv",
        type=str,
        default="Ophthalmic_Scans/splits/tumor_and_fluid_segmentation_oct/train.csv",
        help="Ścieżka do train.csv",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default="Ophthalmic_Scans/splits/tumor_and_fluid_segmentation_oct/val.csv",
        help="Ścieżka do val.csv",
    )
    parser.add_argument("--save_path", type=str, default=None, help="Opcjonalna ścieżka zapisu modelu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=16)

    args = parser.parse_args()
    main(**vars(args))
