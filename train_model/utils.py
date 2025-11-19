import pandas as pd
import os

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