import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def _get_image_metadata(image_path: str, label_path: str):
    filename = os.path.basename(image_path)
    filename = filename.split('.')[0]
    parts = filename.split('_')
    patient_id = parts[4]
    date = parts[5]
    area = parts[3]
    with open(label_path, 'r') as f:
        lines = f.readlines()
        has_tumor = any(x for x in lines if x[0] == '1')
        has_fluid = any(x for x in lines if x[0] == '0')
        
    return {
        'patient_id': patient_id,
        'date': date,
        'area': area,
        'has_tumor': has_tumor,
        'has_fluid': has_fluid,
        'filename': filename
    }

def _print_dataset_stats(df: pd.DataFrame, print_header = True):
    if print_header:
        print("Stats before grouping:")
    print("Images with tumor only:", df[df["has_tumor"] & ~df["has_fluid"]].shape[0])
    print("Images with fluid only:", df[~df["has_tumor"] & df["has_fluid"]].shape[0])
    print("Images without tumor or fluid:", df[~df["has_tumor"] & ~df["has_fluid"]].shape[0])
    print("Images with tumor and fluid:", df[df["has_tumor"] & df["has_fluid"]].shape[0])  

def _group_data(df: pd.DataFrame) -> pd.DataFrame:
    df["group_id"] = df["patient_id"].astype(str) + "_" + df["date"].astype(str)
    df = df.groupby("group_id").agg(
        n_images=("filename", "count"),
        has_tumor=("has_tumor", "max"),
        has_fluid=("has_fluid", "max")
    ).reset_index()
    return df

def _print_group_stats(df: pd.DataFrame):
    print("\n--- Groupped data stats ---")
    print("Number of unique groups (patient-date is a key):", df.shape[0])
    print("Groups with tumor only:" , ((df["has_tumor"] == 1) & (df["has_fluid"] == 0)).sum())
    print("Groups with fluid only:", ((df["has_tumor"] == 0) & (df["has_fluid"] == 1)).sum())
    print("Groups without fluid or tumor:", ((df["has_tumor"] == 0) & (df["has_fluid"] == 0)).sum())
    print("Groups with fluid and tumor:", ((df["has_tumor"] == 1) & (df["has_fluid"] == 1)).sum())
    
def _assign_category(row):
    if row["has_tumor"] and row["has_fluid"]:
        return "tumor+fluid"
    elif row["has_tumor"]:
        return "tumor"
    elif row["has_fluid"]:
        return "fluid"
    else:
        return "healthy"

def _stratified_group_split(group_summary, splits, random_state=42):
    np.random.seed(random_state)
    result = {k: [] for k in splits.keys()}

    for category in group_summary["category"].unique():
        sub = group_summary[group_summary["category"] == category]
        total_images = sub["n_images"].sum()

        shuffled = sub.sample(frac=1, random_state=random_state)
        targets = {k: v * total_images for k, v in splits.items()}
        counters = {k: 0 for k in splits.keys()}

        for _, row in shuffled.iterrows():
            remaining = {k: targets[k] - counters[k] for k in splits.keys()}
            split_choice = max(remaining, key=remaining.get)

            result[split_choice].append(row["group_id"])
            counters[split_choice] += row["n_images"]

    return result

def _print_split_stats(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame):
    print("\n--- Split stats ---")
    print("Train:")
    _print_dataset_stats(df_train, print_header=False)
    print("Validation:")
    _print_dataset_stats(df_val, print_header=False)
    print("Test:")
    _print_dataset_stats(df_test, print_header=False)

def _save_splits_metadata(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "splits", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "splits", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "splits", "test"), exist_ok=True)
    df_train.to_csv(os.path.join(output_dir, "splits", "train", "train.tsv"), index=False, sep="\t")
    df_val.to_csv(os.path.join(output_dir, "splits", "val", "val.tsv"), index=False, sep="\t")
    df_test.to_csv(os.path.join(output_dir, "splits", "test", "test.tsv"), index=False, sep="\t")
    
def _move_images_and_masks(source_root: str, destination: str, df_train: pd.DataFrame,
                           df_val: pd.DataFrame, df_test: pd.DataFrame):
    images_dir = os.path.join(source_root, 'images')
    masks_dir = os.path.join(source_root, 'masks')
    for image in tqdm(os.listdir(images_dir), desc="Copying images", unit="image"):
        if not image.endswith('.jpg') and not image.endswith('.png') and not image.endswith('.jpeg'):
            continue
        without_ext = os.path.splitext(image)[0]
        ext = os.path.splitext(image)[1]
        target_split = None
        proper_df = None
        if df_train[df_train["filename"] == without_ext].shape[0] > 0:
            target_split = 'train'
            proper_df = df_train
        elif df_val[df_val["filename"] == without_ext].shape[0] > 0:
            target_split = 'val'
            proper_df = df_val
        elif df_test[df_test["filename"] == without_ext].shape[0] > 0:
            target_split = 'test'
            proper_df = df_test

        id = proper_df[proper_df['filename'] == without_ext].iloc[0]['id']
        image_target_path = os.path.join(destination, 'splits', target_split, 'images', f'{id}{ext}')
        mask_target_path = os.path.join(destination, 'splits', target_split, 'masks', f'{id}{ext}')
        os.makedirs(os.path.dirname(image_target_path), exist_ok=True)
        os.makedirs(os.path.dirname(mask_target_path), exist_ok=True)
        mask_source_path = os.path.join(masks_dir, f'{without_ext}{ext}')
        image_source_path = os.path.join(images_dir, image)
        os.system(f'copy "{image_source_path}" "{image_target_path}"')
        os.system(f'copy "{mask_source_path}" "{mask_target_path}"')

def make_dataset(source_root: str, destination: str):
    """
    Creates a structured dataset split into training, validation, and test subsets,
    with associated metadata and copied images/masks.

    This function processes a dataset organized with three subdirectories:
    - ``images``: containing the raw image files
    - ``labels``: containing YOLO-format label files (one per image)
    - ``masks``: containing pre-generated mask files (one per image)

    The function extracts metadata from the file names and labels, groups images
    by patient and date, assigns category labels (tumor, fluid, tumor+fluid, healthy),
    and performs a stratified split into train/validation/test subsets. Each split
    is saved along with its metadata, and the corresponding images and masks are copied
    into the appropriate output directories. An incremental ``id`` column is added
    to each split, and the training set additionally receives an ``is_augment_output``
    column set to ``False``.

    Args:
        source_root (str):
            Path to the root directory containing the input subdirectories
            ``images``, ``labels``, and ``masks``.
        destination (str):
            Path to the output directory where the processed dataset will be stored.
            The function will create the necessary subdirectories and CSV metadata files.

    Raises:
        Exception: If the required ``images``, ``labels``, or ``masks`` directories
        do not exist inside ``source_root``.

    Side Effects:
        - Prints dataset statistics before and after grouping, as well as for each split.
        - Creates directories and writes CSV files with split metadata.
        - Copies image and mask files into the appropriate split folders
          (train/val/test).

    Output Structure (under ``destination``):
        splits/
            train/
                images/   (renamed image files with incremental IDs)
                masks/    (corresponding masks)
                train.csv (metadata for the training split)
            val/
                images/
                masks/
                val.csv
            test/
                images/
                masks/
                test.csv
    """
    images_dir = os.path.join(source_root, 'images')
    labels_dir = os.path.join(source_root, 'labels')
    masks_dir = os.path.join(source_root, 'masks')
    
    if not os.path.isdir(images_dir):
        raise Exception("Images directory must exist")
    if not os.path.isdir(labels_dir):
        raise Exception("Labels directory must exist")
    if not os.path.isdir(masks_dir):
        raise Exception("Masks directory must exist")
    
    images = os.listdir(images_dir)
    rows = []
    for image in images:
        image_path = os.path.join(images_dir, image)
        label_path = os.path.join(labels_dir, image.replace('.png', '.txt'))
        rows.append(_get_image_metadata(image_path, label_path))
    df = pd.DataFrame(rows)
    _print_dataset_stats(df)
    df = _group_data(df)
    _print_group_stats(df)
    df["category"] = df.apply(_assign_category, axis=1)
    splits = {"train": 0.7, "val": 0.15, "test": 0.15}
    split_result = _stratified_group_split(df, splits)
    df_train = df[df["group_id"].isin(split_result["train"])].reset_index(drop=True)
    df_val   = df[df["group_id"].isin(split_result["val"])].reset_index(drop=True)
    df_test  = df[df["group_id"].isin(split_result["test"])].reset_index(drop=True)
    df_train.insert(0, "id", df_train.index)
    df_val.insert(0, "id", df_val.index)
    df_test.insert(0, "id", df_test.index)
    df_train["is_augment_output"] = False
    _print_split_stats(df_train, df_val, df_test)
    os.makedirs(destination, exist_ok=True)
    _save_splits_metadata(df_train, df_val, df_test, destination)
    _move_images_and_masks(source_root, destination, df_train, df_val, df_test)