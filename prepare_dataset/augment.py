import pandas as pd
from tqdm import tqdm
import random
import os
import cv2
import albumentations as A

COMBOS = ['nothing', 'fluid_only', 'tumor_only', 'both']
TARGET_PER_COMBO = 1250

def _normalize_bool(v):
    if pd.isna(v):
        return False
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    return s in ('1', 'true', 't', 'yes', 'y')

def _combo_from_row(row):
    t = _normalize_bool(row['has_tumor'])
    f = _normalize_bool(row['has_fluid'])
    if t and f:
        return 'both'
    if t and not f:
        return 'tumor_only'
    if not t and f:
        return 'fluid_only'
    return 'nothing'

def _make_augmentation_plan(train_data_dir: str):
    print("Making augmentation plan...")
    rng = random.Random(42)
    df = pd.read_csv(os.path.join(train_data_dir, 'train.csv'), sep='\t')
    df['combo'] = df.apply(_combo_from_row, axis=1)
    groups = {}
    for area in ['fovea', 'lesion']:
        for combo in COMBOS:
            key = (area, combo)
            ids = df[(df['area'] == area) & (df['combo'] == combo)]['id'].tolist()
            groups[key] = ids

    plan = {}
    summary = {}

    for key, ids in groups.items():
        area, combo = key
        n = len(ids)
        target = TARGET_PER_COMBO
        summary[key] = {'n': n, 'target': target}
        if n == 0:
            # can't augment from nothing
            for _id in []:
                plan[_id] = 0
            print(f"WARNING: no images for {area}-{combo} (n=0).")
            continue
        if n <= target:
            extra = target - n
            base = extra // n
            rem = extra % n
            ids_shuffled = ids.copy()
            rng.shuffle(ids_shuffled)
            for i, _id in enumerate(ids_shuffled):
                aug = base + (1 if i < rem else 0)
                plan[_id] = aug
        else:
            ids_shuffled = ids.copy()
            rng.shuffle(ids_shuffled)
            keep = set(ids_shuffled[:target])
            for _id in ids:
                if _id in keep:
                    plan[_id] = 0
                else:
                    plan[_id] = -1 
    
    for _id in df['id'].tolist():
        if _id not in plan:
            plan[_id] = 0

    out_df = pd.DataFrame({'id': df['id'].tolist(),
                           'augmentations': [int(plan[_id]) for _id in df['id'].tolist()]})
    return out_df, summary

def _make_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.7)
    ])

def augment_training_data(train_data_dir: str):
    """
    Perform data augmentation on training images and masks.

    This function generates an augmentation plan based on the class balance
    (tumor/fluid/nothing/both) and applies transformations such as flips,
    rotations, and scaling to produce a balanced dataset. The augmented images
    and masks are saved back into the `images` and `masks` directories inside
    the given training data directory. A new `train.tsv` file is written with
    updated metadata including the augmented samples.

    Args:
        train_data_dir (str): Path to the training dataset directory. The
            directory must contain:
            - `train.csv` (tab-separated metadata with at least `id`, `area`,
              `has_tumor`, `has_fluid` columns).
            - `images/` (directory with input images).
            - `masks/` (directory with corresponding segmentation masks).

    Side Effects:
        - Writes augmented images into `images/`.
        - Writes augmented masks into `masks/`.
        - Creates/overwrites `train.tsv` with updated dataset metadata.

    Notes:
        - Images with augmentation count `-1` are excluded from the output
          (downsampling).
        - Each image is assigned enough augmentations to reach the target number
          of samples per class/area combination.
        - Uses `albumentations` for transformations and `cv2` for I/O.

    Example:
        >>> augment_training_data("/path/to/train_data")

        This will balance the dataset and save augmented files plus a new
        `train.tsv`.
    """
    plan = _make_augmentation_plan(train_data_dir)
    train_df = pd.read_csv(os.path.join(train_data_dir, 'train.csv'), sep='\t')
    merged = pd.merge(train_df, plan, on='id', how='inner')
    images_dir = os.path.join(train_data_dir, 'images')
    masks_dir = os.path.join(train_data_dir, 'masks')
    transform = _make_transform()
    augmented_rows = []
    
    for _, row in  tqdm(merged.iterrows(), desc="Augmentation", unit="image"):
        aug_count = row['augmentations']
        if aug_count == -1:
            continue 
        src_id = str(row['id'])
        image_path = next((os.path.join(root, f) for root, _, files in os.walk(images_dir) for f in files if f.startswith(src_id)), None)
        ext = image_path.split('.')[-1]
        mask_path = next((os.path.join(root, f) for root, _, files in os.walk(masks_dir) for f in files if f.startswith(src_id)), None)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        out_path = os.path.join(images_dir, f"{src_id}.{ext}")
        out_mask_path = os.path.join(masks_dir, f"{src_id}.{ext}")
        
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(out_mask_path, mask)
        row_dict = row.to_dict()
        del row_dict['augmentations']
        augmented_rows.append(row_dict)

        if aug_count > 0:
            for _ in range(aug_count):
                transformed = transform(image=img, mask=mask)
                aug_img = transformed['image']
                aug_mask = transformed['mask']
                new_filename = f"{id}.{ext}"
                out_path = os.path.join(images_dir, new_filename)
                out_mask_path = os.path.join(masks_dir, new_filename)
                cv2.imwrite(out_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(out_mask_path, aug_mask)
                new_row = row.to_dict()
                new_row['id'] = id
                new_row['filename'] = row['filename']
                new_row['patient_id'] = row['patient_id']
                new_row['date'] = row['date']
                new_row['area'] = row['area']
                new_row['has_tumor'] = row['has_tumor']
                new_row['has_fluid'] = row['has_fluid']
                new_row['is_augment_output'] = True
                augmented_rows.append(new_row)
                id += 1

    out_df = pd.DataFrame(augmented_rows)
    out_df['id'] = out_df['id'].astype(int)
    out_df.sort_values('id', inplace=True)
    out_df.to_csv(os.path.join(train_data_dir, 'train.tsv'), sep='\t', index=False)