import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.model_selection import StratifiedGroupKFold
import os
import sys
import cv2
import tqdm
import augment as MyA
from yolo_labels_utils import mask2yolo_separate_inputs

dataset_scripts_dir = os.path.join(os.path.dirname(__file__), "..", "dataset_scripts")
sys.path.append(dataset_scripts_dir)

from utils import get_all_labeled_images

# --- 1. Funkcja do określania klasy obrazu ---
def has_content(mask_path):
    if mask_path is None:
        return False
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False
    
    return np.any(mask > 0)

def get_class(tumor_mask_path, fluid_mask_path):
    has_tumor = has_content(tumor_mask_path)
    has_fluid = has_content(fluid_mask_path)

    if has_tumor and has_fluid:
        return 3  # Tumor + fluid
    elif has_tumor:
        return 1  # Tumor only
    elif has_fluid:
        return 2  # Fluid only
    else:
        return 0  # Nothing

# --- 2. Wczytanie wszystkich obrazów ---
data = get_all_labeled_images()

# --- 3. Grupowanie po (patient_id + date) ---
group_to_items = defaultdict(list)

missing_metadata = 0
for item in data:
    img_path, resized, yolo, tumor_mask, fluid_mask = item
    metadata_path = yolo.replace("labels", "metadata").replace(".txt", ".json")
    if not os.path.isfile(metadata_path):
        missing_metadata += 1
        print(f"Metadata file not found for {yolo}, skipping.")
        continue
    
    with open(metadata_path) as f:
        meta = json.load(f)
    
    group_id = f"{meta['patient_id']}_{meta['date']}"
    cls = get_class(tumor_mask, fluid_mask)
    
    group_to_items[group_id].append((item, cls))
print(f"Missing metadata: {missing_metadata}")

# --- 4. Przypisanie klasy grupie (priorytet: Tumor+Fluid > Tumor > Fluid > Nothing) ---
group_ids = []
group_labels = []

for gid, items in group_to_items.items():
    cls_list = [c for _, c in items]
    if 3 in cls_list:
        group_cls = 3
    elif 1 in cls_list:
        group_cls = 1
    elif 2 in cls_list:
        group_cls = 2
    else:
        group_cls = 0
    group_ids.append(gid)
    group_labels.append(group_cls)

group_ids = np.array(group_ids)
group_labels = np.array(group_labels)

# --- 5. StratifiedGroupKFold: split train (70%) / temp (30%) ---
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, temp_idx = next(sgkf.split(group_ids, group_labels, groups=group_ids))

train_groups = group_ids[train_idx]
temp_groups = group_ids[temp_idx]
temp_labels = group_labels[temp_idx]

# --- 6. Split temp na val/test (50/50) ---
sgkf2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
val_idx, test_idx = next(sgkf2.split(temp_groups, temp_labels, groups=temp_groups))

val_groups = temp_groups[val_idx]
test_groups = temp_groups[test_idx]

# --- 7. Mapowanie grup na obrazy ---
train_data, val_data, test_data = [], [], []

for gid, items in group_to_items.items():
    if gid in train_groups:
        train_data.extend([x for x, _ in items])
    elif gid in val_groups:
        val_data.extend([x for x, _ in items])
    else:
        test_data.extend([x for x, _ in items])

# --- 8. Tworzenie DataFrame ---
all_data = []
for dataset, split in [(train_data, "train"), (val_data, "val"), (test_data, "test")]:
    for item in dataset:
        img_path, resized, yolo, tumor_mask, fluid_mask = item
        metadata_path = yolo.replace("labels", "metadata").replace(".txt", ".json")
        cls = get_class(tumor_mask, fluid_mask)
        all_data.append({
            "metadata": metadata_path,
            "class": cls,
            "split": split
        })

df = pd.DataFrame(all_data)

# --- 9. Oblicz augment_times dla train ---
train_df = df[df.split=="train"]
class_counts = train_df['class'].value_counts().to_dict()

# Cel: maksymalnie 3000 na klasę w train (można zmienić)
max_target = 3000
augment_targets = {cls: max_target for cls in class_counts.keys()}

def compute_augment_times(row):
    if row.split != "train":
        return 0
    cls = row['class']
    current_count = class_counts[cls]
    target = augment_targets[cls]
    times = max((target // current_count) - 1, 0)
    return times

df['augment_times'] = df.apply(compute_augment_times, axis=1)

# --- 10. Zostaw tylko potrzebne kolumny ---
df_final = df[['metadata', 'split', 'augment_times']]

# --- 11. Podsumowanie: ile w każdym splicie z jakiej klasy ---
summary = df.groupby(['split', 'class']).size().unstack(fill_value=0)
class_names = {0: "Nothing", 1: "Tumor only", 2: "Fluid only", 3: "Tumor + fluid"}
summary.rename(columns=class_names, inplace=True)
print("Liczba obrazów w każdym splicie per klasa:")
print(summary)

# --- 12. Gotowy DataFrame wynikowy ---
print("\nPrzykład df_final:")
print(df_final.head())

print("Augmentation started...")

df_train = pd.DataFrame({'image_path': [], 'label_path': [], 'tumor_mask_path': [], 'fluid_mask_path': []})
df_test = pd.DataFrame({'image_path': [], 'label_path': [], 'tumor_mask_path': [], 'fluid_mask_path': []})
df_val = pd.DataFrame({'image_path': [], 'label_path': [], 'tumor_mask_path': [], 'fluid_mask_path': []})

for idx, row in tqdm.tqdm(df_final.iterrows(), total=len(df_final)):
    image_path = row['metadata'].replace('metadata', 'resized_images').replace('.json', '.png')
    label_path = row['metadata'].replace('metadata', 'labels').replace('.json', '.txt')
    tumor_mask_path = row['metadata'].replace('metadata', 'masks/tumor').replace('.json', '.png')
    fluid_mask_path = row['metadata'].replace('metadata', 'masks/fluid').replace('.json', '.png')
    original_image_path = row['metadata'].replace('metadata', 'original_images').replace('.json', '.png')
    if row['split'] == 'test':
        df_test = pd.concat([df_test, pd.DataFrame({
            'image_path': [image_path],
            'label_path': [label_path],
            'tumor_mask_path': [tumor_mask_path],
            'fluid_mask_path': [fluid_mask_path]
        })], ignore_index=True)
    elif row['split'] == 'val':
        df_val = pd.concat([df_val, pd.DataFrame({
            'image_path': [image_path],
            'label_path': [label_path],
            'tumor_mask_path': [tumor_mask_path],
            'fluid_mask_path': [fluid_mask_path]
        })], ignore_index=True)
    else:
        df_train = pd.concat([df_train, pd.DataFrame({
            'image_path': [image_path],
            'label_path': [label_path],
            'tumor_mask_path': [tumor_mask_path],
            'fluid_mask_path': [fluid_mask_path]
        })], ignore_index=True)
        augment_times = row['augment_times']
        if augment_times > 0:
            augmentations = MyA.get_all_oct_image_augmentations(original_image_path)
            potentials = augmentations[1]
            for augmentation_paths in augmentations[0]:
                df_train = pd.concat([df_train, pd.DataFrame({
                    'image_path': [augmentation_paths[0]],
                    'label_path': [augmentation_paths[1]],
                    'tumor_mask_path': [augmentation_paths[2]],
                    'fluid_mask_path': [augmentation_paths[3]]
                })], ignore_index=True)
                augment_times -= 1
                if augment_times <= 0:
                    break
            if augment_times > 0:
                for _ in range(augment_times):
                    (augmented_image, 
                     augmented_fluid_mask, 
                     augmented_tumor_mask) = MyA.augment_image(image_path, 
                                                               fluid_mask_path, 
                                                               tumor_mask_path)
                    potentials += 1
                    augmented_resized_image_path = os.path.join(os.path.dirname(
                        image_path.replace('raw', 'processed').replace('OCT', 'OCT_augmented')
                    ), f'{potentials}.png')
                    augmented_label_path = os.path.join(os.path.dirname(
                        label_path.replace('raw', 'processed').replace('OCT', 'OCT_augmented')
                    ), f'{potentials}.txt')
                    augmented_tumor_mask_path = os.path.join(os.path.dirname(
                        tumor_mask_path.replace('raw', 'processed').replace('OCT', 'OCT_augmented')
                    ), f'{potentials}.png')
                    augmented_fluid_mask_path = os.path.join(os.path.dirname(
                        fluid_mask_path.replace('raw', 'processed').replace('OCT', 'OCT_augmented')
                    ), f'{potentials}.png')
                    augmented_metadata_path = os.path.join(os.path.dirname(
                        row['metadata'].replace('raw', 'processed').replace('OCT', 'OCT_augmented')), 
                        f'{potentials}.json'
                    )
                    paths = [
                        augmented_resized_image_path,
                        augmented_label_path,
                        augmented_tumor_mask_path,
                        augmented_fluid_mask_path,
                        augmented_metadata_path,
                    ]

                    for p in paths:
                        os.makedirs(os.path.dirname(p), exist_ok=True)
                        
                    cv2.imwrite(augmented_resized_image_path, augmented_image)
                    cv2.imwrite(augmented_tumor_mask_path, augmented_tumor_mask)
                    cv2.imwrite(augmented_fluid_mask_path, augmented_fluid_mask)
                    mask2yolo_separate_inputs(
                        augmented_tumor_mask_path,
                        augmented_fluid_mask_path,
                        augmented_label_path,
                        override=False
                    )
                    with open(row['metadata'], 'r') as f:
                        meta = json.load(f)
                        meta['raw_source'] = original_image_path.replace('Ophthalmic_Scans\\', '')
                        meta['image_type'] = meta['image_type'] + '_augmented'
                    with open(augmented_metadata_path, 'w') as f:
                        json.dump(meta, f, indent=4)
                    df_train = pd.concat([df_train, pd.DataFrame({
                        'image_path': [augmented_resized_image_path],
                        'label_path': [augmented_label_path],
                        'tumor_mask_path': [augmented_tumor_mask_path],
                        'fluid_mask_path': [augmented_fluid_mask_path]
                    })])
                    

print("Augmentation finished.")

df_train.to_csv('train.csv', index=False)
df_test.to_csv('test.csv', index=False)
df_val.to_csv('val.csv', index=False)