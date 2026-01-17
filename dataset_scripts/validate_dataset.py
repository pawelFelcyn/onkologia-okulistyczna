import os
import pandas as pd

TEST = 'test'
TRAIN = 'train'
VAL = 'val'

def validate_split_file(path: str, split_kind: str):
    if not os.path.isfile(path):
        print(f"⚠️  Warning: {split_kind} split file not found at '{path}'.")
        return
    print(f"✅  {split_kind} split file found at '{path}'.")
    df = pd.read_csv(path)
    if df.shape[0] == 0:
        print(f"⚠️  Empty {path}")
        return
    error_flag = False
    for _, row in df.iterrows():
        image_path = os.path.join('Ophthalmic_Scans', row["image_path"])
        label_path = os.path.join('Ophthalmic_Scans', row["label_path"])
        tumor_mask_path = os.path.join('Ophthalmic_Scans', row["tumor_mask_path"])
        fluid_mask_path = os.path.join('Ophthalmic_Scans', row["fluid_mask_path"])
        #for now dont validate augmented images
        if 'augmented' in image_path:
            continue
        if not os.path.isfile(image_path):
            error_flag = True
            print(f"⛔  Error: Image not found at '{image_path}' in {split_kind} split.")
        if not os.path.isfile(label_path):
            error_flag = True
            print(f"⛔  Error: Label not found at '{label_path}' in {split_kind} split.")
        if not os.path.isfile(tumor_mask_path):
            error_flag = True
            print(f"⛔  Error: Tumor mask not found at '{tumor_mask_path}' in {split_kind} split.")
        if not os.path.isfile(fluid_mask_path):
            error_flag = True
            print(f"⛔  Error: Fluid mask not found at '{fluid_mask_path}' in {split_kind} split.")
    if not error_flag:
        print(f"✅  All paths found in {split_kind} split.")

print("Validating dataset in progress...")

splits_dir = 'Ophthalmic_Scans/splits/'

print("Validating splits...")
for split_dir in [path for path in os.listdir(splits_dir) if os.path.isdir(os.path.join(splits_dir, path))]:
    print(f"Validating {split_dir}...")
    test_path = os.path.join(splits_dir, split_dir, 'test.csv')
    train_path = os.path.join(splits_dir, split_dir, 'train.csv')
    val_path = os.path.join(splits_dir, split_dir, 'val.csv')
    validate_split_file(test_path, TEST)
    validate_split_file(train_path, TRAIN)
    validate_split_file(val_path, VAL)
print("Splits validated.")

print("Validating masks and labels consistency.")

root = 'Ophthalmic_Scans'

not_found_tumor = 0
not_found_fluid = 0
tumor_found = 0
fluid_found = 0

for dirpath, dirnames, filenames in [x for x in os.walk(root) if os.path.basename(x[0]).endswith('labels')]:
    for filename in filenames:
        label_path = os.path.join(dirpath, filename)
        tumor_mask_path = label_path.replace('/labels/', '/masks/tumor/').replace('\\labels\\', '\\masks\\tumor\\').replace('.txt', '.png')
        fluid_mask_path = label_path.replace('/labels/', '/masks/fluid/').replace('\\labels\\', '\\masks\\fluid\\').replace('.txt', '.png')
        if not os.path.isfile(tumor_mask_path):
            not_found_tumor += 1
            print(f"⚠️  Warning: Tumor mask not found for label '{label_path}'.")
        if not os.path.isfile(fluid_mask_path):
            not_found_fluid += 1
            print(f"⚠️  Warning: Fluid mask not found for label '{label_path}'.")
        if os.path.isfile(tumor_mask_path):
            tumor_found += 1
        if os.path.isfile(fluid_mask_path):
            fluid_found += 1

for dirpath, dirnames, filenames in [x for x in os.walk(root) if os.path.basename(x[0]).endswith('masks/tumor') or os.path.basename(x[0]).endswith('masks\\fluid')]:
    for filename in [x for x in filenames if x.endswith('.png')]:
        tumor_mask_path = os.path.join(dirpath, filename)
        label_path = label_path.replace('/masks/tumor/', '/labels/').replace('\\masks\\tumor\\', '\\labels\\').replace('.png', '.txt')
        fluid_mask_path = label_path.replace('/masks/tumor/', '/masks/fluid/').replace('\\masks\\tumor\\', '\\masks\\fluid\\')
        if not os.path.isfile(label_path):
            print(f"⚠️  Warning: Label not found for tumor mask '{tumor_mask_path}'.")
        if not os.path.isfile(fluid_mask_path):
            print(f"⚠️  Warning: Fluid mask not found for tumor mask '{tumor_mask_path}'.")
            
for dirpath, dirnames, filenames in [x for x in os.walk(root) if os.path.basename(x[0]).endswith('masks/fluid') or os.path.basename(x[0]).endswith('masks\\fluid')]:
    for filename in [x for x in filenames if x.endswith('.png')]:
        fluid_mask_path = os.path.join(dirpath, filename)
        label_path = label_path.replace('/masks/fluid/', '/labels/').replace('\\masks\\fluid\\', '\\labels\\').replace('.png', '.txt')
        tumor_mask_path = label_path.replace('/masks/fluid/', '/masks/tumor/').replace('\\masks\\fluid\\', '\\masks\\tumor\\')
        if not os.path.isfile(label_path):
            print(f"⚠️  Warning: Label not found for fluid mask '{tumor_mask_path}'.")
        if not os.path.isfile(fluid_mask_path):
            print(f"⚠️  Warning: Tumor mask not found for fluid mask '{tumor_mask_path}'.")
            
print("Validated masks and labels consistency.")
print("Not found tumor masks:", not_found_tumor)
print("Not found fluid masks:", not_found_fluid)
print("Found tumor masks:", tumor_found)
print("Found fluid masks:", fluid_found)