import os
import pandas as pd

split_path = os.path.join("Ophthalmic_Scans", "splits", "tumor_and_fluid_segmentation_oct")

def print_split_details(split: str):
    path = os.path.join(split_path, f"{split}.csv")
    print(f"Reading {path}...")
    df = pd.read_csv(path)
    tumor_only = 0
    fluid_only = 0
    tumor_and_fluid = 0
    no_tumor_and_fluid = 0
    augmented = 0
    for _, row in df.iterrows():
        label_path = os.path.join('Ophthalmic_Scans', row['label_path'])
        if 'augmented' in label_path:
            augmented += 1
        with open(label_path, 'r') as f:
            lines = f.readlines()
            has_tumor = any(x for x in lines if x[0] == '1')
            has_fluid = any(x for x in lines if x[0] == '0')
            if has_tumor and has_fluid:
                tumor_and_fluid += 1
            elif has_tumor:
                tumor_only += 1
            elif has_fluid:
                fluid_only += 1
            else:
                no_tumor_and_fluid += 1
    print(f"Split: {split}")
    print(f"Tumor only: {tumor_only}")
    print(f"Fluid only: {fluid_only}")
    print(f"Tumor and fluid: {tumor_and_fluid}")
    print(f"No tumor and fluid: {no_tumor_and_fluid}")
    print(f"Total: {tumor_only + fluid_only + tumor_and_fluid + no_tumor_and_fluid}")
    print(f"Augmented: {augmented}")

print_split_details("train")
print_split_details("val")
print_split_details("test")