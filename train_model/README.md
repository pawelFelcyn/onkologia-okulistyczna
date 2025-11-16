# YOLO Training Pipeline from CSV (Ophthalmic Scans)

This project trains a YOLOv8 model using image--label pairs defined in
CSV files.\
Instead of copying files, the script creates **hardlinks**, which are
stable on Windows and OneDrive.

------------------------------------------------------------------------

## 🚀 Features

-   Builds YOLO dataset structure automatically:

        yolo_dataset/
            images/train/
            images/val/
            labels/train/
            labels/val/

-   Uses **hardlinks (os.link)** → zero extra disk usage.

-   Reads CSVs containing:

    -   `image_path`
    -   `label_path`

-   Trains YOLOv8 on the generated dataset.

-   Saves model weights with automatic unique filenames.

------------------------------------------------------------------------

## 📦 CSV Format

Each CSV row must contain:

  -----------------------------------------------------------------------
  Column                               Description
  ------------------------------------ ----------------------------------
  `image_path`                         relative path to image inside
                                       `Ophthalmic_Scans/`

  `label_path`                         relative path to YOLO TXT label
                                       inside `Ophthalmic_Scans/`
  -----------------------------------------------------------------------

Example:

    image_path,label_path
    sub-xx/.../img001.jpg,sub-xx/.../img001.txt
    ...

------------------------------------------------------------------------

## 🏋️ Training

Run:

``` bash
python train_yolo.py     --train_csv Ophthalmic_Scans/splits/tumor_and_fluid_segmentation_oct/train.csv     --val_csv Ophthalmic_Scans/splits/tumor_and_fluid_segmentation_oct/val.csv     --epochs 50     --imgsz 1024     --batch 16
```

------------------------------------------------------------------------

## 📁 Output

Weights are saved into:

    models/weights.pt

If file exists, script auto-generates:

    weights(1).pt
    weights(2).pt
    ...

------------------------------------------------------------------------

## 🛠️ Implementation Notes

-   Hardlinks (`os.link`) ensure:
    -   YOLO treats images as normal files.
    -   No duplicated disk usage.
    -   Full compatibility with Windows + OneDrive.
-   `os.path.lexists()` ensures broken links get removed before
    recreating them.

------------------------------------------------------------------------

## ✔️ Summary

This script provides a **fully automated YOLO training pipeline** based
on CSV-defined datasets, optimized for Windows environments, avoiding
broken symlink issues by using hardlinks.
