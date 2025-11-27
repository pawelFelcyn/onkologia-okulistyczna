# YOLO & UNET Training & Testing Pipeline from CSV (Ophthalmic Scans)

This project provides a complete pipeline for **training and testing YOLOv8 and UNET** models using image--label pairs defined in CSV files.
The pipeline does **not copy files** --- instead, it creates **hardlinks**, which work reliably on Windows and OneDrive.

---

## 🚀 Features

* Automatically builds the YOLO dataset structure:

  ```
  yolo_dataset/
      images/train/
      images/val/
      images/test/
      labels/train/
      labels/val/
      labels/test/
  ```

* Reads CSVs with:

  * `image_path` and `label_path` for YOLO
  * `image_path` and `mask_path` for UNET

* Trains **YOLOv8** models from CSV-generated splits

* Tests trained YOLO models on a CSV-defined test set

* Trains **UNET** models for segmentation tasks

* Tests trained UNET models using confusion matrices and metrics

* Automatic unique filenames for saved model weights

* Supports overriding defaults using a `.env` file

---

## 📦 CSV Format

### YOLO

Each CSV row must contain:

---

Column                               Description

---

`image_path`                         relative path to image inside
`Ophthalmic_Scans/`

`label_path`                         relative path to YOLO TXT label
inside `Ophthalmic_Scans/`
--------------------------

Example:

```
image_path,label_path
sub-xx/.../img001.jpg,sub-xx/.../img001.txt
...
```

### UNET

Each CSV row must contain:

---

Column                               Description

---

`image_path`                         relative path to image inside
`Ophthalmic_Scans/`

`mask_path`                          relative path to mask image
inside `Ophthalmic_Scans/`
--------------------------

Example:

```
image_path,mask_path
sub-xx/.../img001.jpg,sub-xx/.../mask001.png
...
```

---

## 🌍 Environment Variable Overrides (`.env`)

Both training and testing scripts load:

```
train_model/.env
```

Environment variables:

---

Variable               Description                  Default

---

`SPLIT`                Path to folder containing    `Ophthalmic_Scans/splits/tumor_and_fluid_segmentation`
`train.csv`, `val.csv`,
`test.csv`

`EPOCHS`               Number of training epochs    `50`

## `BATCH`                Batch size                   `16`

Example `.env`:

```
SPLIT=Ophthalmic_Scans/splits/custom_split
EPOCHS=100
BATCH=32
```

---

## 🏋️ YOLO Training

Run manual training:

```bash
python train_yolo.py --train_csv Ophthalmic_Scans/splits/tumor_and_fluid_segmentation_oct/train.csv --val_csv Ophthalmic_Scans/splits/tumor_and_fluid_segmentation_oct/val.csv --epochs 50 --imgsz 1024 --batch 16
```

Or rely entirely on `.env`:

```bash
python train_yolo.py
```

## 📁 YOLO Output (Training)

Weights are saved inside:

```
models/weights.pt
```

If this file exists, the script automatically saves:

```
weights(1).pt
weights(2).pt
...
```

---

## 🧪 YOLO Testing

Use:

```bash
python test_yolo.py --test_csv Ophthalmic_Scans/splits/tumor_and_fluid_segmentation/test.csv --model_to_test models/weights.pt
```

Or using `.env` defaults:

```bash
python test_yolo.py
```

The script:

1. Builds a YOLO `test` split using hardlinks
2. Runs `model.val(split="test")`
3. Produces full YOLOv8 evaluation metrics

---

## 🏋️ UNET Training

Run manual training:

```bash
python train_unet.py --train_csv Ophthalmic_Scans/splits/tumor_and_fluid_segmentation/train.csv --val_csv Ophthalmic_Scans/splits/tumor_and_fluid_segmentation/val.csv --epochs 50 --imgsz 1024 --batch 16
```

Or using `.env` defaults:

```bash
python train_unet.py
```

### Output

Weights are saved inside:

```
models/unet/weights.pth
```

If this file exists, the script automatically saves:

```
weights(1).pth
weights(2).pth
...
```

---

## 🧪 UNET Testing

Use:

```bash
python test_unet.py --split Ophthalmic_Scans/splits/tumor_and_fluid_segmentation --model_to_test models/unet/weights.pth --batch 16
```

The script:

1. Loads `test.csv` from the split folder
2. Runs the UNET model on test images
3. Computes confusion matrices and metrics for fluid and tumor
4. Saves metrics and confusion matrices in JSON files

---

## 🛠️ Implementation Notes

* Hardlinks (`os.link`) ensure:

  * No duplicated disk usage
  * Full compatibility with Windows + OneDrive
  * YOLO treats the files as normal images
* UNET datasets convert masks into multi-channel tensors for fluid and tumor
* Both training and testing support silent default override through `.env`

---

## ✔️ Summary

This repository offers a **complete YOLO and UNET training & testing workflow** based on CSV-defined datasets.
Hardlinks keep disk usage to a minimum, while `.env` overrides make the pipeline customizable and easy to automate.