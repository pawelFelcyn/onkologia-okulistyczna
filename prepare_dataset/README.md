Short instructions for running prepare_dataset.py

Minimal purpose
---------------
Run the `prepare_dataset.py` script to:
- generate segmentation masks from YOLO polygon labels (saved to a `masks/` folder inside the root dataset directory),
- create a train/val/test split under the destination folder, and
- run augmentation on the training split.

Required input layout
---------------------
The script expects a dataset root directory with at least the following subfolders:

- images/       # source images (png/jpg/jpeg)
- labels/       # YOLO label files (.txt) with polygon annotations

By default the script uses the directory containing `prepare_dataset.py` as the root.

Usage (PowerShell)
------------------
Basic run using defaults (root = script directory, dest = ./dataset):

	python .\prepare_dataset.py

Run with an explicit dataset root and destination (PowerShell):

	python .\prepare_dataset.py --root_dir "C:\path\to\my_dataset_root" --dest "C:\path\to\output_dataset"

Notes
-----
- `--root_dir` should contain `images/` and `labels/` subfolders. The script will create a `masks/` folder inside `--root_dir` when generating masks.
- `--dest` is where the split (train/val/test) will be created; inside it the script places a `splits/train` folder which is then augmented.
- Augmentation parameters are defined in `augment.py` and use the training split CSV produced by the split utilities.

Example quick check
-------------------
After running, check that:

- masks/ contains corresponding mask PNGs for images in images/
- <dest>/splits/train contains augmented images and a `train.csv` describing them

