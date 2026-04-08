# Transfer Learning – Kermany OCT Encoder Pretraining

## Goal

Pretrain a **U-Net encoder** on the [Kermany OCT 2018](https://www.kaggle.com/datasets/paultimothymooney/kermany2018) classification dataset before using it for retinal layer segmentation.

The encoder learns OCT-specific features (retinal layers, fluid textures, pathologies) from ~84k labelled images across 4 classes. These weights are then transferred to Stage B (U-Net segmentation).

---

## File Structure

```
transfer_learning/
├── kermany_dataset.py   # Dataset: download, validation, summary, transforms, DataLoaders
├── kermany_model.py     # Model: UNetEncoder, KermanyClassifier
├── prepare_kermany.py   # Download + validate + summarize dataset
├── train_kermany.py     # Training loop on already prepared data
└── eval_kermany.py      # Evaluation: accuracy, macro F1, per-class F1, confusion matrix
```

---

## Dataset

**Kermany OCT 2018** – 4-class OCT classification:

| Class  | Description              |
|--------|--------------------------|
| CNV    | Choroidal neovascularisation |
| DME    | Diabetic macular oedema  |
| DRUSEN | Drusen deposits          |
| NORMAL | Healthy retina           |

Expected directory layout after download:

```
OCT2018/
    train/
        CNV/      DME/      DRUSEN/      NORMAL/
    test/
        CNV/      DME/      DRUSEN/      NORMAL/
```

---

## Requirements

```bash
pip install torch torchvision torchmetrics tensorboard tqdm kagglehub kaggle
```

Preferred download backend: `kagglehub`.

Fallback download backend: `kaggle` API if `kagglehub` is unavailable or fails in your environment.

If you use the fallback Kaggle API flow, you may need an API token:  
Get it at https://www.kaggle.com/settings → **API → Create New Token**  
Place the downloaded `kaggle.json` in `~/.kaggle/kaggle.json`

---

## Step-by-Step Guide

### Step 1 – Download the dataset

```bash
python prepare_kermany.py --download --data_dir ./OCT2018
```

This script:
- downloads the dataset with `kagglehub` by default,
- falls back to the Kaggle API if needed,
- validates the expected `train/` and `test/` folder structure,
- prints image counts for every class and split.

It does not apply augmentation and does not rewrite the images on disk. All resizing and augmentation are applied later, on the fly, inside the PyTorch data pipeline used by training and evaluation.

Note: the Kaggle dataset slug is `kermany2018`, but some downloaded copies may contain an internal folder named `OCT2017`. The preparation script normalizes that and copies the usable dataset into your requested `--data_dir`.

If the dataset is already downloaded, you can run only the validation/summary step:

```bash
python prepare_kermany.py --data_dir ./OCT2018
```

---

### Step 2 – Train the classifier

```bash
python train_kermany.py \
    --data_dir   ./OCT2018 \
    --epochs     25 \
    --batch_size 8 \
    --seed       42 \
    --output_dir ./runs_kermany
```

Recommended after download: run training directly on the prepared dataset. The current pipeline resizes samples to `512×512` during loading, which matches downstream U-Net fine-tuning.

The `--seed` flag fixes the train/validation split, DataLoader shuffling, worker RNG state and the main PyTorch/NumPy/Python RNGs, which makes experiments much easier to reproduce.

**What happens:**
- Splits the train set into train (90%) and validation (10%)
- Trains a U-Net encoder + linear classification head with `CrossEntropyLoss`
- Monitors **macro F1** on the validation set
- Saves **only the encoder** on every improvement → `runs_kermany/encoder_kermany_pretrained.pth`
- Applies **early stopping** (default patience = 5 epochs)
- Logs loss / accuracy / F1 to **TensorBoard**

**All CLI flags:**

| Flag             | Default           | Description                          |
|------------------|-------------------|--------------------------------------|
| `--data_dir`     | `./OCT2018`       | Path to the dataset root             |
| `--epochs`       | `25`              | Maximum number of epochs             |
| `--batch_size`   | `32`              | Batch size                           |
| `--lr`           | `1e-4`            | Learning rate (Adam)                 |
| `--weight_decay` | `1e-4`            | Weight decay (Adam)                  |
| `--early_stop`   | `5`               | Early stopping patience              |
| `--base`         | `64`              | Base channels of the U-Net encoder   |
| `--num_workers`  | `4`               | DataLoader worker threads            |
| `--val_split`    | `0.1`             | Fraction of train set used for val   |
| `--seed`         | `42`              | Random seed for reproducible runs    |
| `--output_dir`   | `./runs_kermany`  | Directory for weights and logs       |

**CLI flags for `prepare_kermany.py`:**

| Flag         | Default     | Description                                 |
|--------------|-------------|---------------------------------------------|
| `--download` | off         | Download dataset before validation          |
| `--data_dir` | `./OCT2018` | Path to the dataset root                    |

---

### Step 3 – Monitor training (optional)

```bash
tensorboard --logdir ./runs_kermany/tensorboard
```

Open http://localhost:6006 in your browser.

---

### Step 4 – Evaluate on the test set

```bash
python eval_kermany.py \
    --weights    ./runs_kermany/encoder_kermany_pretrained.pth \
    --data_dir   ./OCT2018 \
    --output_dir ./runs_kermany
```

**Output:**
- Accuracy (macro), F1 (macro), F1 per class
- Confusion matrix printed to console
- Results saved to `runs_kermany/eval_results.json`

The script accepts both a **full model checkpoint** and an **encoder-only checkpoint** – it detects which one automatically.

---

## Stage B – Transfer encoder to U-Net

After pretraining, use the encoder weights in segmentation training via `train_unet.py`.

Current project flow loads Kermany encoder weights with:
- `--encoder_weights <path_to_encoder_checkpoint>`
- `model.load_state_dict(state, strict=False)` inside `train_model/train_unet.py`

This initializes matching encoder blocks (`conv1`-`conv5`) from the pretrained checkpoint, while decoder/output layers remain randomly initialized and are trained on segmentation data.

Example command:

```bash
python train_model/train_unet.py \
    --train_csv Ophthalmic_Scans/splits/tumor_and_fluid_segmentation_oct/train.csv \
    --val_csv   Ophthalmic_Scans/splits/tumor_and_fluid_segmentation_oct/val.csv \
    --epochs    50 \
    --imgsz     512 \
    --batch     8 \
    --approach  transfer \
    --seed      42 \
    --save_path models/unet/kermany_transfer_seed42.pth \
    --encoder_weights train_model/transfer_learning/runs_kermany_seed42/encoder_kermany_pretrained.pth
```

If you want transfer learning with a frozen encoder, add `--freeze_encoder`.

The encoder architecture (`double_conv` blocks, `base=64`) is aligned between Kermany pretraining and `UNet` in `train_model/unet_utils.py`, so matching weights are loaded automatically.

---

## Augmentation Policy

The following transforms are applied on the fly during data loading. The downloaded dataset in `OCT2018/` stays unchanged on disk.

| Transform        | Train | Val / Test | Notes                                      |
|------------------|-------|------------|--------------------------------------------|
| Resize(560)      | ✓     |            | Prepares a larger canvas before cropping   |
| RandomCrop(512)  | ✓     |            | Final train size                           |
| Resize(512)      |       | ✓          | Final validation/test size                 |
| CenterCrop(512)  |       | ✓          | Keeps evaluation deterministic             |
| HorizontalFlip   | ✓     |            |                                            |
| Rotation ±10°    | ✓     |            |                                            |
| ColorJitter      | ✓     |            | brightness=0.2, contrast=0.2               |
| VerticalFlip     | ✗     | ✗          | Disabled – OCT has anatomical top-to-bottom orientation |
| Normalize(0.5)   | ✓     | ✓          | mean=0.5, std=0.5 per channel              |
