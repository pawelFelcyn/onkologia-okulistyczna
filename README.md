# Ophthalmic Scans Project

This repository contains data preparation, model training, evaluation, and demo application code for ophthalmic OCT scans.

This README focuses on the **YOLOv8 segmentation training and testing workflow**. A new user should be able to:

1. clone the repository,
2. download the dataset,
3. train the YOLO model,
4. test the trained model,
5. do this either locally or on the UAM computing cluster.

## Repository Areas

- `train_model/` - local YOLO and U-Net training/testing scripts
- `cluster_scripts/` - Slurm job scripts for running training/testing on the UAM computing cluster
- `Ophthalmic_Scans/` - DVC-managed dataset, metadata, prompts, and predefined dataset splits
- `base_models/` - starting checkpoint for YOLO training
- `models/` - exported trained weights
- `app/` - demo application for inference and visualization

If you are interested in the demo application, see [app/README.md](app/README.md).

## Current YOLO Split

The **current split version used for YOLO training and testing is**:

- `Ophthalmic_Scans/splits/tumor_and_fluid_segmentation_oct3`

Older split directories such as `tumor_and_fluid_segmentation_oct` and `tumor_and_fluid_segmentation_oct2` are historical versions and should not be used for the main workflow unless you intentionally want to reproduce older experiments.

## Dataset Access

The dataset is versioned with DVC.

### Local machine

1. Install [DVC](https://dvc.org/doc/install).
2. Connect to the WMI UAM VPN if required for dataset access: [https://laboratoria.wmi.amu.edu.pl/uslugi/vpn/](https://laboratoria.wmi.amu.edu.pl/uslugi/vpn/).
3. Ask the repository owner for access to the vm with dataset, if you don't have it.
4. From the repository root, download the data:

```bash
cd Ophthalmic_Scans
dvc pull
cd ..
```

### UAM cluster

If you are working directly on the UAM infrastructure, clone the repository on the cluster first, then run `dvc pull` there inside `Ophthalmic_Scans/`.

## YOLO Training Pipeline Overview

The local and cluster workflows both use the same Python scripts:

- `train_model/train_yolo.py` - training
- `train_model/test_yolo.py` - evaluation on the test split

What these scripts do:

1. read CSV split files,
2. build `yolo_dataset/images/...` and `yolo_dataset/labels/...` automatically using hardlinks,
3. train or validate a YOLOv8 segmentation model using `data.yaml`,
4. export final weights and metrics.

The dataset defined by `data.yaml` uses:

- `fluid`
- `tumor`

## Important Defaults and Current Project Convention

There are two important details to know before running anything:

1. The current project split is `tumor_and_fluid_segmentation_oct3`, but the Python scripts still contain an older default split path. To avoid accidental use of the old split, pass the split paths explicitly or define them in `train_model/.env`.
2. `train_model/test_yolo.py` contains an older default model path (`models/weights.pt`). The training script currently saves exported weights in `models/yolo/weights.pt`, so for testing you should pass `--model_to_test models/yolo/weights.pt` explicitly or set `TEST_MODEL=models/yolo/weights.pt` in `train_model/.env`.

Because of that, the commands below always use explicit arguments or an explicit `.env` file.

## Option A: Train and Test YOLO Locally

### Prerequisites

- Python 3.10 or 3.11
- `pip`
- DVC
- Access to the dataset
- Preferably an NVIDIA GPU with CUDA support

The repository `requirements.txt` is currently prepared for CUDA 12.4 PyTorch wheels. If your machine does not match this setup, the cluster workflow is the safer option.

### 1. Clone the repository

```bash
git clone https://github.com/pawelFelcyn/onkologia-okulistyczna.git
cd onkologia-okulistyczna
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download the dataset

```bash
cd Ophthalmic_Scans
dvc pull
cd ..
```

### 5. Create `train_model/.env`

Create `train_model/.env` with the current YOLO split and the correct exported model path:

```env
SPLIT=Ophthalmic_Scans/splits/tumor_and_fluid_segmentation_oct3
EPOCHS=100
BATCH=16
TEST_MODEL=models/yolo/weights.pt
```

Optional additional variable:

```env
CONTINUE_FROM_EPOCH=21
```

Use `CONTINUE_FROM_EPOCH` only if you want to continue training from the most recent run saved inside `runs/segment/`.

### 6. Start YOLO training

Recommended explicit command:

```bash
python train_model/train_yolo.py \
   --train_csv Ophthalmic_Scans/splits/tumor_and_fluid_segmentation_oct3/train.csv \
   --val_csv Ophthalmic_Scans/splits/tumor_and_fluid_segmentation_oct3/val.csv \
   --epochs 100 \
   --imgsz 512 \
   --batch 16
```

Or, if `train_model/.env` is already configured:

```bash
python train_model/train_yolo.py
```

### 7. Check training outputs

During and after training, look at:

- `runs/segment/` - YOLO run directory created by Ultralytics
- `models/yolo/weights.pt` - exported model checkpoint
- `models/yolo/weights(1).pt`, `weights(2).pt`, ... - if a previous file already exists

### 8. Test the trained model locally

Recommended explicit command:

```bash
python train_model/test_yolo.py \
   --test_csv Ophthalmic_Scans/splits/tumor_and_fluid_segmentation_oct3/test.csv \
   --model_to_test models/yolo/weights.pt
```

Or, if `train_model/.env` is already configured:

```bash
python train_model/test_yolo.py
```

### 9. Check evaluation outputs

After testing, inspect:

- `metrics.json` - serialized evaluation metrics from YOLO validation on the test split
- `runs/segment/` - additional Ultralytics outputs from validation

## Option B: Train and Test YOLO on the UAM Computing Cluster

Cluster documentation: [https://laboratoria.wmi.amu.edu.pl/uslugi/klaster-obliczeniowy/](https://laboratoria.wmi.amu.edu.pl/uslugi/klaster-obliczeniowy/)

The repository already contains Slurm scripts for YOLO:

- `cluster_scripts/train_yolo.sh`
- `cluster_scripts/test_yolo.sh`

These scripts currently assume:

- the repository exists at `/projects/onkokul/onkologia-okulistyczna`
- an Anaconda environment named `nn_train` already exists
- training and testing are launched from the repository root after `cd /projects/onkokul/onkologia-okulistyczna`

If you clone the repository to a different path on the cluster, you must update the `cd` line inside those Slurm scripts.

### 1. Log in to the cluster

Use the access method described in the UAM cluster documentation.

### 2. Clone the repository on the cluster

Example:

```bash
cd /projects/onkokul
git clone https://github.com/pawelFelcyn/onkologia-okulistyczna.git
cd onkologia-okulistyczna
```

If your team uses SSH instead of HTTPS, use the corresponding Git URL.

### 3. Prepare the Conda environment

The Slurm scripts expect this environment name:

```bash
nn_train
```

Example setup:

```bash
module load anaconda
conda create -n nn_train python=3.10 -y
conda activate nn_train
pip install --upgrade pip
pip install -r requirements.txt
```

If the cluster provides a preferred PyTorch/CUDA installation method, follow the cluster recommendation and then install the remaining packages from `requirements.txt`.

### 4. Download the dataset on the cluster

```bash
cd Ophthalmic_Scans
dvc pull
cd ..
```

### 5. Configure `train_model/.env` on the cluster

Create the same file as for the local workflow:

```env
SPLIT=Ophthalmic_Scans/splits/tumor_and_fluid_segmentation_oct3
EPOCHS=50
BATCH=16
TEST_MODEL=models/yolo/weights.pt
```

If you want to resume a previous run:

```env
CONTINUE_FROM_EPOCH=21
```

### 6. Submit the YOLO training job

From the repository root:

```bash
sbatch cluster_scripts/train_yolo.sh
```

What this script currently does:

- requests 1 GPU on the `gpu_spot` partition
- requests 8 CPU cores and 16 GB RAM
- activates `nn_train`
- runs `python train_model/train_yolo.py`

### 7. Monitor the job

Useful commands:

```bash
squeue -u $USER
tail -f yolo_train_task-<job_id>.out
```

### 8. Submit the YOLO test job

After training completes and the model is available in `models/yolo/weights.pt`:

```bash
sbatch cluster_scripts/test_yolo.sh
```

### 9. Check cluster outputs

Look at:

- `yolo_train_task-<job_id>.out` - Slurm logs
- `runs/segment/` - Ultralytics outputs
- `models/yolo/weights.pt` - exported model
- `metrics.json` - evaluation result written by `train_model/test_yolo.py`

## Running Without `train_model/.env`

If you do not want to create `train_model/.env`, always pass the split and model paths explicitly.

Training:

```bash
python train_model/train_yolo.py \
   --train_csv Ophthalmic_Scans/splits/tumor_and_fluid_segmentation_oct3/train.csv \
   --val_csv Ophthalmic_Scans/splits/tumor_and_fluid_segmentation_oct3/val.csv \
   --epochs 50 \
   --imgsz 512 \
   --batch 16
```

Testing:

```bash
python train_model/test_yolo.py \
   --test_csv Ophthalmic_Scans/splits/tumor_and_fluid_segmentation_oct3/test.csv \
   --model_to_test models/yolo/weights.pt
```

## Troubleshooting

### `dvc pull` does not download data

- check VPN access if you are outside the university network
- verify that you have permission to access the DVC remote

### Training starts on the wrong split

- verify that `train_model/.env` contains `SPLIT=Ophthalmic_Scans/splits/tumor_and_fluid_segmentation_oct3`
- or pass `--train_csv`, `--val_csv`, and `--test_csv` explicitly

### Testing cannot find the trained weights

- use `--model_to_test models/yolo/weights.pt`
- or set `TEST_MODEL=models/yolo/weights.pt` in `train_model/.env`

### Cluster job exits immediately

- verify that the repository really exists at `/projects/onkokul/onkologia-okulistyczna`
- otherwise edit `cluster_scripts/train_yolo.sh` and `cluster_scripts/test_yolo.sh`
- verify that the Conda environment `nn_train` exists and contains the required packages

## Additional Documentation

- Local training scripts: [train_model/README.md](train_model/README.md)
- U-Net transfer learning: [train_model/transfer_learning/README.md](train_model/transfer_learning/README.md)
- Demo application: [app/README.md](app/README.md)
