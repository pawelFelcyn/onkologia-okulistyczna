#!/usr/bin/env bash
#SBATCH --job-name=train_unet_model
#SBATCH --partition=gpu_spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --output=unet_train_task-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mem=16G

# Usage:
#   sbatch train_unet_tensorboard.sh 13
#   sbatch --job-name=train_unet_model_s13 train_unet_tensorboard.sh 13

set -euo pipefail

SEED=${1:-42}

module load anaconda
conda activate nn_train
pip install "tensorboard>=2.14"
cd /projects/onkokul/onkologia-okulistyczna || exit -1

echo "[INFO] Baseline UNet training | seed=${SEED}"

srun python train_model/train_unet.py \
  --train_csv Ophthalmic_Scans/splits/tumor_and_fluid_segmentation_oct/train.csv \
  --val_csv   Ophthalmic_Scans/splits/tumor_and_fluid_segmentation_oct/val.csv \
  --epochs    50 \
  --imgsz     512 \
  --batch     8 \
  --seed      "${SEED}" \
  --save_path "models/unet/baseline_scratch_seed${SEED}.pth"