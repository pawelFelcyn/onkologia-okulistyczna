#!/usr/bin/env bash
#SBATCH --job-name=train_kermany_unet_model
#SBATCH --partition=gpu_spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --output=unet_train_task-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mem=16G

# Usage:
#   sbatch train_unet_transfer_kermany_freeze.sh 13
# Optional 2nd arg: Kermany seed used to pick pretrained encoder weights (default: 42)
#   sbatch train_unet_transfer_kermany_freeze.sh 13 42
# Optional 3rd arg: explicit encoder weights path (overrides 2nd arg)
#   sbatch train_unet_transfer_kermany_freeze.sh 13 42 train_model/transfer_learning/runs_kermany_seed42/encoder_kermany_pretrained.pth

set -euo pipefail

SEED=${1:-42}
KERMANY_SEED=${2:-42}
ENCODER_WEIGHTS=${3:-train_model/transfer_learning/runs_kermany_seed${KERMANY_SEED}/encoder_kermany_pretrained.pth}

module load anaconda
conda activate nn_train
cd /projects/onkokul/onkologia-okulistyczna || exit -1

echo "[INFO] Kermany transfer + freeze UNet training | run_seed=${SEED} | kermany_seed=${KERMANY_SEED} | encoder=${ENCODER_WEIGHTS}"

srun python train_model/train_unet.py \
  --train_csv Ophthalmic_Scans/splits/tumor_and_fluid_segmentation_oct/train.csv \
  --val_csv   Ophthalmic_Scans/splits/tumor_and_fluid_segmentation_oct/val.csv \
  --epochs    50 \
  --imgsz     512 \
  --batch     8 \
  --approach  transfer_freeze \
  --seed      "${SEED}" \
  --save_path "models/unet/kermany_transfer_frozen_seed${SEED}.pth" \
  --encoder_weights "${ENCODER_WEIGHTS}" \
  --freeze_encoder