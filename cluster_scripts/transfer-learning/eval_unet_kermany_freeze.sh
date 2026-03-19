#!/usr/bin/env bash
#SBATCH --job-name=eval_unet_kermany_model
#SBATCH --partition=gpu_spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=unet_kermany_eval_task-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mem=8G

# Usage:
#   sbatch eval_unet_kermany_freeze.sh 13

set -euo pipefail

SEED=${1:-42}

module load anaconda
conda activate nn_train
cd /projects/onkokul/onkologia-okulistyczna || exit -1

MODEL_PATH="models/unet/kermany_transfer_frozen_seed${SEED}.pth"

echo "[INFO] Evaluating Kermany transfer + freeze UNet | seed=${SEED} | model=${MODEL_PATH}"

srun python train_model/test_unet.py \
  --split         Ophthalmic_Scans/splits/tumor_and_fluid_segmentation_oct \
  --model_to_test "${MODEL_PATH}" \
  --imgsz         512 \
  --batch         8
