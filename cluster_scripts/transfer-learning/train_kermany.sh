#!/usr/bin/env bash
#SBATCH --job-name=train_kermany_model
#SBATCH --partition=gpu_spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --output=train_kermany_task-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mem=16G

# Usage:
#   sbatch train_kermany.sh 13

set -euo pipefail

SEED=${1:-42}

module load anaconda
conda activate nn_train
cd /projects/onkokul/onkologia-okulistyczna || exit -1

OUT_DIR="train_model/transfer_learning/runs_kermany_seed${SEED}"

echo "[INFO] Kermany pretraining | seed=${SEED} | out=${OUT_DIR}"

srun python train_model/transfer_learning/train_kermany.py \
  --data_dir   train_model/transfer_learning/OCT2018 \
  --epochs     25 \
  --batch_size 8 \
  --seed       "${SEED}" \
  --output_dir "${OUT_DIR}"