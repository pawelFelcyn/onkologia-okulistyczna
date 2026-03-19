#!/usr/bin/env bash
#SBATCH --job-name=eval_kermany_model
#SBATCH --partition=gpu_spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=eval_kermany_task-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mem=16G

# Usage:
#   sbatch eval_kermany.sh 13
# Optional 2nd arg: weights path
#   sbatch eval_kermany.sh 13 train_model/transfer_learning/runs_kermany_seed13/encoder_kermany_pretrained.pth

set -euo pipefail

SEED=${1:-42}
OUT_DIR="train_model/transfer_learning/runs_kermany_seed${SEED}"
WEIGHTS=${2:-"${OUT_DIR}/encoder_kermany_pretrained.pth"}

module load anaconda
conda activate nn_train
cd /projects/onkokul/onkologia-okulistyczna || exit 1

echo "[INFO] Kermany eval | seed=${SEED} | weights=${WEIGHTS} | out=${OUT_DIR}"

srun python train_model/transfer_learning/eval_kermany.py \
  --weights    "${WEIGHTS}" \
  --data_dir   train_model/transfer_learning/OCT2018 \
  --output_dir "${OUT_DIR}"