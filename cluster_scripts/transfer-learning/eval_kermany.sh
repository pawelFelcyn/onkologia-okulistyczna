#!/usr/bin/env bash
#SBATCH --job-name=eval_kermany_model
#SBATCH --partition=gpu_spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=eval_kermany_task-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mem=16G

module load anaconda
conda activate nn_train
cd /projects/onkokul/onkologia-okulistyczna || exit 1

srun python train_model/transfer_learning/eval_kermany.py \
  --weights    train_model/transfer_learning/runs_kermany_seed42/encoder_kermany_pretrained.pth \
  --data_dir   train_model/transfer_learning/OCT2018 \
  --output_dir train_model/transfer_learning/runs_kermany_seed42