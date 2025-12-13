#!/usr/bin/env bash
#SBATCH --job-name=train_yolo_model
#SBATCH --partition=gpu_spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=yolo_train_task-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mem=16G

module load anaconda
conda activate nn_train
cd /projects/onkokul/onkologia-okulistyczna || exit -1

srun python train_model/test_yolo.py