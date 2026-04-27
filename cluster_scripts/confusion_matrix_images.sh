#!/usr/bin/env bash
#SBATCH --job-name=confusion_matrix_images
#SBATCH --partition=gpu_spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=confusion_matrix_images-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mem=16G

module load anaconda
conda activate nn_train
cd /projects/onkokul/onkologia-okulistyczna || exit -1

srun python train_model/confusion_matrix_images.py
