#!/usr/bin/env bash
#SBATCH --job-name=train_unet_model
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=zadanie-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pawfel1@st.amu.edu.pl

module load anaconda
conda activate nn_train
cd /projects/onkokul/onkologia-okulistyczna || exit -1

srun python train_model/train_unet.py