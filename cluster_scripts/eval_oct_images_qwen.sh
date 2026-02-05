#!/usr/bin/env bash
#SBATCH --job-name=eval_oct_gens
#SBATCH --partition=gpu_spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=eval_oct_images-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mem=30G

module load anaconda
conda activate llm_eval
cd /projects/onkokul/onkologia-okulistyczna/llm/generated_scans_evaluation || exit -1
pip install -r requirements.txt
cd ../.. || exit -1

srun llm/generated_scans_evaluation/describe_qwen.py