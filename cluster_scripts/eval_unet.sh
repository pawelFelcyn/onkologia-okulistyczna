#!/usr/bin/env bash
#SBATCH --job-name=eval_unet_model
#SBATCH --partition=gpu_spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=unet_eval_task-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mem=8G

module load anaconda
conda activate nn_train
cd /projects/onkokul/onkologia-okulistyczna || exit -1

# ── Edit before submitting ──────────────────────────────────────────────────
# Point MODEL_PATH to the checkpoint you want to evaluate, e.g.:
#   runs_unet/run1/weights/best.pth
#   models/unet/baseline_scratch_seed42.pth
MODEL_PATH="models/unet/baseline_scratch_seed42.pth"
# ───────────────────────────────────────────────────────────────────────────

srun python train_model/test_unet.py \
  --split         Ophthalmic_Scans/splits/tumor_and_fluid_segmentation_oct \
  --model_to_test "${MODEL_PATH}" \
  --imgsz         512 \
  --batch         8
