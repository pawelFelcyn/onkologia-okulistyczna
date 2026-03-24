# Transfer Learning - Seeds And Runs

## Why Use Multiple Seeds
Results can vary between runs, especially in medical imaging tasks, so stability should be reported.

Use this setup:
- run each variant on the same seed set,
- keep seeds identical across baseline vs transfer vs freeze,
- report mean +- std across seeds.

## Seed Set Used In This Project
For this thesis workflow, we use 5 seeds:
- 13
- 42
- 123
- 73
- 99


## How To Run Scripts (Arguments)
Scripts in cluster_scripts/transfer-learning/ accept run seed as the first argument.
If no run seed is provided, default is 42.

### Kermany Pretraining (Encoder)
- Train: sbatch cluster_scripts/transfer-learning/train_kermany.sh 13
- Evaluate: sbatch cluster_scripts/transfer-learning/eval_kermany.sh 13

Default encoder output path:
- train_model/transfer_learning/runs_kermany_seed<SEED>/encoder_kermany_pretrained.pth

### UNet Baseline (Scratch)
- sbatch cluster_scripts/transfer-learning/train_unet_baseline.sh 13

Saved model path:
- models/unet/baseline_scratch_seed<SEED>.pth

### UNet Transfer (Kermany)
- sbatch cluster_scripts/transfer-learning/train_unet_transfer_kermany.sh 13

Arguments:
- arg1: run seed (for current UNet training run),
- arg2: Kermany seed used to select pretrained encoder weights (default 42),
- arg3: explicit encoder weights path (optional, overrides arg2-derived path).

By default, encoder weights are loaded from runs_kermany_seed<ARG2_OR_42>.

Examples:
- use run seed 13 and Kermany weights from seed 42:
	sbatch cluster_scripts/transfer-learning/train_unet_transfer_kermany.sh 13 42
- override encoder path explicitly:
	sbatch cluster_scripts/transfer-learning/train_unet_transfer_kermany.sh 13 42 train_model/transfer_learning/runs_kermany_seed42/encoder_kermany_pretrained.pth

Saved model path:
- models/unet/kermany_transfer_seed<SEED>.pth

### UNet Transfer + Frozen Encoder
- sbatch cluster_scripts/transfer-learning/train_unet_transfer_kermany_freeze.sh 13

Arguments are the same as in UNet Transfer:
- arg1: run seed,
- arg2: Kermany seed for encoder weights (default 42),
- arg3: optional explicit encoder weights path.

Example:
- sbatch cluster_scripts/transfer-learning/train_unet_transfer_kermany_freeze.sh 13 42

Saved model path:
- models/unet/kermany_transfer_frozen_seed<SEED>.pth

### UNet Test Evaluation
- transfer: sbatch cluster_scripts/transfer-learning/eval_unet_kermany.sh 13
- freeze: sbatch cluster_scripts/transfer-learning/eval_unet_kermany_freeze.sh 13

Test outputs are written to runs_unet/test_run* (from train_model/test_unet.py).

## Final Training Outputs (What You Get Now)
Each UNet training run now creates a named directory in runs_unet with approach, seed and timestamp encoded in the folder name.

In each run directory you will get:
- run_meta.json: configuration and provenance (seed, approach, split, commit, SLURM job id, resume mode),
- run_summary.json: final summary with best metrics and saved model path,
- weights/: best.pth, last.pth, best_tumor.pth, best_fluid.pth,
- tensorboard/: TensorBoard logs,
- epoch_<N>/epoch_data.json: per-epoch train/val metrics.

Global index:
- runs_unet/experiments_index.csv is appended after each training run (one row per run).
