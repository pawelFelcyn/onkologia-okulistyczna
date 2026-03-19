# Transfer Learning - Seeds And Runs

## Why Use Multiple Seeds
Results can vary between runs, especially in medical imaging tasks, so stability should be reported.

Use this setup:
- run each variant on the same seed set,
- keep seeds identical across baseline vs transfer vs freeze,
- report mean +- std across seeds.

## Seed Set Used In This Project
For this thesis workflow, we use 3 seeds:
- 13
- 42
- 123

## How To Run Scripts (Seed As Argument)
Scripts in cluster_scripts/transfer-learning/ accept seed as the first argument.
If no seed is provided, default is 42.

### Kermany Pretraining (Encoder)
- Train: sbatch cluster_scripts/transfer-learning/train_kermany.sh 13
- Evaluate: sbatch cluster_scripts/transfer-learning/eval_kermany.sh 13

Default encoder output path:
- train_model/transfer_learning/runs_kermany_seed<SEED>/encoder_kermany_pretrained.pth

### UNet Baseline (Scratch)
- sbatch cluster_scripts/transfer-learning/train_unet_tensorboard.sh 13

Saved model path:
- models/unet/baseline_scratch_seed<SEED>.pth

### UNet Transfer (Kermany)
- sbatch cluster_scripts/transfer-learning/train_unet_transfer_kermany.sh 13

By default, encoder weights are loaded from runs_kermany_seed<SEED>.
You can override encoder path with argument 2:
- sbatch cluster_scripts/transfer-learning/train_unet_transfer_kermany.sh 13 train_model/transfer_learning/runs_kermany_seed42/encoder_kermany_pretrained.pth

Saved model path:
- models/unet/kermany_transfer_seed<SEED>.pth

### UNet Transfer + Frozen Encoder
- sbatch cluster_scripts/transfer-learning/train_unet_transfer_kermany_freeze.sh 13

Saved model path:
- models/unet/kermany_transfer_frozen_seed<SEED>.pth

### UNet Test Evaluation
- transfer: sbatch cluster_scripts/transfer-learning/eval_unet_kermany.sh 13
- freeze: sbatch cluster_scripts/transfer-learning/eval_unet_kermany_freeze.sh 13

Test outputs are written to runs_unet/test_run* (from train_model/test_unet.py).
