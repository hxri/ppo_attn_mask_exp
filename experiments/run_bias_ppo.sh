#!/bin/bash
set -euo pipefail

conda activate attn-ppo

python scripts/train.py configs/bias_ppo.yaml \
    training.output_dir=outputs \
    training.wandb_run_name=bias_ppo_qwen0.5b \
    "$@"
