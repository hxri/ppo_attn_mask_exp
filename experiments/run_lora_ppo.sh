#!/bin/bash
set -euo pipefail

python scripts/train.py configs/lora_ppo.yaml \
    training.output_dir=outputs \
    training.wandb_run_name=lora_ppo_qwen0.5b \
    "$@"
