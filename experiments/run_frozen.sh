#!/bin/bash
set -euo pipefail

conda activate attn-ppo

python scripts/eval.py \
    --condition frozen \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --n_samples 500 \
    --output outputs/frozen/eval_results.json \
    "$@"
