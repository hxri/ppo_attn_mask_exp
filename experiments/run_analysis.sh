#!/bin/bash
set -euo pipefail

conda activate attn-ppo

python scripts/analyze.py \
    --checkpoint outputs/bias_ppo/final \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --output_dir outputs/analysis \
    --n_examples 100 \
    --top_k_heads 5 \
    "$@"
