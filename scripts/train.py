#!/usr/bin/env python3
"""
Main training entry point.

Usage:
  python scripts/train.py configs/bias_ppo.yaml
  python scripts/train.py configs/lora_ppo.yaml training.output_dir=outputs/lora
  python scripts/train.py configs/bias_ppo.yaml model.name=Qwen/Qwen2.5-1.5B-Instruct
"""

import logging
import sys
import os

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omegaconf import OmegaConf
from src.attn_bias.trainer import run_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/train.py <config.yaml> [key=value ...]")
        sys.exit(1)

    config_path = sys.argv[1]
    overrides = sys.argv[2:]

    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))

    run_name = cfg.training.get("wandb_run_name", None) or f"{cfg.condition}_{cfg.model.name.split('/')[-1]}"
    run_training(cfg, run_name=run_name)


if __name__ == "__main__":
    main()
