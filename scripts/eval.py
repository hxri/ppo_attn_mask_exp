#!/usr/bin/env python3
"""
Evaluate a trained checkpoint on GSM8K.

Usage:
  # Evaluate bias_ppo checkpoint
  python scripts/eval.py --condition bias_ppo --checkpoint outputs/bias_ppo/final

  # Evaluate frozen baseline (no checkpoint needed)
  python scripts/eval.py --condition frozen --model Qwen/Qwen2.5-0.5B-Instruct

  # Compare all conditions
  python scripts/eval.py --compare --output_dir outputs
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoTokenizer

from src.attn_bias.model import (
    build_biased_model_for_ppo,
    build_frozen_model_for_ppo,
    AttentionBias,
)
from src.attn_bias.eval import evaluate_gsm8k


def load_checkpoint(condition: str, model_name: str, checkpoint_dir: str, max_len: int = 512):
    dtype = torch.bfloat16

    if condition == "frozen":
        model = build_frozen_model_for_ppo(model_name, torch_dtype=dtype)
        return model, None

    if condition == "bias_ppo":
        model, attn_bias = build_biased_model_for_ppo(model_name, max_len=max_len, torch_dtype=dtype)
        bias_path = os.path.join(checkpoint_dir, "attention_bias.pt")
        if os.path.exists(bias_path):
            attn_bias.load_state_dict(torch.load(bias_path, map_location="cpu"))
            print(f"Loaded attention bias from {bias_path}")
        else:
            print(f"Warning: no bias checkpoint at {bias_path}, using init weights")
        return model, attn_bias

    if condition == "lora_ppo":
        from src.attn_bias.model import build_lora_model_for_ppo
        from peft import PeftModel
        model = build_lora_model_for_ppo(model_name, torch_dtype=dtype)
        if checkpoint_dir and os.path.exists(checkpoint_dir):
            model.pretrained_model = PeftModel.from_pretrained(model.pretrained_model, checkpoint_dir)
        return model, None

    raise ValueError(f"Unknown condition: {condition}")


def eval_single(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, _ = load_checkpoint(args.condition, args.model, args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    results = evaluate_gsm8k(
        model, tokenizer,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        device=device,
    )
    results["condition"] = args.condition
    print(json.dumps(results, indent=2))

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
    return results


def eval_compare(args):
    conditions = ["frozen", "bias_ppo", "lora_ppo"]
    all_results = {}

    for cond in conditions:
        ckpt = os.path.join(args.output_dir, cond, "final")
        if not os.path.exists(ckpt) and cond != "frozen":
            print(f"Skipping {cond}: no checkpoint at {ckpt}")
            continue

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model, _ = load_checkpoint(cond, args.model, ckpt)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        res = evaluate_gsm8k(model, tokenizer, n_samples=args.n_samples, device=device)
        res["condition"] = cond
        all_results[cond] = res
        print(f"{cond}: accuracy={res['accuracy']:.3f}")

        del model
        torch.cuda.empty_cache()

    summary_path = os.path.join(args.output_dir, "eval_comparison.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved comparison to {summary_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["frozen", "bias_ppo", "lora_ppo", "full_ppo"])
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output", default="")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    if args.compare:
        eval_compare(args)
    else:
        if not args.condition:
            parser.error("--condition required unless --compare")
        eval_single(args)


if __name__ == "__main__":
    main()
