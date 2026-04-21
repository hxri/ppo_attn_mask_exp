#!/usr/bin/env python3
"""
Run interpretability analysis on a trained bias_ppo checkpoint.

Usage:
  python scripts/analyze.py --checkpoint outputs/bias_ppo/final --output_dir outputs/analysis
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoTokenizer

from src.attn_bias.model import build_biased_model_for_ppo, AttentionBias
from src.attn_bias.analysis import (
    visualize_bias,
    visualize_all_heads_grid,
    head_importance,
    plot_head_importance,
    semantic_analysis,
    plot_learning_curves,
)
from src.attn_bias.data import load_gsm8k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to bias_ppo checkpoint dir")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--output_dir", default="outputs/analysis")
    parser.add_argument("--n_examples", type=int, default=100, help="For semantic analysis")
    parser.add_argument("--top_k_heads", type=int, default=5, help="Heads to visualize")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, attn_bias = build_biased_model_for_ppo(args.model, max_len=args.max_len)
    bias_path = os.path.join(args.checkpoint, "attention_bias.pt")
    attn_bias.load_state_dict(torch.load(bias_path, map_location="cpu"))
    print(f"Loaded bias from {bias_path}")

    # 1. Head importance ranking
    print("\n=== Head Importance Ranking ===")
    ranking = head_importance(attn_bias)
    print(f"{'Rank':>4}  {'Norm':>10}  Layer  Head")
    for rank, (norm, layer, head) in enumerate(ranking[:20], 1):
        print(f"{rank:>4}  {norm:>10.4f}  {layer:>5}  {head:>4}")

    ranking_path = os.path.join(args.output_dir, "head_importance.json")
    with open(ranking_path, "w") as f:
        json.dump([{"norm": n, "layer": l, "head": h} for n, l, h in ranking], f, indent=2)

    fig = plot_head_importance(attn_bias, top_k=20, save_path=os.path.join(args.output_dir, "head_importance.pdf"))
    print(f"Saved head importance plot to {args.output_dir}/head_importance.pdf")

    # 2. Visualize top-k heads individually
    print(f"\n=== Visualizing top-{args.top_k_heads} heads ===")
    for rank, (norm, layer, head) in enumerate(ranking[:args.top_k_heads], 1):
        save = os.path.join(args.output_dir, f"bias_rank{rank}_L{layer}H{head}.pdf")
        visualize_bias(attn_bias, layer, head, save_path=save)
        print(f"  Rank {rank}: L{layer}H{head} (norm={norm:.4f}) → {save}")

    # 3. Per-layer head grids for top-3 layers
    layer_norms = attn_bias.norms().sum(dim=1)  # sum over heads
    top_layers = layer_norms.argsort(descending=True)[:3].tolist()
    for layer in top_layers:
        save = os.path.join(args.output_dir, f"layer_{layer}_all_heads.pdf")
        visualize_all_heads_grid(attn_bias, layer, save_path=save)
        print(f"  Layer {layer} grid → {save}")

    # 4. Semantic analysis on top head
    print(f"\n=== Semantic Analysis (top head: L{ranking[0][1]}H{ranking[0][2]}) ===")
    dataset = load_gsm8k("test", tokenizer=None)
    examples = [ex["prompt"] for ex in dataset.examples[:args.n_examples]]
    top_layer, top_head = ranking[0][1], ranking[0][2]

    sem = semantic_analysis(attn_bias, tokenizer, examples, top_layer, top_head)
    print("Token-type pair counts (query_type → key_type):")
    for (q_type, k_type), count in list(sem["pair_type_counts"].items())[:10]:
        print(f"  {q_type:15s} → {k_type:15s}  {count}")

    sem_path = os.path.join(args.output_dir, "semantic_analysis.json")
    with open(sem_path, "w") as f:
        json.dump(sem, f, indent=2)
    print(f"Saved semantic analysis to {sem_path}")


if __name__ == "__main__":
    main()
