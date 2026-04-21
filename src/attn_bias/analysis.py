"""
Interpretability analysis for learned attention biases.
Run after training to understand what PPO routing changes were made.
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import torch

from .model import AttentionBias


# ──────────────────────────────────────────────────────────────
# Heatmap visualization
# ──────────────────────────────────────────────────────────────

def visualize_bias(
    attention_bias: AttentionBias,
    layer_idx: int,
    head_idx: int,
    tokens: Optional[list[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of B_h[layer, head]. Rows = query pos (attend FROM), cols = key pos (attend TO).
    """
    bias = attention_bias.biases[layer_idx][head_idx].detach().cpu().float().numpy()
    T = bias.shape[0]

    tick_labels = tokens if tokens else [str(i) for i in range(T)]
    max_ticks = 32
    if T > max_ticks:
        step = T // max_ticks
        tick_idx = list(range(0, T, step))
        x_labels = [tick_labels[i] if i < len(tick_labels) else str(i) for i in tick_idx]
    else:
        tick_idx = list(range(T))
        x_labels = tick_labels[:T]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        bias,
        cmap="RdBu_r",
        center=0,
        ax=ax,
        xticklabels=False,
        yticklabels=False,
    )
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(x_labels, rotation=90, fontsize=7)
    ax.set_yticks(tick_idx)
    ax.set_yticklabels(x_labels, rotation=0, fontsize=7)
    ax.set_title(f"Attention bias — layer {layer_idx}, head {head_idx}", fontsize=12)
    ax.set_xlabel("Key position (attend TO)")
    ax.set_ylabel("Query position (attend FROM)")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_all_heads_grid(
    attention_bias: AttentionBias,
    layer_idx: int,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Grid of all head heatmaps for a single layer."""
    n_heads = attention_bias.n_heads
    ncols = min(4, n_heads)
    nrows = (n_heads + ncols - 1) // ncols

    fig = plt.figure(figsize=(4 * ncols, 3.5 * nrows))
    gs = gridspec.GridSpec(nrows, ncols, figure=fig)

    biases = attention_bias.biases[layer_idx].detach().cpu().float().numpy()
    vmax = np.abs(biases).max()

    for h in range(n_heads):
        r, c = divmod(h, ncols)
        ax = fig.add_subplot(gs[r, c])
        sns.heatmap(biases[h], cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax, ax=ax,
                    xticklabels=False, yticklabels=False, cbar=False)
        ax.set_title(f"head {h}", fontsize=9)

    fig.suptitle(f"Attention biases — layer {layer_idx}", fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ──────────────────────────────────────────────────────────────
# Head importance ranking
# ──────────────────────────────────────────────────────────────

def head_importance(attention_bias: AttentionBias) -> list[tuple[float, int, int]]:
    """
    Rank (layer, head) pairs by Frobenius norm of learned bias, descending.
    Returns list of (norm, layer_idx, head_idx).
    """
    scores = []
    for layer_idx, bias_param in enumerate(attention_bias.biases):
        bias = bias_param.detach().cpu().float()
        for head_idx in range(bias.shape[0]):
            norm = bias[head_idx].norm(p="fro").item()
            scores.append((norm, layer_idx, head_idx))
    return sorted(scores, reverse=True)


def plot_head_importance(
    attention_bias: AttentionBias,
    top_k: int = 20,
    save_path: Optional[str] = None,
) -> plt.Figure:
    norms = attention_bias.norms().numpy()  # (n_layers, n_heads)
    n_layers, n_heads = norms.shape

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: heatmap of norms by (layer, head)
    sns.heatmap(norms, cmap="viridis", ax=axes[0], xticklabels=range(n_heads),
                yticklabels=range(n_layers))
    axes[0].set_title("||B_h||_F by (layer, head)")
    axes[0].set_xlabel("Head")
    axes[0].set_ylabel("Layer")

    # Right: bar chart of top-k heads
    ranking = head_importance(attention_bias)[:top_k]
    labels = [f"L{l}H{h}" for _, l, h in ranking]
    values = [v for v, _, _ in ranking]
    axes[1].barh(labels[::-1], values[::-1], color="steelblue")
    axes[1].set_title(f"Top-{top_k} heads by bias norm")
    axes[1].set_xlabel("||B_h||_F")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ──────────────────────────────────────────────────────────────
# Semantic analysis
# ──────────────────────────────────────────────────────────────

TOKEN_TYPES = {
    "number": r"\d",
    "operator": r"[+\-*/=]",
    "question": r"(?i)\b(what|how|find|calculate|compute|solve)\b",
    "answer_marker": r"(?i)(answer|result|therefore|so)",
    "punctuation": r"[.,;:?!]",
}


def classify_token(tok: str) -> str:
    import re
    for label, pattern in TOKEN_TYPES.items():
        if re.search(pattern, tok):
            return label
    return "other"


def semantic_analysis(
    attention_bias: AttentionBias,
    tokenizer,
    examples: list[str],
    layer_idx: int,
    head_idx: int,
    top_k_pairs: int = 20,
) -> dict:
    """
    For a given (layer, head), find the top-k (query_pos, key_pos) pairs by bias magnitude,
    then classify the tokens at those positions by type.

    Returns a dict with pair analysis and type-pair frequency counts.
    """
    import re
    from collections import Counter

    bias = attention_bias.biases[layer_idx][head_idx].detach().cpu().float()

    pair_type_counts = Counter()
    pair_details = []

    for text in examples:
        tokens = tokenizer.tokenize(text)
        T = min(len(tokens), bias.shape[0])
        b = bias[:T, :T].numpy()

        # Top-k positive entries (RL strengthened)
        flat = b.flatten()
        top_indices = np.argsort(flat)[-top_k_pairs:][::-1]
        for idx in top_indices:
            q_pos, k_pos = divmod(int(idx), T)
            q_tok = tokens[q_pos] if q_pos < len(tokens) else "<pad>"
            k_tok = tokens[k_pos] if k_pos < len(tokens) else "<pad>"
            q_type = classify_token(q_tok)
            k_type = classify_token(k_tok)
            pair_type_counts[(q_type, k_type)] += 1
            pair_details.append({
                "q_pos": q_pos, "k_pos": k_pos,
                "q_tok": q_tok, "k_tok": k_tok,
                "q_type": q_type, "k_type": k_type,
                "bias_val": float(b[q_pos, k_pos]),
            })

    return {
        "pair_type_counts": dict(pair_type_counts.most_common()),
        "top_pairs": sorted(pair_details, key=lambda x: -x["bias_val"])[:top_k_pairs],
    }


# ──────────────────────────────────────────────────────────────
# Learning curve plotting
# ──────────────────────────────────────────────────────────────

def plot_learning_curves(
    results: dict[str, dict],
    metric: str = "accuracy",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    results: {condition_name: {"steps": [...], "values": [...]}}
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"frozen": "gray", "bias_ppo": "steelblue", "lora_ppo": "darkorange", "full_ppo": "darkgreen"}

    for condition, data in results.items():
        color = colors.get(condition, "black")
        ax.plot(data["steps"], data["values"], label=condition, color=color, linewidth=2)

    ax.set_xlabel("Training steps")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"GSM8K {metric} by condition")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
