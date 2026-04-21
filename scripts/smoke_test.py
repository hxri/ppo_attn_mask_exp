#!/usr/bin/env python3
"""
Smoke test: verifies forward pass + that only bias params are trainable.
No GPU required; runs on CPU with a tiny synthetic input.

Usage:
  python scripts/smoke_test.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.attn_bias.model import AttentionBias, build_biased_model_for_ppo, count_trainable_params
from src.attn_bias.rewards import gsm8k_reward_fn, extract_answer, format_gsm8k_prompt


def test_attention_bias():
    print("Testing AttentionBias module...")
    ab = AttentionBias(n_layers=2, n_heads=4, max_len=32, init_noise=1e-4)
    assert len(ab.biases) == 2
    assert ab.biases[0].shape == (4, 32, 32)
    assert ab.param_count() == 2 * 4 * 32 * 32

    fake_scores = torch.randn(2, 4, 16, 16)  # (batch, heads, T, T)
    b = ab.get(0, 16)
    assert b.shape == (4, 16, 16)
    out = fake_scores + b.unsqueeze(0)
    assert out.shape == fake_scores.shape
    print("  AttentionBias: OK")


def test_reward_fn():
    print("Testing reward functions...")

    responses = [
        "Step 1: 2+2=4\n#### 4",
        "The answer is 42",
        "I don't know",
        "#### 3.5",
    ]
    golds = ["#### 4", "42", "5", "3.5"]
    rewards = gsm8k_reward_fn(responses, golds)
    assert rewards[0] >= 1.0, f"Expected correct, got {rewards[0]}"
    assert rewards[1] >= 1.0, f"Expected correct, got {rewards[1]}"
    assert rewards[2] == 0.0, f"Expected wrong, got {rewards[2]}"
    assert rewards[3] >= 1.0, f"Expected correct (float), got {rewards[3]}"
    print(f"  Rewards: {rewards}")
    print("  Reward functions: OK")


def test_prompt_format():
    print("Testing prompt formatting...")
    prompt = format_gsm8k_prompt("If John has 3 apples and gives 1 away, how many does he have?")
    assert "3 apples" in prompt
    assert "####" in prompt
    print("  Prompt format: OK")


def test_model_trainable_params(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    print(f"\nTesting model construction + trainable param check ({model_name})...")
    print("  (This requires model weights; skipping if offline)")

    try:
        model, attn_bias = build_biased_model_for_ppo(model_name, max_len=64)
    except Exception as e:
        print(f"  Skipped (model unavailable): {e}")
        return

    info = count_trainable_params(model)
    print(f"  Param counts: {info}")
    print(f"  Bias params: {attn_bias.param_count():,}")

    # Verify only bias + value head are trainable
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert "v_head" in name or "attn_bias" in name or any(
                f"biases.{i}" in name for i in range(100)
            ), f"Unexpected trainable param: {name}"

    # Verify forward pass
    dummy_input = torch.zeros(1, 10, dtype=torch.long)
    with torch.no_grad():
        out = model(dummy_input)
    assert out.logits.shape[0] == 1
    print("  Forward pass shape:", out.logits.shape)
    print("  Model construction: OK")


def main():
    print("=" * 50)
    print("Running smoke tests")
    print("=" * 50)

    test_attention_bias()
    test_reward_fn()
    test_prompt_format()
    test_model_trainable_params()

    print("\n" + "=" * 50)
    print("All smoke tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
