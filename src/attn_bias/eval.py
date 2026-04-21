"""Evaluation utilities: run GSM8K pass@1 accuracy on a model."""

import logging
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from .data import load_gsm8k, collate_fn
from .rewards import gsm8k_reward_fn

log = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_gsm8k(
    model,
    tokenizer,
    n_samples: int = 500,
    max_new_tokens: int = 256,
    temperature: float = 0.0,   # greedy by default for eval
    batch_size: int = 8,
    device: Optional[str] = None,
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    if hasattr(model, "pretrained_model"):
        model.pretrained_model.eval()

    dataset = load_gsm8k("test", tokenizer=tokenizer, max_prompt_len=256)
    dataset.examples = dataset.examples[:n_samples]

    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    correct = 0
    total = 0
    all_rewards = []

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else 1.0,
        "pad_token_id": tokenizer.pad_token_id,
    }

    for batch in tqdm(loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        out = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)
        # Decode only new tokens
        new_tokens = out[:, input_ids.shape[1]:]
        responses = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        rewards = gsm8k_reward_fn(responses, batch["answers"])
        all_rewards.extend(rewards)
        correct += sum(1 for r in rewards if r >= 1.0)
        total += len(rewards)

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "mean_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
    }
