"""
PPO training loop for all conditions:
  - frozen:    no params updated (eval only, establishes baseline)
  - bias_ppo:  only AttentionBias params + value head
  - lora_ppo:  LoRA params + value head
  - full_ppo:  all model params + value head
"""

import os
import logging
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig

from .model import (
    AttentionBias,
    build_biased_model_for_ppo,
    build_frozen_model_for_ppo,
    build_lora_model_for_ppo,
    count_trainable_params,
)
from .data import load_gsm8k, collate_fn
from .rewards import gsm8k_reward_fn

log = logging.getLogger(__name__)


def _make_ppo_config(cfg, run_name: str) -> PPOConfig:
    return PPOConfig(
        model_name=cfg.model.name,
        learning_rate=cfg.ppo.learning_rate,
        batch_size=cfg.ppo.batch_size,
        mini_batch_size=cfg.ppo.mini_batch_size,
        gradient_accumulation_steps=cfg.ppo.gradient_accumulation_steps,
        ppo_epochs=cfg.ppo.ppo_epochs,
        cliprange=cfg.ppo.cliprange,
        vf_coef=cfg.ppo.vf_coef,
        kl_penalty=cfg.ppo.kl_penalty,
        target_kl=cfg.ppo.target_kl,
        max_grad_norm=cfg.ppo.max_grad_norm,
        log_with="wandb",
        tracker_project_name=cfg.training.wandb_project,
        tracker_kwargs={"name": run_name},
        seed=cfg.training.seed,
    )


def build_model_and_bias(cfg):
    """Factory: returns (model, attention_bias_or_None) for the configured condition."""
    condition = cfg.condition
    model_name = cfg.model.name
    dtype = getattr(torch, cfg.model.torch_dtype)

    if condition == "frozen":
        model = build_frozen_model_for_ppo(model_name, torch_dtype=dtype)
        return model, None

    if condition == "bias_ppo":
        model, attn_bias = build_biased_model_for_ppo(
            model_name,
            max_len=cfg.model.max_len,
            init_noise=cfg.bias.init_noise,
            shared_bias=cfg.bias.shared,
            max_norm=cfg.bias.max_norm,
            torch_dtype=dtype,
        )
        return model, attn_bias

    if condition == "lora_ppo":
        model = build_lora_model_for_ppo(
            model_name,
            lora_r=cfg.ppo.lora_r if hasattr(cfg.ppo, "lora_r") else 8,
            lora_alpha=cfg.ppo.lora_alpha if hasattr(cfg.ppo, "lora_alpha") else 32,
            lora_dropout=cfg.ppo.lora_dropout if hasattr(cfg.ppo, "lora_dropout") else 0.05,
            torch_dtype=dtype,
        )
        return model, None

    if condition == "full_ppo":
        from trl import AutoModelForCausalLMWithValueHead
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name, torch_dtype=dtype)
        return model, None

    raise ValueError(f"Unknown condition: {condition}")


def run_training(cfg, run_name: Optional[str] = None):
    if run_name is None:
        run_name = f"{cfg.condition}_{cfg.model.name.split('/')[-1]}"

    torch.manual_seed(cfg.training.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, attn_bias = build_model_and_bias(cfg)

    param_info = count_trainable_params(model)
    if attn_bias is not None:
        param_info["bias_params"] = attn_bias.param_count()
    log.info("Trainable params: %s", param_info)

    # Reference model for KL penalty (always the frozen base)
    from trl import AutoModelForCausalLMWithValueHead
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        cfg.model.name, torch_dtype=getattr(torch, cfg.model.torch_dtype)
    )
    for param in ref_model.parameters():
        param.requires_grad = False

    ppo_config = _make_ppo_config(cfg, run_name)

    # Build optimizer over trainable params only
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.ppo.learning_rate)

    dataset = load_gsm8k("train", tokenizer=tokenizer, max_prompt_len=cfg.model.max_len // 2)

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        dataset=dataset,
        data_collator=collate_fn,
    )

    generation_kwargs = {
        "max_new_tokens": cfg.training.max_new_tokens,
        "do_sample": True,
        "temperature": cfg.training.temperature,
        "top_p": cfg.training.top_p,
        "pad_token_id": tokenizer.pad_token_id,
    }

    os.makedirs(cfg.training.output_dir, exist_ok=True)
    step = 0

    for epoch in range(cfg.training.n_epochs):
        for batch in ppo_trainer.dataloader:
            # Strip left-padding so TRL doesn't mistake pad=EOS tokens for end-of-response
            query_tensors = [
                ids[mask.bool()]
                for ids, mask in zip(batch["input_ids"], batch["attention_mask"])
            ]
            gold_answers = batch["answers"]

            # Rollout — return_prompt=False gives response-only tensors
            response_tensors = ppo_trainer.generate(
                query_tensors, return_prompt=False, **generation_kwargs
            )
            responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

            rewards = [torch.tensor(r, dtype=torch.float32) for r in gsm8k_reward_fn(responses, gold_answers)]

            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

            if step % cfg.training.log_steps == 0:
                mean_reward = sum(r.item() for r in rewards) / len(rewards)
                log.info("step=%d epoch=%d reward=%.4f kl=%.4f", step, epoch, mean_reward, stats.get("objective/kl", 0))

            if step % cfg.training.save_steps == 0 and step > 0:
                _save_checkpoint(model, attn_bias, cfg, step)

            step += 1

    _save_checkpoint(model, attn_bias, cfg, step, final=True)
    return model, attn_bias


def _save_checkpoint(model, attn_bias, cfg, step: int, final: bool = False):
    tag = "final" if final else f"step_{step}"
    out = os.path.join(cfg.training.output_dir, cfg.condition, tag)
    os.makedirs(out, exist_ok=True)

    if attn_bias is not None:
        torch.save(attn_bias.state_dict(), os.path.join(out, "attention_bias.pt"))
        log.info("Saved attention bias to %s", out)

    # Save value head weights
    if hasattr(model, "v_head"):
        torch.save(model.v_head.state_dict(), os.path.join(out, "value_head.pt"))
