import inspect
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    repeat_kv,
    apply_rotary_pos_emb,
)

# transformers 4.44 and earlier: rotary_emb(x, seq_len=int), apply_rotary_pos_emb(q, k, cos, sin, position_ids)
# transformers 4.45+:            rotary_emb(x, position_ids),  apply_rotary_pos_emb(q, k, cos, sin)
_NEW_ROPE_API = "position_ids" not in inspect.signature(apply_rotary_pos_emb).parameters


class AttentionBias(nn.Module):
    """
    Learnable additive biases on attention score matrices — one per head per layer.
    Shape per layer: (n_heads, max_len, max_len).
    All other model weights are frozen; only these parameters receive PPO gradients.
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        max_len: int = 512,
        init_noise: float = 1e-4,
        shared: bool = False,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_len = max_len
        self.shared = shared

        def _make_param():
            return nn.Parameter(
                torch.zeros(n_heads, max_len, max_len)
                + init_noise * torch.randn(n_heads, max_len, max_len)
            )

        if shared:
            # Single bias broadcast across all layers (ablation)
            shared_param = _make_param()
            self.biases = nn.ParameterList([shared_param] * n_layers)
        else:
            self.biases = nn.ParameterList([_make_param() for _ in range(n_layers)])

    def get(self, layer_idx: int, T: int) -> torch.Tensor:
        """Returns bias for layer_idx sliced to sequence length T: (n_heads, T, T)."""
        return self.biases[layer_idx][:, :T, :T]

    def param_count(self) -> int:
        if self.shared:
            return self.biases[0].numel()
        return sum(p.numel() for p in self.biases)

    def norms(self) -> torch.Tensor:
        """Frobenius norm per (layer, head): shape (n_layers, n_heads)."""
        out = torch.zeros(self.n_layers, self.n_heads)
        for layer_idx, bias in enumerate(self.biases):
            for h in range(self.n_heads):
                out[layer_idx, h] = bias[h].norm(p="fro").item()
        return out


class BiasedQwen2Attention(Qwen2Attention):
    """
    Qwen2 attention with an additive learned bias injected before softmax.
    Inherits frozen QKV/O projections; attn_bias_param is the only trainable tensor.
    """

    def __init__(self, config, layer_idx: int, attn_bias_param: nn.Parameter, max_norm: float = 0.0):
        super().__init__(config, layer_idx)
        # Registered as a buffer-like attribute, not an nn.Parameter on this module,
        # so that optimizer targeting AttentionBias.biases works cleanly.
        self.attn_bias_param = attn_bias_param
        self.max_norm = max_norm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if _NEW_ROPE_API:
            # transformers >= 4.45: position_ids baked into cos/sin
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        else:
            # transformers 4.44 and earlier: seq_len scalar, position_ids passed to apply
            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # GQA: expand kv heads to match query heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # === INJECT LEARNED ATTENTION BIAS ===
        T = attn_weights.size(-1)
        bias = self.attn_bias_param[:, :T, :T]  # (n_heads, T, T)

        if self.max_norm > 0:
            # Per-head norm clipping to stabilize training
            norms = bias.norm(p="fro", dim=(-2, -1), keepdim=True)  # (n_heads, 1, 1)
            scale = (self.max_norm / norms.clamp(min=1e-8)).clamp(max=1.0)
            bias = bias * scale

        attn_weights = attn_weights + bias.unsqueeze(0)  # broadcast over batch
        # =====================================

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def build_biased_model_for_ppo(
    model_name: str,
    max_len: int = 512,
    init_noise: float = 1e-4,
    shared_bias: bool = False,
    max_norm: float = 0.0,
    torch_dtype=torch.bfloat16,
):
    """
    Loads model, freezes all weights, injects trainable attention biases, wraps with
    trl's value head. Returns (model_with_value_head, attention_bias).

    Only attention_bias.parameters() and model.v_head.parameters() have requires_grad=True.
    """
    from trl import AutoModelForCausalLMWithValueHead

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    )

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Value head stays trainable (critic, not policy)
    for param in model.v_head.parameters():
        param.requires_grad = True

    cfg = model.pretrained_model.config
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads

    attention_bias = AttentionBias(n_layers, n_heads, max_len, init_noise, shared_bias)

    for layer_idx, layer in enumerate(model.pretrained_model.model.layers):
        biased_attn = BiasedQwen2Attention(
            cfg,
            layer_idx,
            attention_bias.biases[layer_idx],
            max_norm=max_norm,
        )
        # Copy frozen weights from original attention (already loaded)
        biased_attn.load_state_dict(layer.self_attn.state_dict(), strict=False)
        biased_attn.q_proj = layer.self_attn.q_proj
        biased_attn.k_proj = layer.self_attn.k_proj
        biased_attn.v_proj = layer.self_attn.v_proj
        biased_attn.o_proj = layer.self_attn.o_proj
        biased_attn.rotary_emb = layer.self_attn.rotary_emb
        layer.self_attn = biased_attn

    return model, attention_bias


def build_frozen_model_for_ppo(model_name: str, torch_dtype=torch.bfloat16):
    """Fully frozen model with trainable value head only (frozen baseline)."""
    from trl import AutoModelForCausalLMWithValueHead

    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name, torch_dtype=torch_dtype)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.v_head.parameters():
        param.requires_grad = True
    return model


def build_lora_model_for_ppo(
    model_name: str,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules=None,
    torch_dtype=torch.bfloat16,
):
    """LoRA baseline: low-rank updates to QKV projections + trainable value head."""
    from trl import AutoModelForCausalLMWithValueHead
    from peft import LoraConfig, get_peft_model

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj"]

    base = AutoModelForCausalLMWithValueHead.from_pretrained(model_name, torch_dtype=torch_dtype)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    base.pretrained_model = get_peft_model(base.pretrained_model, lora_config)
    return base


def count_trainable_params(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "pct": 100 * trainable / total}
