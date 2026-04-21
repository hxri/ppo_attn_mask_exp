# Attention Bias PPO (AB-PPO)

**Thesis**: RL fine-tuning of LLMs can be done entirely by learning additive biases on attention score matrices, with all transformer weights frozen. This is a *routing-only* update — it changes *who tokens listen to*, not *what they mean*.

In standard attention:
```
A_h = softmax( Q_h K_h^T / sqrt(d) ) * V_h
```

AB-PPO adds a learned bias **B_h** of shape `(n_heads, T, T)`:
```
A_h = softmax( Q_h K_h^T / sqrt(d) + B_h ) * V_h
```

**B_h is the only trainable parameter.** Everything else — W_Q, W_K, W_V, W_O, W_ff, embeddings — is completely frozen. PPO gradients flow only through B_h.

---

## Why This Is Interesting

This is not just parameter-efficient fine-tuning:

| Method | What changes | Interpretable? |
|---|---|---|
| Full PPO | All weights | No |
| LoRA PPO | Low-rank projections of W_Q, W_K, etc. | No |
| Prefix tuning | Prepended KV pairs | Partially |
| **AB-PPO (ours)** | **Attention routing only** | **Yes — directly** |

LoRA still changes what tokens *are* (their representations). AB-PPO only changes who they *listen to*. The inductive bias is: **RL fine-tuning is a routing problem, not a representation problem.**

If AB-PPO achieves ≥ 70% of full PPO accuracy → RL fine-tuning is primarily routing.  
If it fails → representational change is essential.  
Either outcome is publishable. The interpretability heatmaps are a contribution regardless.

---

## Setup

```bash
conda create -n attn-ppo python=3.11
conda activate attn-ppo

pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .

# Pre-download model and dataset (optional, avoids timeouts during training)
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')"
python -c "from datasets import load_dataset; load_dataset('gsm8k', 'main')"
```

**Hardware**: Single A100 80GB is ideal. V100 32GB works with reduced batch size. Qwen2.5-0.5B in bfloat16 ≈ 1 GB model weights; bias params are negligible. Most memory is KV cache during rollout.

---

## Running Experiments

### Smoke test (no GPU required)

Validates the core logic — `AttentionBias` forward pass, reward functions, prompt formatting — without loading model weights:

```bash
python scripts/smoke_test.py
```

### Frozen baseline

Establishes the zero-shot GSM8K accuracy of the unmodified model:

```bash
python scripts/eval.py \
    --condition frozen \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --n_samples 500 \
    --output outputs/frozen/eval_results.json
```

Or via the experiment script:

```bash
bash experiments/run_frozen.sh
```

### AB-PPO (main experiment)

```bash
python scripts/train.py configs/bias_ppo.yaml
```

Override any config key from the CLI:

```bash
python scripts/train.py configs/bias_ppo.yaml \
    model.name=Qwen/Qwen2.5-1.5B-Instruct \
    bias.max_norm=5.0 \
    ppo.target_kl=0.1 \
    training.output_dir=outputs/ablation
```

### LoRA PPO baseline

```bash
python scripts/train.py configs/lora_ppo.yaml
```

### Full PPO upper bound

```bash
python scripts/train.py configs/full_ppo.yaml
```

### Evaluate a trained checkpoint

```bash
python scripts/eval.py \
    --condition bias_ppo \
    --checkpoint outputs/bias_ppo/final \
    --n_samples 500

# Compare all conditions at once
python scripts/eval.py --compare --output_dir outputs
```

---

## Interpretability Analysis

After training, generate heatmaps and head importance rankings:

```bash
python scripts/analyze.py \
    --checkpoint outputs/bias_ppo/final \
    --output_dir outputs/analysis \
    --top_k_heads 5 \
    --n_examples 100
```

Outputs (all saved to `--output_dir`):
- `head_importance.pdf` — Frobenius norm heatmap by (layer, head) + top-20 bar chart
- `bias_rank{N}_L{layer}H{head}.pdf` — heatmap for each of the top-k heads
- `layer_{N}_all_heads.pdf` — grid of all head heatmaps for the most-moved layers
- `head_importance.json` — full ranking as JSON
- `semantic_analysis.json` — token-type pair counts for the top head

---

## Project Structure

```
ppo-attn-mask/
├── src/attn_bias/
│   ├── model.py       # AttentionBias, BiasedQwen2Attention, model builders
│   ├── config.py      # Typed dataclasses for all hyperparameters
│   ├── rewards.py     # GSM8K answer extraction and reward function
│   ├── data.py        # GSM8KDataset, collate_fn, prompt formatting
│   ├── trainer.py     # PPO training loop (all four conditions)
│   ├── eval.py        # pass@1 evaluation on test split
│   └── analysis.py    # Heatmaps, head importance, semantic token analysis
├── configs/
│   ├── bias_ppo.yaml  # Main experiment
│   ├── frozen.yaml    # Zero-shot baseline
│   ├── lora_ppo.yaml  # LoRA rank-8 on Q/K/V
│   └── full_ppo.yaml  # Full fine-tuning upper bound
├── scripts/
│   ├── train.py       # Training entry point
│   ├── eval.py        # Evaluation
│   ├── analyze.py     # Interpretability analysis
│   └── smoke_test.py  # Offline unit tests
└── experiments/
    ├── run_bias_ppo.sh
    ├── run_frozen.sh
    ├── run_lora_ppo.sh
    └── run_analysis.sh
```

---

## Key Hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `model.max_len` | 512 | Bias tensor size is `n_heads × max_len²` |
| `bias.init_noise` | 1e-4 | Perturbation from zero-init |
| `bias.max_norm` | 0.0 | Per-head norm clip; 0 = off. Try 5.0 if training is unstable |
| `ppo.learning_rate` | 1e-4 | Higher than full PPO is fine — param space is tiny |
| `ppo.target_kl` | 0.05 | Tight to stay close to base model; loosen to 0.1 if no learning signal |
| `ppo.vf_coef` | 0.1 | Value head loss weight |
| `bias.shared` | false | Ablation: share one bias across all layers |

---

## Baselines and Expected Results

| Condition | Trainable params | Expected GSM8K accuracy |
|---|---|---|
| Frozen | 0 | ~40–50% (Qwen2.5-0.5B-Instruct baseline) |
| AB-PPO (ours) | ~n_layers × n_heads × T² ≈ 30M | TBD — the experiment |
| LoRA PPO (rank 8) | ~6M | ~60–70% |
| Full PPO | ~500M | ~70–80% |

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| No learning signal (KL too tight) | Loosen `target_kl` to 0.1; reduce `ppo.kl_penalty` weight |
| Bias explodes | Set `bias.max_norm=5.0` to clip per-head; reduce learning rate |
| Bias too constrained to express useful routing | Reduce sequence length (`max_len=256`); check if T² param count is sufficient |
| Qwen2 attention API changes between versions | Pin `transformers==4.44.0`; fall back to GPT-2 for prototyping if needed |

---

## Extensions

- **Bias surgery**: Zero out all but the top-3 heads by norm. Does performance hold? Identifies the minimal RL circuit.
- **Cross-task transfer**: Train B_h on math, evaluate zero-shot on code. Do routing patterns generalize?
- **Combining with LoRA**: Bias PPO for routing + LoRA for representation — do they interact additively?
- **Scaling**: Does the fraction of full-PPO accuracy that AB-PPO achieves go up or down with model size?
- **Reward hacking analysis**: Does bias PPO produce qualitatively different failure modes than full PPO?
