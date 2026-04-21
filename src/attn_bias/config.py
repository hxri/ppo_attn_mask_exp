from dataclasses import dataclass, field
from typing import Literal, Optional
from omegaconf import MISSING


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_len: int = 512
    torch_dtype: str = "bfloat16"


@dataclass
class BiasConfig:
    # Bias-specific init scale; 0 = pure zeros, >0 = small noise
    init_noise: float = 1e-4
    # If True, share one bias across all layers (ablation)
    shared: bool = False
    # Clip bias norm per head during forward pass (0 = no clip)
    max_norm: float = 0.0


@dataclass
class PPOTrainConfig:
    learning_rate: float = 1e-4
    batch_size: int = 32
    mini_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    ppo_epochs: int = 4
    cliprange: float = 0.2
    vf_coef: float = 0.1
    kl_penalty: str = "kl"
    target_kl: float = 0.05
    max_grad_norm: float = 1.0
    # Value head learning rate (separate from bias LR)
    vf_lr: float = 1e-4


@dataclass
class LoRAPPOConfig(PPOTrainConfig):
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj"])


@dataclass
class TrainingConfig:
    n_epochs: int = 3
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    seed: int = 42
    # Logging
    log_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 50
    output_dir: str = "outputs"
    wandb_project: str = "attn-bias-ppo"
    wandb_run_name: str = MISSING


@dataclass
class ExperimentConfig:
    condition: Literal["frozen", "bias_ppo", "lora_ppo", "full_ppo"] = MISSING
    model: ModelConfig = field(default_factory=ModelConfig)
    bias: BiasConfig = field(default_factory=BiasConfig)
    ppo: PPOTrainConfig = field(default_factory=PPOTrainConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: str = "gsm8k"
    n_eval_samples: int = 500
