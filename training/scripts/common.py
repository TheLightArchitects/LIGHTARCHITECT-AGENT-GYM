"""Shared utilities for training scripts.

Centralizes config loading, path constants, and model registry
used by smoke_test, sft_train, merge_export, and dpo_train.
"""

from __future__ import annotations

from pathlib import Path

import yaml

MODEL_CONFIGS: dict[str, str] = {
    "mixtral-8x7b": "configs/mixtral_8x7b.yaml",
    "qwen3-8b": "configs/qwen3_8b.yaml",
    "llama31-8b": "configs/llama31_8b.yaml",
}

TRAINING_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = TRAINING_ROOT.parent  # mcp-agent-gym/


def load_configs(model_key: str) -> tuple[dict, dict]:
    """Load base + model-specific config.

    Returns (base_cfg, model_cfg) tuple.
    Model-specific LoRA config overrides base LoRA config (critical for MoE).
    """
    with open(TRAINING_ROOT / "configs" / "base.yaml") as f:
        base = yaml.safe_load(f)

    config_path = TRAINING_ROOT / MODEL_CONFIGS[model_key]
    with open(config_path) as f:
        model_cfg = yaml.safe_load(f)

    # MoE models override LoRA config (rank 64, explicit target modules, bf16)
    if "lora" in model_cfg:
        base["lora"] = model_cfg["lora"]

    return base, model_cfg
