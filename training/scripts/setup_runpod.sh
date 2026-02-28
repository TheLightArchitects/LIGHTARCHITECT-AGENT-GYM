#!/usr/bin/env bash
# RunPod A100 Environment Setup — sovereign-forging-lion Phase 7
# Run this first on a fresh RunPod A100 (40GB) instance.
#
# Usage: bash scripts/setup_runpod.sh
#
# Prerequisites:
#   - RunPod A100 40GB instance (pytorch template recommended)
#   - WANDB_API_KEY set in RunPod secrets (optional)
#   - HF_TOKEN set in RunPod secrets (required for Llama 3.1)

set -euo pipefail

echo "=== Light Architects Foundation Model — RunPod Setup ==="
echo "Phase 7: SFT Fine-Tuning (QLoRA + Unsloth)"
echo ""

# 1. System info
echo "--- System Info ---"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch not installed"
echo ""

# 2. Install Unsloth (handles torch/transformers compatibility)
echo "--- Installing Unsloth ---"
pip install --upgrade pip
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git@2025.2"
pip install --no-deps "trl<0.14.0" peft accelerate bitsandbytes

# 3. Install training dependencies
echo "--- Installing training dependencies ---"
pip install pyyaml jsonlines wandb tensorboard typer rich datasets

# 4. Verify installations
echo "--- Verification ---"
python3 -c "
import torch
import unsloth
import transformers
import trl
import peft

print(f'PyTorch:       {torch.__version__}')
print(f'CUDA avail:    {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:           {torch.cuda.get_device_name(0)}')
    print(f'VRAM:          {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print(f'Unsloth:       {unsloth.__version__}')
print(f'Transformers:  {transformers.__version__}')
print(f'TRL:           {trl.__version__}')
print(f'PEFT:          {peft.__version__}')
print()
print('All dependencies verified.')
"

# 5. Login to services (if tokens available)
if [ -n "${HF_TOKEN:-}" ]; then
    echo "--- HuggingFace Login ---"
    python3 -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"
    echo "HuggingFace: authenticated"
else
    echo "WARNING: HF_TOKEN not set. Required for Llama 3.1 (gated model)."
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "--- W&B Login ---"
    wandb login "${WANDB_API_KEY}" 2>/dev/null || true
    echo "W&B: authenticated"
else
    echo "INFO: WANDB_API_KEY not set. Training will log to tensorboard only."
fi

echo ""
echo "=== Setup Complete ==="
echo "Next: python scripts/smoke_test.py --model qwen3-8b"
