#!/usr/bin/env bash
# RunPod Environment Setup — Light Architects Foundation Model
#
# VRAM Requirements:
#   Mixtral 8x7B (bf16 LoRA r=64):  ~48GB → A100 80GB recommended
#   Mixtral 8x7B (4-bit QLoRA r=16): ~24GB → A100 40GB (less stable for MoE)
#   Qwen3-8B / Llama-3.1-8B:        ~12GB → A100 40GB is fine
#
# Usage: bash scripts/setup_runpod.sh
#
# Prerequisites:
#   - RunPod instance with appropriate GPU (see VRAM table above)
#   - WANDB_API_KEY set in RunPod secrets (optional)
#   - HF_TOKEN set in RunPod secrets (required for gated models)

set -euo pipefail

echo "=== Light Architects Foundation Model — RunPod Setup ==="
echo "3-Stage SFT + DPO Pipeline (Unsloth + LoRA)"
echo ""

# 1. System info
echo "--- System Info ---"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch not installed"

# Check VRAM for Mixtral
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
if [ "$VRAM_MB" -lt 40000 ]; then
    echo ""
    echo "WARNING: GPU has ${VRAM_MB}MB VRAM."
    echo "  Mixtral 8x7B bf16 LoRA needs ~48GB (A100 80GB recommended)."
    echo "  Dense 8B models need ~12GB and will work fine."
    echo ""
fi
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
    vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'VRAM:          {vram_gb:.1f} GB')
    if vram_gb >= 70:
        print(f'Recommended:   Mixtral 8x7B (bf16 LoRA r=64) — OPTIMAL')
    elif vram_gb >= 35:
        print(f'Recommended:   Mixtral 8x7B (4-bit QLoRA) or Dense 8B (bf16 LoRA)')
    else:
        print(f'Recommended:   Dense 8B models only (Qwen3-8B, Llama-3.1-8B)')
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
    echo "WARNING: HF_TOKEN not set. Required for gated models (Llama 3.1, Mixtral)."
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
echo ""
echo "Training Pipeline (L-ARC v1 — Qwen3-8B primary):"
echo "  1. Dry run:     python scripts/sft_train.py train --model qwen3-8b --stage 1 --dry-run"
echo "  2. Stage 1 SFT: python scripts/sft_train.py train --model qwen3-8b --stage 1"
echo "  3. Stage 2 SFT: python scripts/sft_train.py train --model qwen3-8b --stage 2"
echo "  4. Stage 3 SFT: python scripts/sft_train.py train --model qwen3-8b --stage 3"
echo "  5. Merge:       python scripts/merge_export.py merge --model qwen3-8b --stage 3"
echo "  6. DPO:         python scripts/dpo_train.py train --model qwen3-8b"
echo "  7. Export GGUF:  python scripts/merge_export.py gguf --model qwen3-8b"
echo ""
echo "Quick check:  python scripts/sft_train.py info --model qwen3-8b"
