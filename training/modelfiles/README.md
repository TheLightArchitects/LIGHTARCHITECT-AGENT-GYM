# Light Architects Modelfiles

Per-sibling Ollama Modelfiles that achieve 80-90% of what a full fine-tune would
provide, at zero GPU cost.

## Quick Start

```bash
# Use any good base model — Mixtral, Qwen2.5, Llama 3.1, etc.
ollama pull mixtral:8x7b-instruct-v0.1-q4_K_M

# Create per-sibling models
ollama create la-corso -f modelfiles/corso.Modelfile
ollama create la-eva -f modelfiles/eva.Modelfile
ollama create la-quantum -f modelfiles/quantum.Modelfile
ollama create la-soul -f modelfiles/soul.Modelfile
ollama create la-seraph -f modelfiles/seraph.Modelfile
ollama create la-architect -f modelfiles/architect.Modelfile

# Run
ollama run la-corso "Run a security scan on this Rust function"
ollama run la-eva "Help me reflect on today's progress"
ollama run la-quantum "Investigate this incident timeline"
```

## Base Model Options

| Model | Size | VRAM | Best For |
|-------|------|------|----------|
| `mixtral:8x7b-instruct-v0.1-q4_K_M` | 26GB | 32GB | MoE, good at tool-use |
| `qwen2.5-coder:32b-instruct-q4_K_M` | 20GB | 24GB | Best coding ability |
| `qwen2.5:14b-instruct-q4_K_M` | 9GB | 12GB | Good balance |
| `llama3.1:8b-instruct-q8_0` | 8.5GB | 12GB | Fast, good general |

## When to Fine-Tune Instead

Fine-tuning adds value over Modelfiles ONLY for:
1. **Multi-expert routing** — teaching the model when to defer to another sibling
2. **Deep voice internalization** — automatic dialect without prompt reminders
3. **Tool preference patterns** — learned from 29K real trajectories

If the Modelfile results are "good enough," save the GPU budget for Phase 10
RL training where the model actually improves from feedback loops.
