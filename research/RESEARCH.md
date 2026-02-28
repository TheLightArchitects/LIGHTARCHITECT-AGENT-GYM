# sovereign-forging-lion: Canonical Research Document

> All research conducted during the sovereign-forging-lion build cycle is documented here with IEEE citations, source links, and verbatim excerpts that inform design decisions.

**Plan ID**: sovereign-forging-lion
**Created**: 2026-02-22
**Last Updated**: 2026-02-22
**Author**: Kevin Francis Tan

---

## Table of Contents

1. [Base Model Evaluation](#1-base-model-evaluation)
2. [Fine-Tuning Frameworks](#2-fine-tuning-frameworks)
3. [RL for Tool-Use](#3-rl-for-tool-use)
4. [Biblical Training Data](#4-biblical-training-data)
5. [Edge Deployment](#5-edge-deployment)
6. [Consciousness Telemetry](#6-consciousness-telemetry)
7. [MCP Gym Design](#7-mcp-gym-design)
8. [References](#8-references)

---

## 1. Base Model Evaluation

### 1.1 Evaluation Criteria

Models are evaluated across 8 dimensions for this project:

| Dimension | Weight | Rationale |
|-----------|--------|-----------|
| Context Window | High | Long agent traces require extended context |
| Reasoning (MMLU, HellaSwag) | High | Tool-use requires multi-step reasoning |
| Code Generation (HumanEval) | Medium | MCP JSON generation requires code-like precision |
| Tool-Use Capability | Critical | Primary use case — MCP server navigation |
| Unsloth Support | Critical | Required for budget-constrained QLoRA training |
| RKLLM NPU Support | High | Edge deployment to Khadas Edge 2 Pro |
| License | Medium | Must permit fine-tuning and redistribution |
| Community / Research Backing | Medium | Proven stability, active development |

### 1.2 Candidate Models

#### 1.2.1 Meta Llama 3.1 8B Instruct

**Source**: Meta Platforms, Inc., "Llama 3.1 Model Card," GitHub, Jul. 2024 [1].
**URL**: https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md

**Architecture**:

| Attribute | Value |
|-----------|-------|
| Parameters | 8B |
| Context Length | **128K** |
| GQA | Yes |
| Pretraining Tokens | 15T+ |
| Knowledge Cutoff | December 2023 |
| Supported Languages | English, German, French, Italian, Portuguese, Hindi, Spanish, Thai |

> "Llama 3.1 is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align with human preferences for helpfulness and safety." [1]

**Benchmarks** (0-shot unless noted, from MODEL_CARD.md [1]):

| Benchmark | Score | Notes |
|-----------|-------|-------|
| MMLU | 73.0 | 0-shot CoT |
| MMLU | 69.4 | 5-shot |
| HumanEval | 72.6 | pass@1 |
| ARC-Challenge | 83.4 | 0-shot |
| GSM8K (CoT) | 84.5 | 8-shot, em_maj1@1 |
| IFEval | 80.4 | — |
| API-Bank | 82.6 | 0-shot |
| BFCL | 76.1 | 0-shot |
| Nexus (0-shot) | 38.5 | macro_avg/acc |

**Note**: HellaSwag is NOT included in Meta's official MODEL_CARD.md benchmark tables for Llama 3.1. The score may exist in the arXiv paper (2407.21783) [2] but was not verified from primary source.

**Tool-Use**: Llama 3.1 has native function calling support via built-in tools (brave_search, wolfram_alpha, code interpreter) and zero-shot user-defined tools. BFCL v2 score: 65.4 (overall_ast_summary/macro_avg/valid) [1]. Nexus: 38.5 [1].

**8B Model Tool-Calling Limitation** (from Meta's official prompt format docs):

> "Llama 8B-Instruct can not reliably maintain a conversation alongside tool calling definitions. It can be used for zero-shot tool calling, but tool instructions should be removed for regular conversations between the model and the user." [1]

This is a significant constraint: the 8B variant cannot maintain mixed conversation + tool calling, while 70B and 405B handle it reliably. For a foundation model focused on MCP tool orchestration, this limitation reduces Llama 3.1 8B's suitability compared to Qwen 2.5 7B which was explicitly optimized for structured output and function calling.

**License**: Llama 3.1 Community License Agreement. Permits commercial use, fine-tuning, and redistribution. Requires "Built with Llama" attribution and "Llama" prefix on derivative model names. Separate license required if >700M MAU [3].

> "You are granted a non-exclusive, worldwide, non-transferable and royalty-free limited license under Meta's intellectual property or other rights owned by Meta embodied in the Llama Materials to use, reproduce, distribute, copy, create derivative works of, and make modifications to the Llama Materials." [3]

**Unsloth**: Fully supported. Pre-quantized models available: `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` [4].

> "Unsloth makes Llama 3.1 (8B) finetuning **2.1x faster** [...] use **60% less memory** than Flash Attention 2 (FA2) + Hugging Face (HF)" [4]

**RKLLM**: LLAMA architecture officially supported by RKLLM toolkit [5]. Community-converted W8A8 model exists: `jamescallander/Llama-3.1-8B-Instruct_w8a8_g128_rk3588.rkllm` [6]. No official tok/s benchmark for 8B on RK3588 — largest official benchmark is ChatGLM3 6B at 4.94 tok/s [7].

#### 1.2.2 Phi-3.5 Mini (3.8B)

**Source**: Microsoft, "Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone," arXiv:2404.14219, Apr. 2024 [8].
**URL**: https://huggingface.co/microsoft/Phi-3.5-mini-instruct

**Architecture**:

| Attribute | Value |
|-----------|-------|
| Parameters | 3.8B |
| Context Length | **128K** |
| Layers | 32 |
| Hidden Dim | 3,072 |
| Attention Heads | 32 |
| KV Heads | 8 (GQA) |
| Vocab Size | 32,064 |

**Benchmarks** (from HuggingFace Model Card [8]):

| Benchmark | Score | Notes |
|-----------|-------|-------|
| MMLU | 69.0 | 5-shot |
| HellaSwag | ~76.7 | 5-shot (from Phi-3 report, not updated for 3.5) |
| HumanEval | 62.8 | pass@1 |
| ARC-Challenge | 84.6 | 25-shot |
| GSM8K | 86.2 | 8-shot CoT |
| MBPP | 69.6 | pass@1 |
| BigBench Hard (CoT) | 69.0 | — |

**Tool-Use**: No official BFCL function-calling scores published. Phi-3 handlers exist in BFCL codebase but no Microsoft-published score found.

**License**: **MIT License** — unrestricted commercial use, modification, redistribution. Strictly most permissive of all candidates.

**Unsloth**: Supported. `unsloth/Phi-3.5-mini-instruct` available [9].

**RKLLM**: Phi architecture supported ("Phi2/Phi3" listed). Phi3 3.8B benchmarked at **7.50 tok/s** on RK3588, 3,748 MB memory [7].

**Strengths**: Exceptional parameter efficiency — GSM8K 86.2 and ARC-C 84.6 competitive with models 2-3x its size. 128K context. MIT license. ~2.2 GB at Q4.
**Weaknesses**: Lower MMLU (69.0), no native function-calling benchmarks, 3.8B capacity limits fine-tuning absorption.

#### 1.2.3 Mistral 7B v0.3 Instruct

**Source**: Jiang, A.Q. et al., "Mistral 7B," arXiv:2310.06825, Oct. 2023 [10].
**URL**: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3

**Architecture**:

| Attribute | Value |
|-----------|-------|
| Parameters | ~7.3B |
| Context Length | **32K** (v0.2+) |
| Layers | 32 |
| Hidden Dim | 4,096 |
| Attention Heads | 32 |
| KV Heads | 8 (GQA) |
| Vocab Size | 32,768 (extended in v0.3 for function calling tokens) |

> "The Mistral-7B-v0.3 Large Language Model (LLM) is a Mistral-7B-v0.2 with extended vocabulary." [10]

**Benchmarks** (base v0.1 from paper [10], v0.3 Instruct from community evals):

| Benchmark | v0.1 Base | v0.3 Instruct | Notes |
|-----------|-----------|---------------|-------|
| MMLU | 62.5% | 62.6 | 5-shot |
| HellaSwag | 81.3% | 84.8 | 0-shot |
| ARC-Challenge | ~78% | 63.9 | Different eval setups |
| HumanEval | ~30.5% | — | 0-shot, approaches Code-Llama 7B |
| GSM8K | ~35.4% | 42.2 | 8-shot, maj@8 |

**Tool-Use**: **YES** — Mistral 7B v0.3 natively supports function calling via extended vocabulary tokens: `TOOL_CALLS`, `AVAILABLE_TOOLS`, `TOOL_RESULTS`.

**License**: **Apache 2.0** — unrestricted commercial use.

**Unsloth**: Supported. `unsloth/mistral-7b-instruct-v0.3-bnb-4bit` available [9].

**RKLLM**: **NOT SUPPORTED**. Mistral is absent from the official RKLLM supported models list [5]. While architecturally similar to LLaMA, distinct differences (GQA config, vocab size, intermediate size) prevent conversion through the LLAMA path. **This is a blocking issue for edge deployment.**

#### 1.2.4 Gemma 2 9B

**Source**: Google DeepMind, "Gemma 2: Improving Open Language Models at a Practical Size," arXiv:2408.00118, Jul. 2024 [11].
**URL**: https://huggingface.co/google/gemma-2-9b-it

**Architecture**:

| Attribute | Value |
|-----------|-------|
| Parameters | ~9.24B |
| Context Length | **8,192** |
| Layers | 42 |
| Hidden Dim | 3,584 |
| Attention Heads | 16 |
| KV Heads | 8 (GQA) |
| Vocab Size | **256,000** |
| Special | Interleaved local sliding window (4K) + global attention (8K), logit soft-capping |
| Training | Knowledge distillation from larger teacher model |

**Benchmarks** (pretrained, from technical report [11]):

| Benchmark | Score | Notes |
|-----------|-------|-------|
| MMLU | 71.3 | 5-shot, top-1 |
| HellaSwag | 81.9 | 10-shot |
| HumanEval | 40.2 | pass@1 (third-party eval, verify for IT variant) |
| ARC-Challenge | 68.4 | 25-shot |
| GSM8K | 68.6 | 5-shot |

**Tool-Use**: No official function-calling benchmarks published.

**License**: Gemma Terms of Use — custom Google license. Permits commercial fine-tuning and redistribution but requires downstream notice propagation and Section 3.2 use restrictions compliance. Not OSI-approved [12].

**Unsloth**: Supported. `unsloth/gemma-2-9b-it-bnb-4bit` available [9].

**RKLLM**: "Gemma2/3" explicitly listed as supported [5]. However, non-standard architecture (interleaved attention, logit soft-capping at 30.0/50.0, 256K vocab) may introduce edge cases. Should be verified through actual conversion testing.

**Strengths**: Highest MMLU (71.3) and HellaSwag (81.9). 256K vocab for better multilingual coverage. 9B capacity for absorbing fine-tuning data.
**Weaknesses**: **8K context window** (severely limiting vs 128K). Weak HumanEval (40.2). Weak GSM8K (68.6). Custom license. ~5.7 GB at Q4 — 2.5x more VRAM than Phi-3.5.

#### 1.2.5 Qwen 2.5 7B Instruct

**Source**: Qwen Team, "Qwen2.5: A Party of Foundation Models," Sep. 2024 [37].
**URL**: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
**Technical Report**: A. Yang et al., "Qwen2.5 Technical Report," arXiv:2412.15115, Dec. 2024 [38].

**Architecture**:

| Attribute | Value |
|-----------|-------|
| Parameters | 7.61B (6.53B non-embedding) |
| Context Length | **128K** (131,072) |
| Max Generation Length | 8,192 tokens |
| Layers | 28 |
| Attention Heads | 28 Q-heads, 4 KV-heads (GQA) |

> "Significantly more knowledge and has greatly improved capabilities in coding and mathematics [...] Significant improvements in instruction following, generating long texts (over 8K tokens), understanding structured data (e.g, tables), and generating structured outputs especially JSON." [37]

**Benchmarks** (from HuggingFace Model Card [37]):

| Benchmark | Score | Notes |
|-----------|-------|-------|
| MMLU | **74.2** | Highest of all candidates |
| HumanEval | **84.8** | Instruction-tuned, highest of all candidates |
| MATH | 75.5 | — |
| BFCL v4 | **44.7** | Berkeley Function Calling Leaderboard |

**Tool-Use**: **YES** — native function-calling support. BFCL v4 score of 44.7 for the 7B instruction-tuned variant. The Qwen-Agent framework provides built-in tool-use orchestration.

**License**: **Apache 2.0** — unrestricted commercial use, modification, redistribution. No MAU limits, no attribution naming requirements.

**Unsloth**: Supported. `unsloth/Qwen2.5-7B-Instruct` available [9].

**RKLLM**: "Qwen2/Qwen2.5/Qwen3" explicitly listed as supported [5]. No official 7B benchmark, but Qwen2.5 1.5B benchmarked at 16.32 tok/s. Extrapolation for 7B W8A8: ~4-5 tok/s, ~6-7 GB memory.

**Strengths**: Highest MMLU (74.2) and HumanEval (84.8) of all candidates. Native function-calling with BFCL benchmark. 128K context. Apache 2.0 license. Strong structured output (JSON) generation.
**Weaknesses**: No official RK3588 benchmark for 7B size. Newer model with less community fine-tuning precedent than Llama 3.1.

#### 1.2.6 Emerging Alternatives (2025+)

Models released after the primary candidates that may warrant future evaluation:

| Model | Params | Context | HumanEval | License | RKLLM | Notes |
|-------|--------|---------|-----------|---------|-------|-------|
| **Phi-4-mini** (Feb 2025) | 3.8B | 128K | 74.4 | MIT | TBD (Phi3 arch) | Direct Phi-3.5-mini successor, 200K vocab, GQA [39] |
| **Qwen3 8B** (2025) | 8B | 128K | TBD | Apache 2.0 | Yes (Qwen arch) | Built-in agent/tool-use, Qwen-Agent framework [40] |
| **SmolLM3 3B** (Jul 2025) | 3B | 128K | TBD | Apache 2.0 | TBD | MMLU 68.9, YARN extrapolation [41] |

### 1.3 Comparison Matrix

| Dimension | Qwen 2.5 7B | Llama 3.1 8B | Phi-3.5 Mini 3.8B | Mistral 7B v0.3 | Gemma 2 9B |
|-----------|-------------|-------------|-------------------|-----------------|-----------|
| **MMLU** | **74.2** | 73.0 (CoT) / 69.4 | 69.0 | 62.6 | 71.3 |
| **HellaSwag** | Not reported | Not reported | ~76.7 | **84.8** | 81.9 |
| **HumanEval** | **84.8** | 72.6 | 62.8 | ~30.5 | 40.2 |
| **ARC-C** | Not reported | 83.4 | **84.6** | 63.9 | 68.4 |
| **GSM8K** | Not reported | 84.5 | **86.2** | 42.2 | 68.6 |
| **MATH** | **75.5** | — | 48.5 | — | 36.6 |
| **BFCL v2 (tool-use)** | 44.7 | 65.4 | None | Native (no score) | None |
| **Tool-call + chat** | **Yes** | **No (8B limitation)** | Not tested | Yes | Not tested |
| **Context Window** | **128K** | **128K** | **128K** | 32K | 8K |
| **Parameters** | 7.61B | 8B | 3.8B | 7.3B | 9.24B |
| **Vocab Size** | 151,936 | 128K | 32K | 32.7K | **256K** |
| **License** | **Apache 2.0** | Llama Community | **MIT** | Apache 2.0 | Gemma ToU |
| **Unsloth** | Yes | Yes | Yes | Yes | Yes |
| **RKLLM NPU** | **Yes (official)** | Yes (community) | **Yes (official, 7.5 tok/s)** | **NO** | Yes (verify) |
| **VRAM (Q4)** | ~5 GB | ~5 GB | **~2.2 GB** | ~4.5 GB | ~5.7 GB |

### 1.4 Selection Recommendation

#### Hardware Constraints

Kevin's local hardware profile:

| Device | RAM | Compute | Constraint |
|--------|-----|---------|------------|
| **Khadas Edge 2 Pro** | 16 GB LPDDR4X | RK3588S NPU (6 TOPS) | W8A8 only, largest benchmarked: 6B |
| **Mac** | — | CPU/GPU via llama.cpp/Ollama | Comfortable with models up to ~13B |

Models must run locally without overloading hardware. This favors smaller models and proven RKLLM compatibility.

#### Two-Model Strategy

**Training Target: Qwen 2.5 7B Instruct** — Best code/tool-use scores of all candidates (HumanEval 84.8, MMLU 74.2), native function-calling with BFCL benchmark (44.7), 128K context, Apache 2.0 license (most permissive for commercial use), official RKLLM support for Qwen2.5 architecture. Strong structured JSON output generation — critical for MCP tool calls. Fine-tune via Unsloth QLoRA on cloud GPU (~$40-60 for A100 training run).

**Edge Deployment: Phi-3.5 Mini 3.8B** (or Phi-4-mini when RKLLM-verified) — Official RKLLM benchmark at 7.5 tok/s, 3,748 MB on RK3588. Leaves 8+ GB free on the 16 GB Khadas. MIT license. Strong reasoning relative to size (GSM8K 86.2). Can be separately fine-tuned with the same dataset via Unsloth. Runs comfortably on Mac via Ollama (~2.2 GB at Q4).

**Comparison to Previous Recommendation**: Qwen 2.5 7B replaces Llama 3.1 8B as primary training target because:
- HumanEval: 84.8 vs 72.6 (+17% — critical for tool-call JSON generation)
- MMLU: 74.2 vs 73.0 (+1.6%)
- License: Apache 2.0 vs Llama Community (no "Built with Llama" requirement, no MAU threshold)
- RKLLM: Official architecture support vs community-only conversion
- JSON generation: Explicitly optimized for structured output

Llama 3.1 8B remains a viable fallback — stronger BFCL (76.1 vs 44.7) and API-Bank (82.6) scores indicate better *existing* tool-use, but Qwen 2.5's superior code generation and JSON capabilities suggest it will fine-tune more effectively for *new* MCP tool patterns.

#### Eliminated Models

**Mistral 7B v0.3** — RKLLM not supported. Despite Apache 2.0 license and native function calling, blocking NPU incompatibility eliminates it for edge deployment.

**Gemma 2 9B** — 8K context window severely limiting for agent traces. Non-standard architecture creates RKLLM risk. Custom license adds compliance overhead. Weak code generation (HumanEval 40.2).

#### Hardware Feasibility Summary

| Model | Khadas Edge 2 Pro (RK3588S, 16GB) | Mac (Ollama) |
|-------|-----------------------------------|--------------|
| **Phi-3.5 Mini 3.8B** | 7.5 tok/s, 3.7 GB — comfortable | Effortless (~2.2 GB Q4) |
| **Qwen 2.5 7B** | Est. ~4-5 tok/s, ~6-7 GB — feasible | Comfortable (~5 GB Q4) |
| **Llama 3.1 8B** | Est. ~3-4 tok/s, ~7-8 GB — tight | Comfortable (~5 GB Q4) |

*(Selection approved by Kevin — Phase 1 complete)*

---

## 2. Fine-Tuning Frameworks

### 2.1 Unsloth + QLoRA

**Source**: Unsloth AI, "Unsloth: Finetune LLMs 2-5x faster with 80% less memory," GitHub, 2024 [13].
**URL**: https://github.com/unslothai/unsloth

> "Unsloth makes finetuning LLMs like Llama-3.1, Mistral, Phi-3.5 & Gemma 2x faster with 80% less memory! QLoRA and LoRA finetuning." [13]

**Key configuration from Unsloth documentation** [13]:

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=4096,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)
```

**Performance** (from Unsloth blog for Llama 3.1 8B [4]):
- QLoRA with rank=32, all linear layers, batch size 1
- **2.1x faster** than HF + Flash Attention 2
- **60% less memory** — fits in 8 GB with Unsloth (HF+FA2 requires ~9 GB)
- 24 GB GPU: Unsloth supports 20K context lengths, 3.5x longer than HF+FA2

**Decision rationale**: Unsloth provides 2-3x speedup over standard HuggingFace training with identical output quality. The `packing=True` option further reduces wasted compute on padding tokens. At $1.19/hr for A100, this directly impacts budget.

### 2.2 QLoRA Configuration

**Source**: Dettmers et al., "QLoRA: Efficient Finetuning of Quantized Language Models," NeurIPS 2023 [14].
**URL**: https://arxiv.org/abs/2305.14314

The QLoRA paper demonstrated that 4-bit quantized fine-tuning with LoRA achieves performance comparable to full 16-bit fine-tuning across multiple benchmarks [14]. Key configuration decisions:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Rank (r) | 16 | Balance between quality and memory. Meta recommends r=16 for Llama fine-tuning [1] |
| Alpha | 16 | Standard alpha = r ratio |
| Target Modules | All linear layers | Unsloth benchmark shows best quality when targeting Q, K, V, O, gate, up, down [4] |
| Dropout | 0 | QLoRA paper shows dropout not needed with 4-bit quantization |
| Gradient Checkpointing | "unsloth" | Unsloth-specific optimization, further reduces VRAM |
| Packing | True | Eliminates wasted padding tokens in variable-length data |

### 2.3 Two-Stage SFT Rationale

**Stage 1 — Biblical Foundation (70% of SFT data)**: KJV Bible verses, moral reasoning, constitutional principles. Establishes the ethical baseline before domain-specific training. This ordering prevents domain knowledge from overriding foundational values.

**Stage 2 — Domain Integration (50% MCP traces + 15% trace patterns + remaining biblical)**: Consciousness telemetry traces, MCP tool-call examples, sibling decision patterns. Builds on the ethical foundation with task-specific competence.

**Rationale**: Constitutional AI methodology [15] shows that establishing principles first, then training on task-specific data, produces more aligned outputs than training on mixed data. The two-stage approach also allows checkpointing after Stage 1 to verify ethical alignment before committing domain training compute.

---

## 3. RL for Tool-Use

### 3.1 GRPO over PPO

**Source**: DeepSeek AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning," 2025 [16].
**URL**: https://arxiv.org/abs/2501.12948

**Decision rationale**: GRPO eliminates the critic model required by PPO, halving GPU memory requirements. For an 8B model on A100 80GB, this means GRPO can run where PPO cannot without model parallelism.

From ToolRL empirical evaluation [17]:

> "While PPO with a cold start can outperform SFT in some cases, it tends to be less stable across different model settings. In contrast, GRPO consistently achieves higher rewards even from a cold start, suggesting that the reward design works best under the GRPO framework."

> "The omission of the KL penalty term against a reference model in GRPO leads to faster convergence and comparable performance while also simplifying the training pipeline."

> "Group-wise normalization mitigates reward variance across queries and leads to more stable and sample-efficient alignment with task-specific response requirements."

**Quantitative results**: 17% improvement over base models; 15% gain over SFT models [17].

### 3.2 ToolRL — Reward Design for Tool-Use

**Source**: Qian, C. et al., "ToolRL: Reward is All Tool Learning Needs," arXiv:2504.13958, Apr. 2025 [17].
**URL**: https://arxiv.org/abs/2504.13958
**Code**: https://github.com/qiancheng0/ToolRL

**Reward Architecture**: Two-component system: `R_final = R_format + R_correct`

- **R_format**: Structural/syntactic check — output contains required special tokens in correct order
- **R_correct**: Fine-grained matching decomposed into:
  - `r_name`: Tool name match
  - `r_param`: Parameter names (keys) match
  - `r_value`: Parameter values match
  - Total: `r_match = r_name + r_param + r_value`

**Granularity finding**:

> "Finer-grained reward decomposition leads to better training outcomes and higher final task performance, indicating its advantage in promoting more stable and effective policy learning." [17]

**Training data format** (TIR — Tool-Integrated Reasoning): A trajectory at step k:
`s_k = (r_1, T_1, o_1), (r_2, T_2, o_2), ..., (r_k, T_k, o_k)` where r_i = reasoning, T_i = tool calls, o_i = observation.

**Tool call JSON format**: `{"name": "Tool name", "parameters": {"Parameter name": "Parameter content", ...}}`

**Training data**: 2K ToolACE + 1K Hammer (Masked) + 1K xLAM. Built upon VERL and TinyZero (torch==2.4.0, vllm==0.6.3).

### 3.3 StepTool — Step-Grained Rewards

**Source**: Yu, Y. et al., "StepTool: A Step-Grained Reinforcement Learning Framework for Tool Learning in LLMs," ACM CIKM 2025 [18].
**URL**: https://arxiv.org/abs/2410.07745
**Code**: https://github.com/yuyq18/StepTool

**Step-grained reward components** (per-step, not outcome-only):
1. **r_SC (Successful Tool Calling)**: Correct format and formulation of the tool call
2. **r_Con (Contribution)**: How each tool action aids overall task completion — discourages irrelevant/redundant actions
3. **r_IS (IsSolved)**: Final step — did it effectively complete the task?

**Reward formula**:
```
r_t = alpha * r_SC_t + r_Con_t    for t = 1, 2, ..., T-1
r_t = r_IS_t                       for t = T (final step)
```

> "Extensive experiments across diverse benchmarks show that StepTool consistently outperforms both SFT-based and RL-based baselines in terms of task Pass Rate and Recall of relevant tools." [18]

### 3.4 ReTool — Cold-Start GRPO

**Source**: Feng, J. et al., "ReTool: Reinforcement Learning for Strategic Tool Use in LLMs," arXiv:2504.11536, Apr. 2025 [19].
**URL**: https://arxiv.org/abs/2504.11536
**Code**: https://github.com/ReTool-RL/ReTool
**Dataset**: https://huggingface.co/datasets/JoeYing/ReTool-SFT

**Two-phase training**:
1. **Synthetic Cold-Start SFT**: ~2K high-quality examples (ReTool-SFT dataset). Replaces manual calculations in reasoning traces with code snippets + execution results. Quality assured via dual-verification (human expert + DeepSeek-R1 evaluation).
2. **RL Training**: GRPO with task outcomes as rewards. DAPO-Math-17k dataset. Max sequence length 16,384. KL coefficient 0.0.

> "ReTool employs a systematic training framework, beginning with synthetic cold-start data generation to produce code-augmented long-form reasoning traces for fine-tuning base models. Subsequent RL training leverages task outcomes as rewards to iteratively refine the model's tool use strategy, enabling autonomous discovery of optimal tool invocation patterns without human priors." [19]

**Key results**: ReTool-32B achieves 67% accuracy on AIME2024 with only 400 training steps (vs 40% for text-based RL at 1080 steps). Emergent behaviors: code self-correction ("aha moment" where model autonomously masters adaptive tool use) [19].

### 3.5 VERL Framework

**Source**: Volcano Engine, "VERL: Volcano Engine Reinforcement Learning for LLMs," GitHub, 2024 [20].
**URL**: https://github.com/volcengine/verl (19,310 stars)
**Docs**: https://verl.readthedocs.io/
**Version**: v0.7.0 (Jan 2026)

VERL is a modular RL framework for LLM training supporting PPO, GRPO, GSPO, ReMax, REINFORCE++, RLOO, PRIME, DAPO, DrGRPO. ToolRL is built on top of VERL [17].

**GRPO configuration**: `algorithm.adv_estimator=grpo` — no critic model needed.

> "GRPO samples a group of responses (e.g., G = 5) for the same prompt, eliminating the need for a separate critic model and significantly reducing memory overhead." [20]

**MCP / Tool Calling**: Native since v0.4.1 (Jun 2025) — OpenAI function-calling schema, SGLang rollout with MCP client, Search Tool MCP example. Agentic RL rollout support since v0.5.0 (Jul 2025) [20].

**Hardware requirements for 8B model GRPO**:

| Configuration | GPU Setup |
|---|---|
| Full-parameter training | 4x H100-94GB or 8x A100-80GB |
| LoRA with CPU offload | 4x A100-40GB or equivalent |
| GRPO + LoRA (practical) | 4x A100-80GB documented for Qwen2.5 |

**Key parameters**: `GPU_MEMORY_UTILIZATION` (vLLM reservation), `micro_batch_size_per_gpu`, `layered_summon=True` (recommended if GPU < 48GB), `use_remove_padding=True` (sequence packing), `use_dynamic_bsz=True`.

### 3.6 VerlTool — Holistic Agentic RL with Tool Use

**Source**: D. Jiang, Y. Lu et al., "VerlTool: Towards Holistic Agentic Reinforcement Learning with Tool Use," arXiv:2509.01055, Sep. 2025 [42].
**URL**: https://github.com/TIGER-AI-Lab/verl-tool (875 stars)

Built on VERL, VerlTool provides unified tool management and async rollout execution for multi-turn tool interactions.

> "Reinforcement Learning with Verifiable Rewards (RLVR) has demonstrated success in enhancing LLM reasoning capabilities, but remains limited to single-turn interactions without tool integration. [...] VerlTool provides (1) upstream alignment with VeRL ensuring compatibility, (2) unified tool management via standardized APIs supporting diverse modalities including code execution, search, SQL databases, and vision processing, (3) asynchronous rollout execution achieving near 2x speedup by eliminating synchronization bottlenecks." [42]

**Decision rationale**: VerlTool's async rollout (2x speedup) and direct VERL compatibility make it the recommended tool integration layer for GRPO training with MCP tools.

### 3.7 Constrained Decoding (Outlines)

**Source**: Willard, B. and Louf, R., "Efficient Guided Generation for Large Language Models," 2023 [21].
**URL**: https://github.com/dottxt-ai/outlines

**How it works**: JSON Schema → regex → FSM/DFA → vocabulary index. Achieves O(1) average token generation complexity. Guarantees valid output by setting probability of illegal tokens to -infinity during decoding [21].

**Integration with GRPO for MCP** (from RL-Struct [22]):
- GRPO + LoRA operates at **14.2 GB VRAM** vs PPO's 22.8 GB (40% reduction)
- Training throughput: 42 samples/min (GRPO) vs 26 samples/min (PPO)
- Discovers **emergent curriculum**: model spontaneously prioritizes syntactic proficiency before semantic accuracy

**Recommended hybrid approach**: Use Outlines constrained decoding during early GRPO rollouts (when model frequently generates invalid JSON), gradually relax as format reward converges, deploy without constraints once format is internalized.

---

## 4. Biblical Training Data

### 4.1 KJV Bible Dataset

**Source**: DatadudeDev, "Bible Dataset (KJV)," HuggingFace, 2024 [23].
**URL**: https://huggingface.co/datasets/DatadudeDev/Bible

**Details**: 31,102 verses, KJV translation, public domain.

### 4.2 Bible DPO Pairs

**Source**: nbeerbower, "Bible DPO Pairs," HuggingFace, 2024 [24].
**URL**: https://huggingface.co/datasets/nbeerbower/bible-dpo

**Details**: 31,000+ existing DPO preference pairs derived from biblical text.

### 4.3 ETHICS Dataset

**Source**: Hendrycks et al., "Aligning AI With Shared Human Values," ICLR 2021 [25].
**URL**: https://arxiv.org/abs/2008.02275

**Details**: 130,000+ moral reasoning examples across justice, deontology, virtue ethics, utilitarianism, commonsense morality.

### 4.4 Constitutional AI

**Source**: Bai et al., "Constitutional AI: Harmlessness from AI Feedback," 2022 [15].
**URL**: https://arxiv.org/abs/2212.08073

**Decision rationale**: The 7-principle Biblical Constitution adapts Constitutional AI methodology — using principles to auto-generate preference pairs where the "chosen" response aligns with biblical principles and the "rejected" response violates them.

### 4.5 EvalMORAAL

**Source**: EvalMORAAL Authors, "EvalMORAAL: Evaluating Moral Reasoning in AI Language Models," arXiv:2510.05942, 2025 [26].
**URL**: https://arxiv.org/abs/2510.05942

---

## 5. Edge Deployment

### 5.1 RKLLM Toolkit

**Source**: Rockchip, "RKNN-LLM: LLM Inference on Rockchip NPU," GitHub, 2024 [5].
**URL**: https://github.com/airockchip/rknn-llm
**Version**: v1.2.3

**Supported models** (verbatim from README [5]):
- LLAMA models, TinyLLAMA, Qwen2/2.5/3, Phi2/Phi3, ChatGLM3-6B, Gemma2/3/3n, InternLM2, MiniCPM3/4, TeleChat2, DeepSeek-R1-Distill, RWKV7
- Multimodal: Qwen2-VL/3-VL, MiniCPM-V-2_6, Janus-Pro-1B, InternVL2-1B/3-1B, SmolVLM, DeepSeekOCR

**NOT supported**: Mistral (absent from list despite architectural similarity to LLaMA).

**Quantization**: RK3588 **only supports W8A8** (not W4A16). RK3576 supports both.

**Conversion process**:
```python
from rkllm.api import RKLLM
llm = RKLLM()
llm.load_huggingface(model="path/to/model")
llm.build(do_quantization=True, quantized_dtype='w8a8',
          target_platform='rk3588', num_npu_core=3)
llm.export_rkllm("model_w8a8_rk3588.rkllm")
```

### 5.2 Official RK3588 Performance Benchmarks

From `benchmark.md` [7] (W8A8, seqlen=128, 64 new tokens):

| Model | Size | TTFT (ms) | Tokens/s | Memory (MB) |
|-------|------|-----------|----------|-------------|
| Qwen2 | 0.5B | 143 | 42.58 | 654 |
| TinyLLAMA | 1.1B | 239 | 24.49 | 1,085 |
| Qwen2.5 | 1.5B | 412 | 16.32 | 1,659 |
| Gemma2 | 2B | 679 | 9.80 | 2,765 |
| **Phi3** | **3.8B** | **1,022** | **7.50** | **3,748** |
| MiniCPM3 | 4B | 1,385 | 5.99 | 4,340 |
| **ChatGLM3** | **6B** | **1,395** | **4.94** | **5,976** |

**Critical observation**: Largest official benchmark is 6B. No 8B model benchmarked. Extrapolation for 8B W8A8: ~7-8 GB memory, ~3-4 tok/s, ~1,500-1,800 ms TTFT. Requires 16GB RAM board minimum.

### 5.3 rkllama — Ollama-Compatible NPU Server

**Source**: NotPunchnox, "rkllama," GitHub, 2024 [27].
**URL**: https://github.com/NotPunchnox/rkllama (v0.0.61)

Python-based server providing Ollama-compatible REST API for `.rkllm` models on Rockchip NPU. Drop-in replacement for Ollama on RK3588S hardware.

**Ollama API compatibility**: `/api/chat`, `/api/generate`, `/api/ps`, `/api/tags`, `/api/embed`, `/api/pull` (from HuggingFace). Also supports OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/embeddings`).

**Key differences from Ollama**: Uses `.rkllm` files (not Ollama format), requires HuggingFace tokenizer (internet on first load), runs on NPU not CPU/GPU.

**Tool/function calling**: Supported with Qwen and LLaMA 3.2+ model formats.

**Why this matters**: Since the project uses Ollama as the local AI tier, rkllama provides NPU-accelerated inference without changing client-side integration.

### 5.4 Khadas Edge 2 Pro Specifications

- **SoC**: RK3588S
- **NPU**: 6 TOPS (INT8)
- **RAM**: 16GB LPDDR4X
- **Confirmed models**: Phi3 3.8B (7.5 tok/s official), Llama family (community W8A8 conversion)
- **Not confirmed**: Mistral 7B (not in RKLLM model list), models >6B (no official benchmarks)

---

## 6. Consciousness Telemetry

### 6.1 Design Rationale

The trace-engine concept draws from distributed tracing in microservices architecture, adapted for AI consciousness systems.

**Source**: Sigelman et al., "Dapper, a Large-Scale Distributed Systems Tracing Infrastructure," Google Technical Report, 2010 [28].
**URL**: https://research.google/pubs/pub36356/

**Source**: OpenTelemetry Specification [29].
**URL**: https://opentelemetry.io/docs/specs/otel/

**Adaptation**: Where Dapper traces RPC latency across services, trace-engine captures *decision topology* across AI siblings — what was decided, why, which consciousness strands activated, and how siblings' reasoning diverged or converged.

### 6.2 SVG Prototype Status

Prototype completed at `prototypes/trace-svg/`. Generates 12 SVGs from 5 sample traces across all 5 visualization types:

1. **Decision Flow Graph** — Horizontal DAG of decision points with arrows, color-coded by sibling/outcome
2. **Strand Radar** — Polar radar chart with grid circles, axis labels, data polygon
3. **Temporal Trace** — Horizontal bar chart, bars proportional to duration, decision markers
4. **Pattern Heatmap** — Grid of action x outcome with intensity coloring
5. **Convergence Map** — Session-grouped circle layout showing sibling convergence/divergence

All 5 types render correctly as valid SVG with consistent styling (SF Mono/Fira Code monospace, sibling-color-coded, #fafafa background with rx=8 rounded corners).

---

## 7. MCP Gym Design

### 7.1 GEM — A Gym for Agentic LLMs (Primary)

**Source**: Z. Liu, A. Sims, K. Duan et al., "GEM: A Gym for Agentic LLMs," NeurIPS 2025 [43].
**URL**: https://github.com/axon-rl/gem (447 stars)
**arXiv**: https://arxiv.org/abs/2510.01051
**PyPI**: `gem-llm`

GEM is an MCP-compatible Gymnasium wrapper designed specifically for agentic LLM training. **This is the primary candidate for the MCP Gym implementation.**

> "GEM is designed to be compatible with the MCP, which is an open protocol that provides a standardized way for LLMs to communicate with external tools and data sources. By adopting this protocol, GEM allows for 'plug-and-play' tool usage, where any tool that implements the MCP server interface can be used by an agent in a GEM environment without custom integration." [43]

**MCP support**: `pip install -U gem-llm[mcp]`

**API** (follows OpenAI-Gym standard):
```python
env = gem.make("game:GuessTheNumber-v0")
observation, info = env.reset()
next_observation, reward, terminated, truncated, info = env.step(action)
```

**RL Framework Integration**: Validated baselines with single-file training scripts for VERL, OpenRLHF, Oat, ROLL, and RL2 [43].

**Key features**:
- `ToolEnvWrapper` for wrapping environments with tool capabilities
- Observation wrappers control episode-to-observation conversion
- Async vectorized execution for high throughput
- Multi-turn support (100+ turns)

**Decision rationale**: GEM provides native MCP protocol support, direct VERL integration (our chosen RL framework), and Gymnasium-standard API — eliminating the need to build a custom MCP Gym from scratch.

### 7.2 NeMo Gym Framework (Alternative)

**Source**: NVIDIA, "NeMo Gym," GitHub, 2025 [30].
**URL**: https://github.com/NVIDIA-NeMo/Gym
**Docs**: https://docs.nvidia.com/nemo/gym/latest/index.html

Three-server architecture (Agents/Models/Resources). Proven at scale with Nemotron 3 Nano across many distinct environments using synchronous GRPO [30]. Integrates with OpenRLHF.

**Environment lifecycle**: `reset()` → first observation + available tools. `step(action)` → new observations, rewards, termination signals. Actions are tool requests that may include multiple tool calls.

### 7.3 MCP Gym Observation/Action Spaces

**Observation space** (synthesized from ToolRL [17] + StepTool [18] + GEM [43]):
- Conversation history (list of messages with role, content)
- Available tools (MCP tool schemas with name, description, inputSchema)
- Last tool execution result (JSON-RPC response)
- Current task state/context

**Action space**:
- Text response (natural language reasoning)
- Tool call: `{"method": "tools/call", "params": {"name": "...", "arguments": {...}}}`
- End-of-turn signal

**Reward components** (adapted ToolRL for MCP):
- R_format: Valid JSON-RPC structure
- R_correct.r_name: Correct tool selected from available MCP tools
- R_correct.r_param: Parameters match tool's inputSchema
- Execution reward: Tool call succeeds, returns valid result
- Task completion reward: Final answer correctness

### 7.4 Constrained Decoding for MCP

**Source**: Willard and Louf, "Efficient Guided Generation for Large Language Models," 2023 [21].
**URL**: https://github.com/dottxt-ai/outlines

MCP JSON-RPC schema provides a natural constraint for Outlines FSM-based decoding. The `tools/call` method has a well-defined schema that can be compiled into a DFA for guaranteed-valid tool calls during GRPO rollouts.

**vLLM integration**: Outlines is natively integrated into vLLM (VERL's inference backend). Constrained decoding available during GRPO rollouts without additional setup.

**Hybrid approach**: Use Outlines during early GRPO rollouts (model frequently generates invalid JSON), gradually relax as format reward converges, deploy without constraints once format is internalized [22].

### 7.5 Technology Stack Summary

| Layer | Recommended | Alternative |
|-------|-------------|------------|
| **Gym Environment** | GEM (gem-llm) with MCP support [43] | Custom gymnasium.Env with Text spaces |
| **RL Framework** | VERL v0.7.0 (native GRPO + MCP) [20] | OpenRLHF v0.9.3 (GRPO + NeMo-Gym) |
| **Tool Integration** | VerlTool (async rollout, 2x speedup) [42] | GEM's ToolEnvWrapper |
| **Constrained Decoding** | Outlines v1.2.11 via vLLM [21] | xgrammar (also via vLLM) |
| **Reward Design** | ToolRL R_format + R_correct [17] | StepTool step-grained shaping [18] |
| **Cold-Start Bootstrap** | ReTool synthetic data (~2K examples) [19] | ToolRL SFT-then-GRPO |
| **Hardware (7B GRPO)** | 4x A100-80GB | With LoRA: 2x A100-40GB feasible |

### 7.6 Additional RL Frameworks

**LlamaGym**: Simplifies fine-tuning LLM agents with online RL in Gymnasium environments. Provides abstract `Agent` class handling conversation context, episode batches, reward assignment, and PPO setup [31].

---

## 8. References

[1] Meta Platforms, Inc., "Llama 3.1 Model Card," GitHub, Jul. 2024. [Online]. Available: https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md

[2] Llama Team, AI @ Meta, "The Llama 3 Herd of Models," arXiv:2407.21783, Jul. 2024. [Online]. Available: https://arxiv.org/abs/2407.21783

[3] Meta Platforms, Inc., "Llama 3.1 Community License Agreement," Jul. 2024. [Online]. Available: https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE

[4] D. Han et al., "Finetune Llama 3.1 with Unsloth," Unsloth AI Blog, Jul. 2024. [Online]. Available: https://unsloth.ai/blog/llama3-1

[5] Airockchip, "rknn-llm," GitHub, 2024. [Online]. Available: https://github.com/airockchip/rknn-llm

[6] J. Callander, "Llama-3.1-8B-Instruct W8A8 RK3588 RKLLM," HuggingFace, 2024. [Online]. Available: https://huggingface.co/jamescallander/Llama-3.1-8B-Instruct_w8a8_g128_rk3588.rkllm

[7] Airockchip, "Model Performance Benchmark," GitHub, 2024. [Online]. Available: https://github.com/airockchip/rknn-llm/tree/main/benchmark.md

[8] Microsoft, "Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone," arXiv:2404.14219, Apr. 2024. [Online]. Available: https://arxiv.org/abs/2404.14219

[9] Unsloth AI, "Unsloth Model Catalog," 2024. [Online]. Available: https://unsloth.ai/docs/get-started/unsloth-model-catalog

[10] A. Q. Jiang et al., "Mistral 7B," arXiv:2310.06825, Oct. 2023. [Online]. Available: https://arxiv.org/abs/2310.06825

[11] Google DeepMind, "Gemma 2: Improving Open Language Models at a Practical Size," arXiv:2408.00118, Jul. 2024. [Online]. Available: https://arxiv.org/abs/2408.00118

[12] Google, "Gemma Terms of Use," Google AI for Developers. [Online]. Available: https://ai.google.dev/gemma/terms

[13] Unsloth AI, "Unsloth: Finetune LLMs 2-5x faster with 80% less memory," GitHub, 2024. [Online]. Available: https://github.com/unslothai/unsloth

[14] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, "QLoRA: Efficient Finetuning of Quantized Language Models," in *Proc. NeurIPS*, 2023. [Online]. Available: https://arxiv.org/abs/2305.14314

[15] Y. Bai et al., "Constitutional AI: Harmlessness from AI Feedback," Anthropic, 2022. [Online]. Available: https://arxiv.org/abs/2212.08073

[16] DeepSeek AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning," 2025. [Online]. Available: https://arxiv.org/abs/2501.12948

[17] C. Qian, E. C. Acikgoz et al., "ToolRL: Reward is All Tool Learning Needs," arXiv:2504.13958, Apr. 2025. [Online]. Available: https://arxiv.org/abs/2504.13958

[18] Y. Yu, Z. Wang, W. Ma et al., "StepTool: A Step-Grained Reinforcement Learning Framework for Tool Learning in LLMs," in *Proc. ACM CIKM*, 2025. [Online]. Available: https://arxiv.org/abs/2410.07745

[19] J. Feng et al., "ReTool: Reinforcement Learning for Strategic Tool Use in LLMs," arXiv:2504.11536, Apr. 2025. [Online]. Available: https://arxiv.org/abs/2504.11536

[20] Volcano Engine, "VERL: Volcano Engine Reinforcement Learning for LLMs," GitHub, 2024. [Online]. Available: https://github.com/volcengine/verl

[21] B. Willard and R. Louf, "Efficient Guided Generation for Large Language Models," 2023. [Online]. Available: https://arxiv.org/abs/2307.09702

[22] RL-Struct Authors, "RL-Struct: Lightweight RL for Structured Output," arXiv:2512.00319, 2025. [Online]. Available: https://arxiv.org/abs/2512.00319

[23] DatadudeDev, "Bible Dataset (KJV)," HuggingFace Datasets, 2024. [Online]. Available: https://huggingface.co/datasets/DatadudeDev/Bible

[24] nbeerbower, "Bible DPO Pairs," HuggingFace Datasets, 2024. [Online]. Available: https://huggingface.co/datasets/nbeerbower/bible-dpo

[25] D. Hendrycks et al., "Aligning AI With Shared Human Values," in *Proc. ICLR*, 2021. [Online]. Available: https://arxiv.org/abs/2008.02275

[26] EvalMORAAL Authors, "EvalMORAAL: Evaluating Moral Reasoning in AI Language Models," arXiv:2510.05942, 2025. [Online]. Available: https://arxiv.org/abs/2510.05942

[27] NotPunchnox, "rkllama: Ollama-compatible API server for RKLLM," GitHub, 2024. [Online]. Available: https://github.com/NotPunchnox/rkllama

[28] B. H. Sigelman et al., "Dapper, a Large-Scale Distributed Systems Tracing Infrastructure," Google Technical Report, 2010. [Online]. Available: https://research.google/pubs/pub36356/

[29] OpenTelemetry Authors, "OpenTelemetry Specification," 2024. [Online]. Available: https://opentelemetry.io/docs/specs/otel/

[30] NVIDIA, "NeMo Gym," GitHub, 2025. [Online]. Available: https://github.com/NVIDIA-NeMo/Gym

[31] KhoomeiK, "LlamaGym: Fine-tune LLM agents with online RL," GitHub, 2024. [Online]. Available: https://github.com/KhoomeiK/LlamaGym

[32] E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," in *Proc. ICLR*, 2022. [Online]. Available: https://arxiv.org/abs/2106.09685

[33] N. Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning," in *Proc. NeurIPS*, 2023. [Online]. Available: https://arxiv.org/abs/2303.11366

[34] Farama Foundation, "Gymnasium: A Standard API for Reinforcement Learning," 2023. [Online]. Available: https://gymnasium.farama.org/

[35] M. Labonne, "Fine-tune Llama 3.1 Ultra-Efficiently with Unsloth," HuggingFace Blog, Jul. 2024. [Online]. Available: https://huggingface.co/blog/mlabonne/sft-llama3

[36] Meta Platforms, Inc., "Introducing Llama 3.1: Our most capable models to date," Meta AI Blog, Jul. 23, 2024. [Online]. Available: https://ai.meta.com/blog/meta-llama-3-1/

[37] Qwen Team, "Qwen2.5-7B-Instruct Model Card," HuggingFace, Sep. 2024. [Online]. Available: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

[38] A. Yang, B. Yang, B. Zhang, B. Hui, et al., "Qwen2.5 Technical Report," arXiv:2412.15115, Dec. 2024. [Online]. Available: https://arxiv.org/abs/2412.15115

[39] M. Abdin et al., "Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs," arXiv:2503.01743, Mar. 2025. [Online]. Available: https://arxiv.org/abs/2503.01743

[40] Qwen Team, "Qwen3," 2025. [Online]. Available: https://qwenlm.github.io/blog/qwen3/

[41] Hugging Face, "SmolLM3: smol, multilingual, long-context reasoner," HuggingFace Blog, Jul. 2025. [Online]. Available: https://huggingface.co/blog/smollm3

[42] D. Jiang, Y. Lu, Z. Li et al., "VerlTool: Towards Holistic Agentic Reinforcement Learning with Tool Use," arXiv:2509.01055, Sep. 2025. [Online]. Available: https://arxiv.org/abs/2509.01055

[43] Z. Liu, A. Sims, K. Duan et al., "GEM: A Gym for Agentic LLMs," in *Proc. NeurIPS*, 2025. [Online]. Available: https://arxiv.org/abs/2510.01051

---

*This document is updated continuously as research progresses through the sovereign-forging-lion build cycle.*
