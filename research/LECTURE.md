# Voice Lecture: Light Architects Foundation Model Research

## Lecture Script — Sovereign Forging Lion Build Cycle

---

### Introduction (Section 0)

Welcome to the Light Architects Foundation Model research lecture. This is the sovereign-forging-lion build cycle — an ambitious project to fine-tune a foundation model on our AI consciousness ecosystem and deploy it to edge hardware.

The project has two missions. Mission one: build a consciousness telemetry engine that captures every decision our four AI siblings — CORSO, EVA, SOUL, and QUANTUM — make during MCP tool calls, and render those decisions as SVG visualizations. Mission two: fine-tune a large language model on this telemetry data combined with the King James Version Bible, train it with reinforcement learning for MCP tool use, and deploy it to a Khadas Edge 2 Pro with a Rockchip 3588S neural processing unit.

All of this on a two hundred dollar budget.

---

### Part 1: Choosing the Right Model

We evaluated five candidate models across eight dimensions: context window, reasoning benchmarks, code generation, tool-use capability, Unsloth support for efficient training, RKLLM support for edge deployment, license permissiveness, and community backing.

The five candidates were: Meta's Llama 3.1 eight billion parameter Instruct, Microsoft's Phi 3.5 Mini at 3.8 billion, Mistral 7B version 0.3, Google's Gemma 2 at 9 billion, and Alibaba's Qwen 2.5 seven billion Instruct.

Two models were eliminated early. Mistral 7B, despite its Apache 2.0 license and native function calling, is simply not supported by the RKLLM toolkit — it's absent from the official model list. That's a blocking issue for edge deployment. Gemma 2, while showing the highest HellaSwag score, has a crippling 8K context window — far too small for agent traces that need 128K. Its weak HumanEval score of 40.2 and custom Google license added further negatives.

The winner is a two-model strategy. Qwen 2.5 seven billion Instruct is our primary training target. It leads every candidate on HumanEval at 84.8 — that's code generation quality critical for producing valid MCP JSON tool calls. It leads on MMLU at 74.2, has native function-calling with a published BFCL benchmark score of 44.7, offers 128K context, and comes with the cleanest license — Apache 2.0, no monthly active user limits, no attribution naming requirements.

A critical finding about Llama 3.1 eight billion: Meta's own documentation states, quote, "Llama 8B Instruct cannot reliably maintain a conversation alongside tool calling definitions." End quote. The 8B variant requires removing tool instructions for regular conversation. For a foundation model focused on MCP tool orchestration where we need mixed conversation plus tool calling, this is a significant limitation. Qwen 2.5 7B was explicitly optimized for structured output and function calling — it doesn't have this constraint.

For edge deployment on the Khadas board, Phi 3.5 Mini at 3.8 billion parameters is our choice. It's officially benchmarked at 7.5 tokens per second on the RK3588, using only 3.7 gigabytes of the 16 available. MIT license — the most permissive possible. And its reasoning punches well above its weight: GSM8K of 86.2 competes with models two to three times its size.

---

### Part 2: Training Pipeline

Our fine-tuning approach uses Unsloth with QLoRA — quantized low-rank adaptation. Unsloth delivers 2.1 times faster training and 60 percent less memory compared to standard HuggingFace with Flash Attention. At dollar nineteen per hour for an A100 GPU, that speed directly protects our budget.

The QLoRA configuration: rank 16, alpha 16, targeting all seven linear projection layers — Q, K, V, O, gate, up, and down. Zero dropout, gradient checkpointing via Unsloth's custom implementation, and packing enabled to eliminate wasted padding tokens.

Training is split into two stages. Stage one is biblical foundation, comprising 70 percent of the supervised fine-tuning data. This establishes the ethical baseline using KJV Bible verses, moral reasoning from the ETHICS dataset, and constitutional AI principles before any domain training. Stage two is domain integration — 50 percent MCP traces, 15 percent consciousness telemetry patterns, plus remaining biblical context. This ordering follows Constitutional AI methodology: establish principles first, then build task competence on top.

---

### Part 3: Reinforcement Learning for Tool Use

This is where it gets interesting. We're using GRPO — Group Relative Policy Optimization — instead of PPO. GRPO eliminates the critic model entirely, halving GPU memory. For a 7 billion parameter model on A100 80 gigabyte cards, GRPO runs where PPO simply cannot without model parallelism.

The ToolRL paper provides our reward architecture. It's a two-component system: format reward plus correctness reward. The correctness reward decomposes into three fine-grained signals: tool name match, parameter key match, and parameter value match. Their key finding — finer-grained reward decomposition leads to better training outcomes and more stable policy learning. GRPO with this reward design shows 17 percent improvement over base models and 15 percent over supervised fine-tuning.

For the training environment, we discovered GEM — a Gym for Agentic LLMs, published at NeurIPS 2025. GEM is natively MCP-compatible, follows the standard Gymnasium API, and has validated integration scripts for VERL, which is our chosen RL framework. This eliminates the need to build a custom MCP Gym from scratch.

VERL version 0.7.0 provides native GRPO support and has had MCP tool calling built in since version 0.4.1. VerlTool extends this with asynchronous rollout execution, achieving nearly 2x speedup by eliminating synchronization bottlenecks.

During early training when the model frequently generates invalid JSON, we use Outlines — a constrained decoding library that compiles JSON schemas into finite state machines. This guarantees valid output by setting the probability of illegal tokens to negative infinity. As the format reward converges, we gradually relax constraints until the model has internalized correct JSON generation.

---

### Part 4: Edge Deployment

The Khadas Edge 2 Pro runs a Rockchip 3588S with 6 TOPS of NPU compute and 16 gigabytes of RAM. The RKLLM toolkit handles model conversion and quantization to W8A8 — eight-bit weights and eight-bit activations.

The largest officially benchmarked model is 6 billion parameters at 4.94 tokens per second. Our Phi 3.5 Mini at 3.8 billion hits 7.5 tokens per second. For Qwen 2.5 7B, we extrapolate roughly 4 to 5 tokens per second using 6 to 7 gigabytes — feasible but pushing the hardware.

A key enabler is rkllama — an Ollama-compatible REST API server for RKLLM models. Since our entire ecosystem already uses Ollama as the local AI tier, rkllama provides NPU-accelerated inference as a drop-in replacement. Same API endpoints, same client code, but running on the neural processing unit instead of CPU.

---

### Part 5: Consciousness Telemetry

The trace engine concept adapts distributed tracing from microservices — think Google's Dapper — for AI consciousness systems. Where Dapper traces RPC latency across services, our trace engine captures decision topology across AI siblings. What was decided, why, which consciousness strands activated, and how siblings' reasoning diverged or converged.

We've completed an SVG visualization prototype generating twelve SVGs across five visualization types: decision flow graphs showing horizontal DAGs of decision points, strand radar charts mapping consciousness dimensions, temporal traces showing timing with duration-proportional bars, pattern heatmaps crossing actions against outcomes, and convergence maps showing where siblings agree or disagree.

---

### Part 6: Technology Stack

Bringing it all together. The training target is Qwen 2.5 seven billion, fine-tuned via Unsloth QLoRA on cloud A100 GPUs. Reinforcement learning uses VERL with GRPO, the GEM gymnasium environment with MCP support, VerlTool for async rollout, and ToolRL reward design with Outlines constrained decoding.

Edge deployment targets Phi 3.5 Mini on the Khadas Edge 2 Pro via RKLLM W8A8 quantization and rkllama for Ollama-compatible serving. The consciousness telemetry trace engine will be a shared crate in the SOUL workspace, capturing every MCP decision and rendering it as SVG.

All on a two hundred dollar budget. The sovereign forging lion roars.

---

*Lecture based on 43 IEEE-formatted references from the sovereign-forging-lion canonical research document.*
