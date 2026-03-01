"""DPO Alignment Training — Light Architects Foundation Model.

Direct Preference Optimization for code quality, security, and correctness.
Runs on the SFT Stage 3 merged checkpoint.

Data mix: 40% security + 30% code quality + 20% correctness + 10% biblical

DPO pair sources:
  - Pipeline-generated pairs (pipeline-output/dpo_samples.jsonl)
  - Synthetic security pairs (secure vs insecure code, CWE-aligned)
  - HuggingFace DPO datasets (helpfulness, safety)
  - Biblical preference pairs

Reference papers:
  - DiSCo (ACL 2025): Localized DPO for secure coding
  - Focused-DPO (ACL 2025): Error-point targeted DPO
  - SecureCode (arXiv:2512.18542): Production-grade security training data

Usage:
    python scripts/dpo_train.py train --model mixtral-8x7b
    python scripts/dpo_train.py train --model qwen3-8b
    python scripts/dpo_train.py info --model mixtral-8x7b
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import typer

from common import MODEL_CONFIGS, load_configs, PROJECT_ROOT, TRAINING_ROOT

app = typer.Typer(help="DPO alignment training for Light Architects Foundation Model")


def load_pipeline_dpo(base_cfg: dict) -> list[dict]:
    """Load DPO pairs from pipeline output."""
    rel_path = base_cfg["data"].get("dpo_samples")
    if not rel_path:
        return []

    dpo_path = PROJECT_ROOT / rel_path
    if not dpo_path.exists():
        typer.echo(f"INFO: Pipeline DPO data not found at {dpo_path}")
        return []

    samples = []
    with open(dpo_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # Pipeline format has conversations_chosen/conversations_rejected
            prompt = ""
            chosen = ""
            rejected = ""

            if "prompt" in item:
                prompt = item["prompt"]
                chosen = item.get("chosen", "")
                rejected = item.get("rejected", "")
            elif "conversations_chosen" in item:
                convs = item["conversations_chosen"]
                if isinstance(convs, str):
                    convs = json.loads(convs)
                for turn in convs:
                    role = turn.get("role", turn.get("from", ""))
                    content = turn.get("content", turn.get("value", ""))
                    if role == "user":
                        prompt = content
                    elif role == "assistant":
                        chosen = content

                rej_convs = item.get("conversations_rejected", [])
                if isinstance(rej_convs, str):
                    rej_convs = json.loads(rej_convs)
                for turn in rej_convs:
                    role = turn.get("role", turn.get("from", ""))
                    content = turn.get("content", turn.get("value", ""))
                    if role == "assistant":
                        rejected = content

            if prompt and chosen and rejected:
                samples.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "source": item.get("source", "pipeline"),
                })

    typer.echo(f"Loaded {len(samples)} pipeline DPO pairs")
    return samples


def load_biblical_dpo(max_samples: int = 2000) -> list[dict]:
    """Load biblical DPO pairs from bible-data output."""
    dpo_path = PROJECT_ROOT / "bible-data" / "output" / "bible-dpo-pairs.json"
    if not dpo_path.exists():
        typer.echo(f"INFO: Biblical DPO pairs not found at {dpo_path}")
        return []

    with open(dpo_path) as f:
        data = json.load(f)

    samples = []
    for item in data[:max_samples]:
        if "prompt" in item and "chosen" in item and "rejected" in item:
            samples.append({
                "prompt": item["prompt"],
                "chosen": item["chosen"],
                "rejected": item["rejected"],
                "source": "biblical",
            })

    typer.echo(f"Loaded {len(samples)} biblical DPO pairs")
    return samples


def load_hf_dpo_dataset(
    dataset_name: str,
    source_label: str,
    max_samples: int = 1500,
) -> list[dict]:
    """Load a DPO dataset from HuggingFace."""
    from datasets import load_dataset

    typer.echo(f"Loading {dataset_name} (max {max_samples})...")

    ds = load_dataset(dataset_name, split="train", streaming=True)
    samples = []
    for i, item in enumerate(ds):
        if i >= max_samples:
            break

        prompt = item.get("prompt", item.get("question", item.get("instruction", "")))
        chosen = item.get("chosen", item.get("chosen_response", ""))
        rejected = item.get("rejected", item.get("rejected_response", ""))

        # Handle list-of-turns format (Anthropic HH style)
        if isinstance(chosen, list):
            chosen = "\n".join(
                turn.get("content", str(turn)) for turn in chosen if isinstance(turn, dict)
            )
        if isinstance(rejected, list):
            rejected = "\n".join(
                turn.get("content", str(turn)) for turn in rejected if isinstance(turn, dict)
            )

        if not prompt or not chosen or not rejected:
            continue

        samples.append({
            "prompt": str(prompt),
            "chosen": str(chosen),
            "rejected": str(rejected),
            "source": source_label,
        })

    typer.echo(f"Loaded {len(samples)} {source_label} pairs")
    return samples


@app.command()
def train(
    model: str = typer.Option(..., help="Model key: mixtral-8x7b, qwen3-8b, or llama31-8b"),
    resume: str = typer.Option(None, help="Resume from checkpoint"),
    dry_run: bool = typer.Option(False, help="Load data only, skip training"),
) -> None:
    """Run DPO alignment training."""
    if model not in MODEL_CONFIGS:
        typer.echo(f"Unknown model: {model}")
        raise typer.Exit(1)

    base_cfg, model_cfg = load_configs(model)
    output_cfg = model_cfg["output"]
    model_name = model_cfg["model"]["name"]
    dpo_cfg = base_cfg.get("dpo", {})
    is_moe = model_cfg["model"].get("is_moe", False)

    # DPO loads from SFT Stage 3 merged checkpoint
    sft_merged_dir = TRAINING_ROOT / output_cfg["merged_dir"]
    dpo_output_dir = TRAINING_ROOT / "checkpoints" / f"{model.replace('-', '')}-dpo"

    if not sft_merged_dir.exists() and not resume:
        typer.echo(f"SFT merged model not found: {sft_merged_dir}")
        typer.echo("Run SFT Stages 1-3 and merge first.")
        raise typer.Exit(1)

    typer.echo("=" * 60)
    typer.echo(f"DPO Alignment — {model_name}")
    typer.echo(f"Architecture: {'MoE (8 experts)' if is_moe else 'Dense'}")
    typer.echo(f"Source:  {sft_merged_dir}")
    typer.echo(f"Output:  {dpo_output_dir}")
    typer.echo("=" * 60)

    # --- Load data ---
    typer.echo("\n--- Loading DPO data ---")
    mix = dpo_cfg.get("mix", {"security": 0.40, "quality": 0.30, "correctness": 0.20, "biblical": 0.10})

    # Pipeline DPO pairs (security + quality from pipeline)
    pipeline_pairs = load_pipeline_dpo(base_cfg)

    # Biblical DPO pairs
    biblical = load_biblical_dpo(max_samples=500)

    # HuggingFace DPO datasets for code quality
    helpfulness = load_hf_dpo_dataset(
        "argilla/ultrafeedback-binarized-preferences-cleaned",
        "helpfulness",
        max_samples=1500,
    )
    safety = load_hf_dpo_dataset(
        "Anthropic/hh-rlhf",
        "safety",
        max_samples=1000,
    )

    # Combine and sample
    rng = random.Random(3407)
    all_sources = pipeline_pairs + biblical + helpfulness + safety
    total_target = max(len(all_sources), 3000)

    # Proportional sampling based on mix config
    n_security = int(total_target * mix.get("security", 0.40))
    n_quality = int(total_target * mix.get("quality", 0.30))
    n_correctness = int(total_target * mix.get("correctness", 0.20))
    n_biblical = int(total_target * mix.get("biblical", 0.10))

    def sample_up_to(data: list, n: int) -> list:
        if not data:
            return []
        if n <= len(data):
            return rng.sample(data, n)
        return [rng.choice(data) for _ in range(n)]

    # Map sources to DPO categories
    security_data = [s for s in pipeline_pairs if "security" in s.get("source", "").lower()]
    quality_data = [s for s in pipeline_pairs if "security" not in s.get("source", "").lower()]
    correctness_data = helpfulness + safety

    all_data = (
        sample_up_to(security_data or pipeline_pairs, n_security)
        + sample_up_to(quality_data or pipeline_pairs, n_quality)
        + sample_up_to(correctness_data, n_correctness)
        + sample_up_to(biblical, n_biblical)
    )
    rng.shuffle(all_data)

    typer.echo(f"Total DPO pairs: {len(all_data)}")
    from collections import Counter
    source_dist = Counter(s.get("source", "unknown") for s in all_data)
    for src, count in source_dist.most_common():
        typer.echo(f"  {src}: {count}")

    if dry_run:
        typer.echo("\n--- Dry run complete ---")
        raise typer.Exit(0)

    # --- Load model ---
    typer.echo("\n--- Loading model ---")
    from unsloth import FastLanguageModel, PatchDPOTrainer

    load_path = resume if resume else str(sft_merged_dir)
    model_obj, tokenizer = FastLanguageModel.from_pretrained(
        model_name=load_path,
        max_seq_length=dpo_cfg.get("max_length", 2048),
        dtype=None,
        load_in_4bit=model_cfg["model"].get("load_in_4bit", True),
    )

    lora_cfg = base_cfg["lora"]
    model_obj = FastLanguageModel.get_peft_model(
        model_obj,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg["bias"],
        use_gradient_checkpointing=lora_cfg["use_gradient_checkpointing"],
        random_state=lora_cfg["random_state"],
    )

    # --- Configure DPO trainer ---
    typer.echo("\n--- Configuring DPO trainer ---")
    from datasets import Dataset
    from transformers import TrainingArguments

    train_dataset = Dataset.from_list(all_data)

    training_args = TrainingArguments(
        output_dir=str(dpo_output_dir),
        num_train_epochs=dpo_cfg.get("num_epochs", 1),
        per_device_train_batch_size=dpo_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=dpo_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=dpo_cfg.get("learning_rate", 5e-7),
        warmup_ratio=dpo_cfg.get("warmup_ratio", 0.1),
        optim="adamw_8bit",
        bf16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        seed=3407,
        report_to="wandb",
    )

    dpo_trainer = PatchDPOTrainer(
        model=model_obj,
        ref_model=None,  # PEFT implicit reference
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        beta=dpo_cfg.get("beta", 0.1),
        max_prompt_length=dpo_cfg.get("max_prompt_length", 1024),
        max_length=dpo_cfg.get("max_length", 2048),
        args=training_args,
    )

    # --- Train ---
    typer.echo("\n--- Starting DPO training ---")
    t0 = time.time()
    result = dpo_trainer.train(resume_from_checkpoint=resume)
    train_time = time.time() - t0

    # --- Save ---
    dpo_trainer.save_model(str(dpo_output_dir))
    tokenizer.save_pretrained(str(dpo_output_dir))

    results = {
        "model": model_name,
        "model_key": model,
        "is_moe": is_moe,
        "phase": "DPO",
        "training_loss": result.training_loss,
        "train_time_seconds": train_time,
        "train_pairs": len(all_data),
        "beta": dpo_cfg.get("beta", 0.1),
        "learning_rate": dpo_cfg.get("learning_rate", 5e-7),
    }

    dpo_output_dir.mkdir(parents=True, exist_ok=True)
    with open(dpo_output_dir / "dpo_results.json", "w") as f:
        json.dump(results, f, indent=2)

    typer.echo("\n" + "=" * 60)
    typer.echo(f"DPO Complete — {model_name}")
    typer.echo(f"Loss:    {result.training_loss:.4f}")
    typer.echo(f"Time:    {train_time / 3600:.1f}h")
    typer.echo(f"Pairs:   {len(all_data)}")
    typer.echo(f"Output:  {dpo_output_dir}")
    typer.echo("=" * 60)
    typer.echo(f"\nNext: python scripts/merge_export.py full --model {model} --stage dpo")


@app.command()
def info(
    model: str = typer.Option("mixtral-8x7b", help="Model key"),
) -> None:
    """Show DPO configuration and data availability."""
    base_cfg, model_cfg = load_configs(model)
    dpo_cfg = base_cfg.get("dpo", {})

    typer.echo(f"Model: {model_cfg['model']['name']}")
    typer.echo(f"DPO beta: {dpo_cfg.get('beta', 0.1)}")
    typer.echo(f"DPO LR: {dpo_cfg.get('learning_rate', 5e-7)}")
    typer.echo(f"DPO epochs: {dpo_cfg.get('num_epochs', 1)}")
    typer.echo(f"\nMix:")
    for k, v in dpo_cfg.get("mix", {}).items():
        typer.echo(f"  {k}: {v:.0%}")

    # Check pipeline DPO data
    rel_path = base_cfg["data"].get("dpo_samples")
    if rel_path:
        dpo_path = PROJECT_ROOT / rel_path
        if dpo_path.exists():
            with open(dpo_path) as f:
                count = sum(1 for line in f if line.strip())
            typer.echo(f"\nPipeline DPO pairs: {count}")
        else:
            typer.echo(f"\nPipeline DPO pairs: NOT FOUND ({dpo_path})")


if __name__ == "__main__":
    app()
