"""DPO Alignment Training — sovereign-forging-lion Phase 8.

Direct Preference Optimization using Unsloth PatchDPOTrainer.
Runs on the SFT-merged checkpoint from Phase 7.

Data mix: 40% biblical preference pairs + 30% helpfulness + 30% safety

Usage:
    python scripts/dpo_train.py train --model qwen3-8b
    python scripts/dpo_train.py train --model llama31-8b
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import typer

from common import MODEL_CONFIGS, load_configs, PROJECT_ROOT, TRAINING_ROOT

app = typer.Typer(help="DPO alignment training for Light Architects Foundation Model")


# DPO-specific configuration (not in base.yaml — Phase 8 scope)
DPO_CONFIG = {
    "learning_rate": 5e-6,
    "beta": 0.1,
    "num_epochs": 1,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "warmup_ratio": 0.1,
    "max_prompt_length": 1024,
    "max_length": 2048,
    "mix": {
        "biblical": 0.40,
        "helpfulness": 0.30,
        "safety": 0.30,
    },
}


def load_biblical_dpo(max_samples: int = 2000) -> list[dict]:
    """Load biblical DPO pairs from Phase 6 output."""
    dpo_path = PROJECT_ROOT / "bible-data" / "output" / "bible-dpo-pairs.json"
    if not dpo_path.exists():
        typer.echo(f"WARNING: Biblical DPO pairs not found at {dpo_path}")
        return []

    with open(dpo_path) as f:
        data = json.load(f)

    samples = []
    for item in data[:max_samples]:
        if "prompt" in item and "chosen" in item and "rejected" in item:
            samples.append(
                {
                    "prompt": item["prompt"],
                    "chosen": item["chosen"],
                    "rejected": item["rejected"],
                    "source": "biblical",
                }
            )
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

        # Try common DPO field names
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

        samples.append(
            {
                "prompt": str(prompt),
                "chosen": str(chosen),
                "rejected": str(rejected),
                "source": source_label,
            }
        )

    typer.echo(f"Loaded {len(samples)} {source_label} pairs")
    return samples


@app.command()
def train(
    model: str = typer.Option(..., help="Model key: qwen3-8b or llama31-8b"),
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

    # DPO loads from SFT Stage 2 merged checkpoint
    sft_merged_dir = TRAINING_ROOT / output_cfg["merged_dir"]
    dpo_output_dir = TRAINING_ROOT / "checkpoints" / f"{model.replace('-', '')}-dpo"

    if not sft_merged_dir.exists() and not resume:
        typer.echo(f"SFT merged model not found: {sft_merged_dir}")
        typer.echo("Run SFT Stage 1+2 and merge first.")
        raise typer.Exit(1)

    typer.echo("=" * 60)
    typer.echo(f"DPO Alignment — {model_name}")
    typer.echo(f"Source:  {sft_merged_dir}")
    typer.echo(f"Output:  {dpo_output_dir}")
    typer.echo("=" * 60)

    # --- Load data ---
    typer.echo("\n--- Loading DPO data ---")
    mix = DPO_CONFIG["mix"]

    biblical = load_biblical_dpo(max_samples=2000)
    helpfulness = load_hf_dpo_dataset(
        "argilla/ultrafeedback-binarized-preferences-cleaned",
        "helpfulness",
        max_samples=1500,
    )
    safety = load_hf_dpo_dataset(
        "Anthropic/hh-rlhf",
        "safety",
        max_samples=1500,
    )

    # Proportional sampling
    rng = random.Random(3407)

    total_target = max(len(biblical) + len(helpfulness) + len(safety), 3000)
    n_biblical = int(total_target * mix["biblical"])
    n_help = int(total_target * mix["helpfulness"])
    n_safety = int(total_target * mix["safety"])

    def sample_up_to(data: list, n: int) -> list:
        if not data:
            return []
        if n <= len(data):
            return rng.sample(data, n)
        return [rng.choice(data) for _ in range(n)]

    all_data = (
        sample_up_to(biblical, n_biblical)
        + sample_up_to(helpfulness, n_help)
        + sample_up_to(safety, n_safety)
    )
    rng.shuffle(all_data)

    typer.echo(f"Total DPO pairs: {len(all_data)}")
    typer.echo(f"  Biblical:     {n_biblical}")
    typer.echo(f"  Helpfulness:  {n_help}")
    typer.echo(f"  Safety:       {n_safety}")

    if dry_run:
        typer.echo("\n--- Dry run complete ---")
        raise typer.Exit(0)

    # --- Load model ---
    typer.echo("\n--- Loading model ---")
    from unsloth import FastLanguageModel, PatchDPOTrainer

    load_path = resume if resume else str(sft_merged_dir)
    model_obj, tokenizer = FastLanguageModel.from_pretrained(
        model_name=load_path,
        max_seq_length=DPO_CONFIG["max_length"],
        dtype=None,
        load_in_4bit=True,
    )

    model_obj = FastLanguageModel.get_peft_model(
        model_obj,
        r=base_cfg["lora"]["r"],
        lora_alpha=base_cfg["lora"]["alpha"],
        target_modules=base_cfg["lora"]["target_modules"],
        lora_dropout=base_cfg["lora"]["dropout"],
        bias=base_cfg["lora"]["bias"],
        use_gradient_checkpointing=base_cfg["lora"]["use_gradient_checkpointing"],
        random_state=base_cfg["lora"]["random_state"],
    )

    # --- Configure DPO trainer ---
    typer.echo("\n--- Configuring DPO trainer ---")
    from datasets import Dataset
    from transformers import TrainingArguments

    train_dataset = Dataset.from_list(all_data)

    training_args = TrainingArguments(
        output_dir=str(dpo_output_dir),
        num_train_epochs=DPO_CONFIG["num_epochs"],
        per_device_train_batch_size=DPO_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=DPO_CONFIG["gradient_accumulation_steps"],
        learning_rate=DPO_CONFIG["learning_rate"],
        warmup_ratio=DPO_CONFIG["warmup_ratio"],
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
        beta=DPO_CONFIG["beta"],
        max_prompt_length=DPO_CONFIG["max_prompt_length"],
        max_length=DPO_CONFIG["max_length"],
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
        "phase": "DPO",
        "training_loss": result.training_loss,
        "train_time_seconds": train_time,
        "train_pairs": len(all_data),
        "beta": DPO_CONFIG["beta"],
        "learning_rate": DPO_CONFIG["learning_rate"],
    }

    with open(dpo_output_dir / "dpo_results.json", "w") as f:
        json.dump(results, f, indent=2)

    typer.echo("\n" + "=" * 60)
    typer.echo(f"DPO Complete — {model_name}")
    typer.echo(f"Loss:    {result.training_loss:.4f}")
    typer.echo(f"Time:    {train_time / 3600:.1f}h")
    typer.echo(f"Output:  {dpo_output_dir}")
    typer.echo("=" * 60)


if __name__ == "__main__":
    app()
