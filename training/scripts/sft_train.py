"""SFT Training Script — sovereign-forging-lion Phase 7.

Two-stage supervised fine-tuning using Unsloth + QLoRA:
  Stage 1 (Biblical Foundation): 70% biblical + 15% domain + 15% general
  Stage 2 (Domain Integration): 50% domain + 20% general + 15% biblical + 15% trace

Usage:
    # Full two-stage training for Qwen3-8B
    python scripts/sft_train.py train --model qwen3-8b --stage 1
    python scripts/sft_train.py train --model qwen3-8b --stage 2

    # Full two-stage training for Llama 3.1 8B
    python scripts/sft_train.py train --model llama31-8b --stage 1
    python scripts/sft_train.py train --model llama31-8b --stage 2

    # Resume from checkpoint
    python scripts/sft_train.py train --model qwen3-8b --stage 1 --resume checkpoints/qwen3-8b-stage1-biblical/checkpoint-400
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import typer

from common import MODEL_CONFIGS, load_configs, PROJECT_ROOT, TRAINING_ROOT

app = typer.Typer(help="SFT training for Light Architects Foundation Model")


def load_local_data(stage: int, base_cfg: dict, seed: int = 3407) -> list[dict]:
    """Load pre-mixed local data for the given stage.

    If a pre-mixed file exists (from data_mixer.py), use it.
    Otherwise, mix on the fly.
    """
    premixed_path = TRAINING_ROOT / "data" / f"stage{stage}_train.jsonl"
    if premixed_path.exists():
        samples = []
        with open(premixed_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    # Mix on the fly
    from data_mixer import mix_for_stage

    mixed, stats = mix_for_stage(stage, base_cfg, seed)
    typer.echo(f"Mixed {stats.total_samples} samples on-the-fly (dedup removed: {stats.dedup_removed})")
    return mixed


def load_general_data(base_cfg: dict, stage: int) -> list[dict]:
    """Load general-purpose data from HuggingFace."""
    from datasets import load_dataset

    stage_cfg = base_cfg[f"stage{stage}"]
    dataset_name = stage_cfg["general_dataset"]
    max_samples = stage_cfg["general_max_samples"]

    typer.echo(f"Loading general data: {dataset_name} (max {max_samples} samples)")

    ds = load_dataset(dataset_name, split="train", streaming=True)
    samples = []
    for i, item in enumerate(ds):
        if i >= max_samples:
            break
        # OpenOrca format: system_prompt, question, response
        instruction = item.get("question", item.get("instruction", ""))
        system = item.get("system_prompt", "")
        response = item.get("response", item.get("output", ""))

        if not instruction or not response:
            continue

        inp = system if system and system != "You are an AI assistant." else ""
        samples.append(
            {
                "instruction": instruction,
                "input": inp,
                "output": response,
                "source": "general/openorca",
            }
        )

    typer.echo(f"Loaded {len(samples)} general samples")
    return samples


def build_formatting_func(model_key: str, tokenizer: Any) -> Any:
    """Build the formatting function using the model's native chat template."""

    def chat_template_format(examples: dict) -> list[str]:
        texts = []
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]

        for instruction, inp, output in zip(instructions, inputs, outputs):
            user_content = f"{instruction}\n\n{inp}" if inp else instruction
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return texts

    return chat_template_format


@app.command()
def train(
    model: str = typer.Option(..., help="Model key: qwen3-8b or llama31-8b"),
    stage: int = typer.Option(..., help="Training stage: 1 or 2"),
    resume: str = typer.Option(None, help="Resume from checkpoint path"),
    seed: int = typer.Option(3407, help="Random seed"),
    dry_run: bool = typer.Option(False, help="Load data and model but skip training"),
) -> None:
    """Run SFT training for the specified model and stage."""
    if model not in MODEL_CONFIGS:
        typer.echo(f"Unknown model: {model}. Choose from: {list(MODEL_CONFIGS.keys())}")
        raise typer.Exit(1)

    if stage not in (1, 2):
        typer.echo("Stage must be 1 or 2")
        raise typer.Exit(1)

    base_cfg, model_cfg = load_configs(model)
    model_name = model_cfg["model"]["name"]
    lora_cfg = base_cfg["lora"]
    train_cfg = base_cfg["training"]
    stage_cfg = base_cfg[f"stage{stage}"]
    output_cfg = model_cfg["output"]
    mon_cfg = base_cfg["monitoring"]

    output_dir = TRAINING_ROOT / output_cfg[f"stage{stage}_dir"]

    typer.echo("=" * 60)
    typer.echo(f"SFT Training — {model_name}")
    typer.echo(f"Stage {stage}: {stage_cfg['name']}")
    typer.echo(f"Output: {output_dir}")
    if resume:
        typer.echo(f"Resuming from: {resume}")
    typer.echo("=" * 60)

    # --- Step 1: Load model ---
    typer.echo("\n--- Loading model ---")
    from unsloth import FastLanguageModel

    if stage == 2 and not resume:
        # Stage 2 loads from Stage 1 merged checkpoint
        stage1_merged = TRAINING_ROOT / output_cfg["stage1_dir"]
        if stage1_merged.exists():
            typer.echo(f"Loading Stage 1 checkpoint from: {stage1_merged}")
            load_model_name = str(stage1_merged)
        else:
            typer.echo("WARNING: Stage 1 checkpoint not found, loading base model")
            load_model_name = model_name
    else:
        load_model_name = resume if resume else model_name

    t0 = time.time()
    model_obj, tokenizer = FastLanguageModel.from_pretrained(
        model_name=load_model_name,
        max_seq_length=train_cfg["max_seq_length"],
        dtype=None,
        load_in_4bit=model_cfg["model"]["load_in_4bit"],
        trust_remote_code=model_cfg["model"].get("trust_remote_code", False),
    )
    typer.echo(f"Model loaded in {time.time() - t0:.1f}s")

    # --- Step 2: Apply LoRA ---
    typer.echo("\n--- Applying QLoRA adapters ---")
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

    trainable = sum(p.numel() for p in model_obj.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model_obj.parameters())
    typer.echo(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # --- Step 3: Load data ---
    typer.echo("\n--- Loading training data ---")
    local_data = load_local_data(stage, base_cfg, seed)
    general_data = load_general_data(base_cfg, stage)
    all_data = local_data + general_data

    import random as rng_module

    rng_module.seed(seed)
    rng_module.shuffle(all_data)

    typer.echo(f"Total training samples: {len(all_data)}")
    typer.echo(f"  Local (biblical + domain + trace): {len(local_data)}")
    typer.echo(f"  General (OpenOrca): {len(general_data)}")

    # Load validation data — normalize to same format as training data
    val_path = PROJECT_ROOT / base_cfg["data"]["domain_val"]
    val_data = []
    if val_path.exists():
        with open(val_path) as f:
            raw_val = json.load(f)
        for item in raw_val:
            val_data.append({
                "instruction": item.get("instruction", item.get("prompt", "")),
                "input": item.get("input", ""),
                "output": item.get("output", item.get("response", "")),
            })
        typer.echo(f"Validation samples: {len(val_data)}")

    from datasets import Dataset

    train_dataset = Dataset.from_list(all_data)
    eval_dataset = Dataset.from_list(val_data) if val_data else None

    if dry_run:
        typer.echo("\n--- Dry run complete ---")
        typer.echo(f"Would train on {len(all_data)} samples")
        raise typer.Exit(0)

    # --- Step 4: Configure trainer ---
    typer.echo("\n--- Configuring SFT trainer ---")
    from trl import SFTTrainer
    from transformers import TrainingArguments

    formatting_func = build_formatting_func(model, tokenizer)

    # W&B setup
    report_to = mon_cfg["report_to"]
    if report_to == "wandb":
        try:
            import wandb

            wandb.init(
                project=mon_cfg["wandb_project"],
                name=f"{model_cfg['monitoring']['wandb_run_name']}-stage{stage}",
                tags=model_cfg["monitoring"]["wandb_tags"] + [f"stage{stage}"],
                config={
                    "model": model_name,
                    "stage": stage,
                    "lora_r": lora_cfg["r"],
                    "lora_alpha": lora_cfg["alpha"],
                    "learning_rate": train_cfg["learning_rate"],
                    "epochs": train_cfg["num_epochs"],
                    "batch_size": train_cfg["per_device_train_batch_size"],
                    "grad_accum": train_cfg["gradient_accumulation_steps"],
                    "max_seq_length": train_cfg["max_seq_length"],
                    "packing": train_cfg["packing"],
                    "train_samples": len(all_data),
                    "val_samples": len(val_data),
                },
            )
        except Exception:
            typer.echo("WARNING: W&B init failed, falling back to tensorboard")
            report_to = "tensorboard"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        optim=train_cfg["optim"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        bf16=train_cfg["bf16"],
        fp16=train_cfg["fp16"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        seed=train_cfg["seed"],
        report_to=report_to,
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=train_cfg["save_steps"] if eval_dataset else None,
        load_best_model_at_end=bool(eval_dataset),
        metric_for_best_model="eval_loss" if eval_dataset else None,
    )

    trainer = SFTTrainer(
        model=model_obj,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_func,
        max_seq_length=train_cfg["max_seq_length"],
        packing=train_cfg["packing"],
        args=training_args,
    )

    # --- Step 5: Train ---
    typer.echo("\n--- Starting training ---")
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=resume)
    train_time = time.time() - t0

    # --- Step 6: Save ---
    typer.echo("\n--- Saving model ---")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save training results
    results = {
        "model": model_name,
        "model_key": model,
        "stage": stage,
        "stage_name": stage_cfg["name"],
        "training_loss": result.training_loss,
        "train_time_seconds": train_time,
        "train_samples": len(all_data),
        "val_samples": len(val_data),
        "epochs": train_cfg["num_epochs"],
        "trainable_params": trainable,
        "total_params": total,
        "lora_r": lora_cfg["r"],
        "lora_alpha": lora_cfg["alpha"],
        "output_dir": str(output_dir),
    }

    results_path = output_dir / "training_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # --- Step 7: Report ---
    typer.echo("\n" + "=" * 60)
    typer.echo(f"SFT Stage {stage} Complete — {model_name}")
    typer.echo("=" * 60)
    typer.echo(f"Training loss:   {result.training_loss:.4f}")
    typer.echo(f"Train time:      {train_time / 3600:.1f}h ({train_time:.0f}s)")
    typer.echo(f"Samples:         {len(all_data)} train, {len(val_data)} val")
    typer.echo(f"Output:          {output_dir}")
    typer.echo(f"Results:         {results_path}")

    if report_to == "wandb":
        try:
            import wandb

            wandb.finish()
        except Exception:
            pass

    if stage == 1:
        typer.echo(f"\nNext: python scripts/sft_train.py train --model {model} --stage 2")
    else:
        typer.echo(f"\nNext: python scripts/merge_export.py merge --model {model}")


@app.command()
def info(
    model: str = typer.Option("qwen3-8b", help="Model key"),
) -> None:
    """Show training configuration without loading anything."""
    base_cfg, model_cfg = load_configs(model)

    typer.echo(f"Model: {model_cfg['model']['name']}")
    typer.echo(f"QLoRA: r={base_cfg['lora']['r']}, alpha={base_cfg['lora']['alpha']}")
    typer.echo(f"LR: {base_cfg['training']['learning_rate']}")
    typer.echo(f"Epochs: {base_cfg['training']['num_epochs']}")
    typer.echo(f"Batch: {base_cfg['training']['per_device_train_batch_size']} x {base_cfg['training']['gradient_accumulation_steps']} grad_accum")
    typer.echo(f"Max seq: {base_cfg['training']['max_seq_length']}")
    typer.echo(f"Packing: {base_cfg['training']['packing']}")

    typer.echo(f"\nStage 1 ({base_cfg['stage1']['name']}):")
    for k, v in base_cfg["stage1"]["mix"].items():
        typer.echo(f"  {k}: {v:.0%}")

    typer.echo(f"\nStage 2 ({base_cfg['stage2']['name']}):")
    for k, v in base_cfg["stage2"]["mix"].items():
        typer.echo(f"  {k}: {v:.0%}")


if __name__ == "__main__":
    app()
