"""Unsloth smoke test — verify QLoRA loads and trains for 100 steps.

Tasks 7.1/7.2: RunPod A100 setup + Unsloth smoke test.
Run after setup_runpod.sh to confirm the training pipeline works.

Usage:
    python scripts/smoke_test.py --model qwen3-8b
    python scripts/smoke_test.py --model llama31-8b
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer

from common import MODEL_CONFIGS, load_configs

app = typer.Typer(help="Unsloth smoke test for Light Architects models")


def create_dummy_dataset(num_samples: int = 200) -> list[dict]:
    """Create a small synthetic dataset for smoke testing."""
    samples = []
    for i in range(num_samples):
        samples.append(
            {
                "instruction": f"Explain principle {i % 7 + 1} of the Biblical Constitution.",
                "input": "",
                "output": f"Principle {i % 7 + 1} teaches us about faithfulness and stewardship. "
                f"As Proverbs 27:23-24 says, we must be diligent to know the state "
                f"of our flocks. This applies to how we build AI systems — with care, "
                f"integrity, and responsibility. Sample {i}.",
            }
        )
    return samples


@app.command()
def run(
    model: str = typer.Option(
        "mixtral-8x7b",
        help="Model to test: mixtral-8x7b, qwen3-8b, or llama31-8b",
    ),
    max_steps: int = typer.Option(100, help="Number of training steps"),
    dry_run: bool = typer.Option(False, help="Skip actual training, just verify load"),
) -> None:
    """Run Unsloth smoke test."""
    if model not in MODEL_CONFIGS:
        typer.echo(f"Unknown model: {model}. Choose from: {list(MODEL_CONFIGS.keys())}")
        raise typer.Exit(1)

    base_cfg, model_cfg = load_configs(model)
    model_name = model_cfg["model"]["name"]
    max_seq_length = base_cfg["training"]["max_seq_length"]
    lora_cfg = base_cfg["lora"]

    typer.echo(f"=== Smoke Test: {model_name} ===")
    typer.echo(f"Max steps: {max_steps}")
    typer.echo(f"Max seq length: {max_seq_length}")
    typer.echo(f"QLoRA r={lora_cfg['r']}, alpha={lora_cfg['alpha']}")
    typer.echo("")

    # Import here so --help works without GPU
    from unsloth import FastLanguageModel

    # Step 1: Load model
    typer.echo("--- Loading model with 4-bit quantization ---")
    t0 = time.time()
    model_obj, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # auto-detect
        load_in_4bit=model_cfg["model"]["load_in_4bit"],
        trust_remote_code=model_cfg["model"].get("trust_remote_code", False),
    )
    load_time = time.time() - t0
    typer.echo(f"Model loaded in {load_time:.1f}s")

    # Step 2: Apply LoRA
    typer.echo("--- Applying QLoRA adapters ---")
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

    # Print trainable params
    trainable = sum(p.numel() for p in model_obj.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model_obj.parameters())
    typer.echo(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    if dry_run:
        typer.echo("\n--- Dry run complete (model loads, LoRA applies) ---")
        raise typer.Exit(0)

    # Step 3: Create dummy dataset
    typer.echo("--- Creating dummy dataset ---")
    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    dummy_data = create_dummy_dataset(200)

    def formatting_func(examples: dict) -> list[str]:
        """Format samples using the model's native chat template."""
        texts = []
        for instruction, inp, output in zip(
            examples["instruction"], examples["input"], examples["output"]
        ):
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

    dataset = Dataset.from_list(dummy_data)

    # Step 4: Train for max_steps
    typer.echo(f"--- Training for {max_steps} steps ---")
    output_dir = f"/tmp/smoke-test-{model}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=base_cfg["training"]["learning_rate"],
        lr_scheduler_type=base_cfg["training"]["lr_scheduler_type"],
        warmup_ratio=base_cfg["training"]["warmup_ratio"],
        optim=base_cfg["training"]["optim"],
        bf16=base_cfg["training"]["bf16"],
        logging_steps=10,
        save_steps=max_steps,
        seed=base_cfg["training"]["seed"],
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model_obj,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=formatting_func,
        max_seq_length=max_seq_length,
        packing=base_cfg["training"]["packing"],
        args=training_args,
    )

    t0 = time.time()
    result = trainer.train()
    train_time = time.time() - t0

    # Step 5: Report
    typer.echo("\n=== Smoke Test Results ===")
    typer.echo(f"Model:          {model_name}")
    typer.echo(f"Steps:          {max_steps}")
    typer.echo(f"Training loss:  {result.training_loss:.4f}")
    typer.echo(f"Train time:     {train_time:.1f}s")
    typer.echo(f"Steps/sec:      {max_steps / train_time:.2f}")
    typer.echo(f"Output dir:     {output_dir}")

    # Write results to JSON for later reference
    results = {
        "model": model_name,
        "model_key": model,
        "max_steps": max_steps,
        "training_loss": result.training_loss,
        "train_time_seconds": train_time,
        "steps_per_second": max_steps / train_time,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": 100 * trainable / total,
    }

    results_path = Path(output_dir) / "smoke_test_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    typer.echo(f"\nResults saved to: {results_path}")
    typer.echo("SMOKE TEST PASSED")


if __name__ == "__main__":
    app()
