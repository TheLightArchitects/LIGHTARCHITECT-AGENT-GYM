"""Checkpoint merge + GGUF export — sovereign-forging-lion Task 7.7.

Merges QLoRA adapters back into base model (16-bit) and exports to GGUF
for llama.cpp / Ollama testing.

Usage:
    # Merge Stage 2 adapters into full 16-bit model
    python scripts/merge_export.py merge --model qwen3-8b

    # Export merged model to GGUF (Q4_K_M and Q8_0)
    python scripts/merge_export.py gguf --model qwen3-8b

    # Full pipeline: merge + export
    python scripts/merge_export.py full --model qwen3-8b
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import typer

from common import MODEL_CONFIGS, load_configs, TRAINING_ROOT

app = typer.Typer(help="Merge QLoRA adapters and export to GGUF")


@app.command()
def merge(
    model: str = typer.Option(..., help="Model key: qwen3-8b or llama31-8b"),
    stage: int = typer.Option(2, help="Which stage checkpoint to merge (default: 2)"),
) -> None:
    """Merge QLoRA adapters into full 16-bit model."""
    if model not in MODEL_CONFIGS:
        typer.echo(f"Unknown model: {model}")
        raise typer.Exit(1)

    base_cfg, model_cfg = load_configs(model)
    output_cfg = model_cfg["output"]

    adapter_dir = TRAINING_ROOT / output_cfg[f"stage{stage}_dir"]
    merged_dir = TRAINING_ROOT / output_cfg["merged_dir"]

    if not adapter_dir.exists():
        typer.echo(f"Adapter directory not found: {adapter_dir}")
        typer.echo(f"Run sft_train.py first for stage {stage}")
        raise typer.Exit(1)

    typer.echo(f"=== Merging {model} Stage {stage} ===")
    typer.echo(f"Adapter:  {adapter_dir}")
    typer.echo(f"Output:   {merged_dir}")

    from unsloth import FastLanguageModel

    t0 = time.time()

    # Load the adapter checkpoint
    model_obj, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=base_cfg["training"]["max_seq_length"],
        dtype=None,
        load_in_4bit=False,  # Load in 16-bit for merging
    )

    # Merge and save
    merged_dir.mkdir(parents=True, exist_ok=True)
    model_obj.save_pretrained_merged(
        str(merged_dir),
        tokenizer,
        save_method="merged_16bit",
    )

    merge_time = time.time() - t0
    typer.echo(f"\nMerge complete in {merge_time:.0f}s")
    typer.echo(f"16-bit model saved to: {merged_dir}")

    # Save merge metadata
    meta = {
        "model_key": model,
        "model_name": model_cfg["model"]["name"],
        "adapter_source": str(adapter_dir),
        "merged_to": str(merged_dir),
        "merge_time_seconds": merge_time,
        "stage_merged": stage,
        "lora_r": base_cfg["lora"]["r"],
        "lora_alpha": base_cfg["lora"]["alpha"],
    }
    with open(merged_dir / "merge_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    typer.echo(f"\nNext: python scripts/merge_export.py gguf --model {model}")


@app.command()
def gguf(
    model: str = typer.Option(..., help="Model key: qwen3-8b or llama31-8b"),
    quantizations: str = typer.Option(
        "q4_k_m,q8_0",
        help="Comma-separated GGUF quantization types",
    ),
) -> None:
    """Export merged model to GGUF format."""
    if model not in MODEL_CONFIGS:
        typer.echo(f"Unknown model: {model}")
        raise typer.Exit(1)

    base_cfg, model_cfg = load_configs(model)
    output_cfg = model_cfg["output"]

    merged_dir = TRAINING_ROOT / output_cfg["merged_dir"]
    gguf_dir = TRAINING_ROOT / output_cfg["gguf_dir"]

    if not merged_dir.exists():
        typer.echo(f"Merged model not found: {merged_dir}")
        typer.echo("Run merge first: python scripts/merge_export.py merge --model " + model)
        raise typer.Exit(1)

    quant_types = [q.strip() for q in quantizations.split(",")]

    typer.echo(f"=== GGUF Export: {model} ===")
    typer.echo(f"Source:       {merged_dir}")
    typer.echo(f"Output:       {gguf_dir}")
    typer.echo(f"Quantizations: {quant_types}")

    from unsloth import FastLanguageModel

    # Load merged model
    model_obj, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(merged_dir),
        max_seq_length=base_cfg["training"]["max_seq_length"],
        dtype=None,
        load_in_4bit=False,
    )

    gguf_dir.mkdir(parents=True, exist_ok=True)

    for quant in quant_types:
        typer.echo(f"\n--- Exporting {quant} ---")
        t0 = time.time()

        model_obj.save_pretrained_gguf(
            str(gguf_dir),
            tokenizer,
            quantization_method=quant,
        )

        export_time = time.time() - t0
        typer.echo(f"Exported {quant} in {export_time:.0f}s")

    # List exported files
    typer.echo("\n--- Exported GGUF files ---")
    for f in sorted(gguf_dir.glob("*.gguf")):
        size_mb = f.stat().st_size / (1024 * 1024)
        typer.echo(f"  {f.name}: {size_mb:.0f} MB")

    # Save export metadata
    meta = {
        "model_key": model,
        "model_name": model_cfg["model"]["name"],
        "model_id": output_cfg["model_id"],
        "source": str(merged_dir),
        "quantizations": quant_types,
        "gguf_dir": str(gguf_dir),
    }
    with open(gguf_dir / "export_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


@app.command()
def full(
    model: str = typer.Option(..., help="Model key: qwen3-8b or llama31-8b"),
    stage: int = typer.Option(2, help="Which stage to merge"),
    quantizations: str = typer.Option("q4_k_m,q8_0", help="GGUF quantizations"),
) -> None:
    """Full pipeline: merge adapters + export GGUF."""
    typer.echo(f"=== Full Merge + Export: {model} ===\n")

    # Merge
    merge(model=model, stage=stage)

    typer.echo("\n")

    # Export
    gguf(model=model, quantizations=quantizations)

    typer.echo("\n=== Full Pipeline Complete ===")
    _, model_cfg = load_configs(model)
    model_id = model_cfg["output"]["model_id"]
    typer.echo(f"Model ID: {model_id}")
    typer.echo(f"Ready for Ollama: ollama create {model_id} -f Modelfile")


if __name__ == "__main__":
    app()
