"""CLI interface for the biblical training data pipeline.

Usage:
    python -m bible_data generate --type verse-to-principle|dilemma|proverbs|narrative|all
    python -m bible_data dpo --output bible-dpo.json
    python -m bible_data integrate --datasets bible-dpo,ethics
    python -m bible_data stats --input output/
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from bible_data.dpo_generator import generate_dpo_pairs
from bible_data.instruction_generator import (
    generate_all,
    generate_dilemma_to_wisdom,
    generate_narrative_to_principle,
    generate_proverbs_to_guidance,
    generate_verse_to_principle,
)
from bible_data.models import GenerationStats

app = typer.Typer(
    name="bible-data",
    help="Biblical training data pipeline for moral reasoning in AI systems.",
    no_args_is_help=True,
)
console = Console()

DEFAULT_OUTPUT = Path("/Users/kft/Projects/LightArchitectsFoundationModel/bible-data/output")


def _ensure_output_dir(output: Path) -> None:
    """Create output directory if it doesn't exist."""
    output.mkdir(parents=True, exist_ok=True)


def _save_json(data: list[dict], path: Path) -> int:
    """Save data to JSON file, return count."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return len(data)


@app.command()
def generate(
    pair_type: Annotated[
        str,
        typer.Option("--type", "-t", help="Type of pairs to generate"),
    ] = "all",
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory"),
    ] = DEFAULT_OUTPUT,
    kjv_path: Annotated[
        Optional[str],
        typer.Option("--kjv", help="Path to KJV Bible JSON"),
    ] = None,
) -> None:
    """Generate instruction-response training pairs."""
    _ensure_output_dir(output)

    valid_types = {"verse-to-principle", "dilemma", "proverbs", "narrative", "all"}
    if pair_type not in valid_types:
        console.print(f"[red]Invalid type: {pair_type}[/red]")
        console.print(f"Valid types: {', '.join(sorted(valid_types))}")
        raise typer.Exit(code=1)

    stats = GenerationStats()

    if pair_type == "all":
        console.print("[bold]Generating all instruction pair types...[/bold]")
        results = generate_all(kjv_path)

        for name, pairs in results.items():
            filename = {
                "verse-to-principle": "verse-to-principle.json",
                "dilemma": "dilemma-to-wisdom.json",
                "proverbs": "proverbs-to-guidance.json",
                "narrative": "narrative-to-principle.json",
            }[name]

            data = [p.model_dump() for p in pairs]
            count = _save_json(data, output / filename)
            console.print(f"  {name}: {count} pairs -> {filename}")

            if name == "verse-to-principle":
                stats.verse_to_principle = count
            elif name == "dilemma":
                stats.dilemma_to_wisdom = count
            elif name == "proverbs":
                stats.proverbs_to_guidance = count
            elif name == "narrative":
                stats.narrative_to_principle = count
    else:
        console.print(f"[bold]Generating {pair_type} pairs...[/bold]")
        generators = {
            "verse-to-principle": (generate_verse_to_principle, "verse-to-principle.json"),
            "dilemma": (generate_dilemma_to_wisdom, "dilemma-to-wisdom.json"),
            "proverbs": (generate_proverbs_to_guidance, "proverbs-to-guidance.json"),
            "narrative": (generate_narrative_to_principle, "narrative-to-principle.json"),
        }

        gen_func, filename = generators[pair_type]
        if pair_type == "verse-to-principle":
            pairs = gen_func(kjv_path)
        else:
            pairs = gen_func()

        data = [p.model_dump() for p in pairs]
        count = _save_json(data, output / filename)
        console.print(f"  Generated {count} pairs -> {filename}")

    stats.compute_total()
    console.print(f"\n[green]Total instruction pairs: {stats.total}[/green]")


@app.command()
def dpo(
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output file path"),
    ] = DEFAULT_OUTPUT / "bible-dpo-pairs.json",
) -> None:
    """Generate DPO preference pairs."""
    output.parent.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Generating DPO preference pairs...[/bold]")
    pairs = generate_dpo_pairs()

    data = [p.model_dump() for p in pairs]
    count = _save_json(data, output)

    console.print(f"[green]Generated {count} DPO pairs -> {output}[/green]")

    # Print principle distribution
    principle_counts: dict[str, int] = {}
    for pair in pairs:
        p = pair.principle.value
        principle_counts[p] = principle_counts.get(p, 0) + 1

    table = Table(title="DPO Pairs by Principle")
    table.add_column("Principle", style="cyan")
    table.add_column("Count", justify="right", style="green")

    for principle, count in sorted(principle_counts.items()):
        table.add_row(principle, str(count))

    console.print(table)


@app.command()
def integrate(
    datasets: Annotated[
        str,
        typer.Option("--datasets", "-d", help="Comma-separated dataset names"),
    ] = "bible-dpo",
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory"),
    ] = DEFAULT_OUTPUT,
) -> None:
    """Integrate external HuggingFace datasets with generated data."""
    from bible_data.dataset_integrator import (
        load_external_dataset,
        merge_datasets,
        save_merged_dataset,
    )

    _ensure_output_dir(output)

    dataset_names = [d.strip() for d in datasets.split(",")]
    console.print(f"[bold]Integrating datasets: {', '.join(dataset_names)}[/bold]")

    # Load generated data if it exists
    generated: list[dict] = []
    for json_file in output.glob("*.json"):
        if json_file.name != "integrated.json":
            with open(json_file, encoding="utf-8") as f:
                generated.extend(json.load(f))

    console.print(f"  Loaded {len(generated)} generated records")

    # Load external datasets
    all_external: list[dict] = []
    for name in dataset_names:
        try:
            records = load_external_dataset(name)
            console.print(f"  Loaded {len(records)} records from {name}")
            all_external.extend(records)
        except (ImportError, ValueError) as e:
            console.print(f"  [yellow]Warning: Could not load {name}: {e}[/yellow]")
        except Exception as e:  # noqa: BLE001
            console.print(f"  [yellow]Warning: Error loading {name}: {e}[/yellow]")

    # Merge and deduplicate
    merged, removed = merge_datasets(generated, all_external)
    save_merged_dataset(merged, output / "integrated.json")

    console.print(f"\n[green]Merged: {len(merged)} records (removed {removed} duplicates)[/green]")
    console.print(f"  Output: {output / 'integrated.json'}")


@app.command()
def stats(
    input_dir: Annotated[
        Path,
        typer.Option("--input", "-i", help="Input directory with JSON files"),
    ] = DEFAULT_OUTPUT,
) -> None:
    """Display statistics for generated training data."""
    if not input_dir.exists():
        console.print(f"[red]Directory not found: {input_dir}[/red]")
        raise typer.Exit(code=1)

    table = Table(title="Biblical Training Data Statistics")
    table.add_column("File", style="cyan")
    table.add_column("Records", justify="right", style="green")
    table.add_column("Size", justify="right", style="yellow")

    total_records = 0
    total_size = 0

    for json_file in sorted(input_dir.glob("*.json")):
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        count = len(data) if isinstance(data, list) else 1
        size = json_file.stat().st_size
        total_records += count
        total_size += size

        table.add_row(
            json_file.name,
            str(count),
            _format_size(size),
        )

    table.add_row("", "", "")
    table.add_row("[bold]TOTAL[/bold]", f"[bold]{total_records}[/bold]", f"[bold]{_format_size(total_size)}[/bold]")

    console.print(table)

    # Principle distribution across all files
    principle_counts: dict[str, int] = {}
    for json_file in input_dir.glob("*.json"):
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for record in data:
                if "principle" in record:
                    p = record["principle"]
                    principle_counts[p] = principle_counts.get(p, 0) + 1

    if principle_counts:
        ptable = Table(title="Records by Principle")
        ptable.add_column("Principle", style="cyan")
        ptable.add_column("Count", justify="right", style="green")

        for principle, count in sorted(principle_counts.items()):
            ptable.add_row(principle, str(count))

        console.print(ptable)


def _format_size(size_bytes: int) -> str:
    """Format byte count as human-readable size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


if __name__ == "__main__":
    app()
