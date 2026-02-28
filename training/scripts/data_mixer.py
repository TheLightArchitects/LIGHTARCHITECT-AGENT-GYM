"""Data mixer for SFT training stages.

Combines domain data, biblical data, trace data, and general-purpose data
into stage-specific training datasets with configurable proportions.

Stage 1 (Biblical Foundation): 70% biblical + 15% domain + 15% general
Stage 2 (Domain Integration): 50% domain + 20% general + 15% biblical + 15% trace

Usage:
    python scripts/data_mixer.py mix --stage 1 --output data/stage1_train.jsonl
    python scripts/data_mixer.py mix --stage 2 --output data/stage2_train.jsonl
    python scripts/data_mixer.py stats --stage 1
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typer
import yaml

app = typer.Typer(help="Data mixer for SFT training stages")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # LightArchitectsFoundationModel/


@dataclass
class MixStats:
    """Statistics for a data mix."""

    total_samples: int = 0
    source_counts: dict[str, int] = field(default_factory=dict)
    dedup_removed: int = 0
    avg_length: float = 0.0


def load_alpaca_file(path: Path) -> list[dict]:
    """Load an Alpaca-format JSON file."""
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return []


def load_bible_samples(base_cfg: dict) -> list[dict]:
    """Load all biblical training samples and normalize to Alpaca format."""
    samples = []
    data_cfg = base_cfg["data"]

    bible_files = [
        ("bible_verse_principle", "verse-to-principle"),
        ("bible_dilemma", "dilemma-to-wisdom"),
        ("bible_proverbs", "proverbs-to-guidance"),
        ("bible_narrative", "narrative-to-principle"),
    ]

    for config_key, source_name in bible_files:
        path = PROJECT_ROOT / data_cfg[config_key]
        if not path.exists():
            continue
        with open(path) as f:
            raw = json.load(f)
        for item in raw:
            sample = normalize_to_alpaca(item, f"biblical/{source_name}")
            if sample is not None:
                samples.append(sample)

    # Also include bible-alpaca from training-data/
    bible_alpaca_path = PROJECT_ROOT / data_cfg.get("bible_alpaca", "training-data/bible-alpaca.json")
    if bible_alpaca_path.exists():
        alpaca_samples = load_alpaca_file(bible_alpaca_path)
        for s in alpaca_samples:
            s.setdefault("source", "biblical/alpaca")
            samples.append(s)

    return samples


def load_domain_samples(base_cfg: dict) -> list[dict]:
    """Load domain training data (helix, quantum, tool-schemas, transcripts)."""
    data_cfg = base_cfg["data"]
    train_path = PROJECT_ROOT / data_cfg["domain_train"]
    if not train_path.exists():
        return []

    with open(train_path) as f:
        data = json.load(f)

    for s in data:
        s.setdefault("source", "domain")
    return data


def load_trace_samples(base_cfg: dict) -> list[dict]:
    """Load trace pattern training data."""
    data_cfg = base_cfg["data"]
    trace_path = PROJECT_ROOT / data_cfg.get("trace_alpaca", "training-data/traces-alpaca.json")
    if not trace_path.exists():
        return []

    samples = load_alpaca_file(trace_path)
    for s in samples:
        s.setdefault("source", "trace")
    return samples


def normalize_to_alpaca(item: dict, source: str) -> dict | None:
    """Normalize various formats to Alpaca format."""
    # Already Alpaca format
    if "instruction" in item and "output" in item:
        item.setdefault("input", "")
        item["source"] = source
        return item

    # Bible data format: {principle, verse_ref, verse_text, instruction, response}
    if "instruction" in item and "response" in item:
        return {
            "instruction": item["instruction"],
            "input": item.get("verse_text", ""),
            "output": item["response"],
            "source": source,
        }

    # DPO pair format (extract chosen as SFT sample)
    if "prompt" in item and "chosen" in item:
        return {
            "instruction": item["prompt"],
            "input": "",
            "output": item["chosen"],
            "source": source,
        }

    return None


def sample_to_hash(sample: dict) -> str:
    """SHA-256 hash for deduplication."""
    content = f"{sample.get('instruction', '')}{sample.get('input', '')}{sample.get('output', '')}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def deduplicate(samples: list[dict]) -> tuple[list[dict], int]:
    """Remove duplicate samples by content hash."""
    seen: set[str] = set()
    unique: list[dict] = []
    for s in samples:
        h = sample_to_hash(s)
        if h not in seen:
            seen.add(h)
            unique.append(s)
    return unique, len(samples) - len(unique)


def mix_for_stage(
    stage: int,
    base_cfg: dict,
    seed: int = 3407,
) -> tuple[list[dict], MixStats]:
    """Create the data mix for a given stage."""
    rng = random.Random(seed)
    stats = MixStats()

    # Load all data sources
    biblical = load_bible_samples(base_cfg)
    domain = load_domain_samples(base_cfg)
    trace = load_trace_samples(base_cfg)

    stats.source_counts["biblical_available"] = len(biblical)
    stats.source_counts["domain_available"] = len(domain)
    stats.source_counts["trace_available"] = len(trace)

    if stage == 1:
        mix_ratios = base_cfg["stage1"]["mix"]
        # Stage 1: 70% biblical + 15% domain + 15% general
        # General data is loaded from HuggingFace at training time,
        # so we prepare biblical + domain here
        target_total = max(len(biblical), 2000)  # at least 2000 samples

        n_biblical = int(target_total * mix_ratios["biblical"])
        n_domain = int(target_total * mix_ratios["domain"])

        # Sample with replacement if needed
        biblical_sampled = _sample_with_replacement(biblical, n_biblical, rng)
        domain_sampled = _sample_with_replacement(domain, n_domain, rng)

        mixed = biblical_sampled + domain_sampled
        stats.source_counts["biblical_used"] = len(biblical_sampled)
        stats.source_counts["domain_used"] = len(domain_sampled)
        stats.source_counts["general_note"] = "loaded from HuggingFace at training time"

    elif stage == 2:
        mix_ratios = base_cfg["stage2"]["mix"]
        # Stage 2: 50% domain + 20% general + 15% biblical replay + 15% trace
        target_total = max(len(domain) * 2, 2000)

        n_domain = int(target_total * mix_ratios["domain"])
        n_biblical = int(target_total * mix_ratios["biblical_replay"])
        n_trace = int(target_total * mix_ratios["trace_patterns"])

        domain_sampled = _sample_with_replacement(domain, n_domain, rng)
        biblical_sampled = _sample_with_replacement(biblical, n_biblical, rng)
        trace_sampled = _sample_with_replacement(trace, n_trace, rng) if trace else []

        mixed = domain_sampled + biblical_sampled + trace_sampled
        stats.source_counts["domain_used"] = len(domain_sampled)
        stats.source_counts["biblical_replay_used"] = len(biblical_sampled)
        stats.source_counts["trace_used"] = len(trace_sampled)
        stats.source_counts["general_note"] = "loaded from HuggingFace at training time"
    else:
        raise ValueError(f"Unknown stage: {stage}. Expected 1 or 2.")

    # Deduplicate
    mixed, removed = deduplicate(mixed)
    stats.dedup_removed = removed

    # Shuffle
    rng.shuffle(mixed)

    stats.total_samples = len(mixed)
    if mixed:
        lengths = [len(s.get("output", "")) for s in mixed]
        stats.avg_length = sum(lengths) / len(lengths)

    return mixed, stats


def _sample_with_replacement(data: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Sample n items, with replacement if n > len(data)."""
    if not data:
        return []
    if n <= len(data):
        return rng.sample(data, n)
    # Need more than available — sample with replacement
    return [rng.choice(data) for _ in range(n)]


@app.command()
def mix(
    stage: int = typer.Option(..., help="Training stage (1 or 2)"),
    output: str = typer.Option(..., help="Output JSONL file path"),
    seed: int = typer.Option(3407, help="Random seed"),
) -> None:
    """Create a mixed dataset for the specified training stage."""
    base_cfg_path = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
    with open(base_cfg_path) as f:
        base_cfg = yaml.safe_load(f)

    typer.echo(f"=== Mixing data for Stage {stage} ===")

    mixed, stats = mix_for_stage(stage, base_cfg, seed)

    # Write output
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for sample in mixed:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    typer.echo(f"Total samples:    {stats.total_samples}")
    typer.echo(f"Dedup removed:    {stats.dedup_removed}")
    typer.echo(f"Avg output len:   {stats.avg_length:.0f} chars")
    for key, val in stats.source_counts.items():
        typer.echo(f"  {key}: {val}")
    typer.echo(f"\nWritten to: {output_path}")


@app.command()
def stats(
    stage: int = typer.Option(..., help="Training stage (1 or 2)"),
    seed: int = typer.Option(3407, help="Random seed"),
) -> None:
    """Show data mix statistics without writing files."""
    base_cfg_path = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
    with open(base_cfg_path) as f:
        base_cfg = yaml.safe_load(f)

    typer.echo(f"=== Stage {stage} Data Mix Statistics ===")

    _, mix_stats = mix_for_stage(stage, base_cfg, seed)

    typer.echo(f"Total samples:    {mix_stats.total_samples}")
    typer.echo(f"Dedup removed:    {mix_stats.dedup_removed}")
    typer.echo(f"Avg output len:   {mix_stats.avg_length:.0f} chars")
    for key, val in mix_stats.source_counts.items():
        typer.echo(f"  {key}: {val}")


if __name__ == "__main__":
    app()
