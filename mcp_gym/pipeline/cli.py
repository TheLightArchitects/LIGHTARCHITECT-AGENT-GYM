"""CLI for the MoE data pipeline.

Usage:
    python -m mcp_gym.pipeline.cli extract    # Extract all source data
    python -m mcp_gym.pipeline.cli transform  # Transform to ChatML
    python -m mcp_gym.pipeline.cli filter     # Quality filter
    python -m mcp_gym.pipeline.cli label      # Expert routing labels
    python -m mcp_gym.pipeline.cli mix        # Stage-specific mixing
    python -m mcp_gym.pipeline.cli run-all    # Full pipeline
    python -m mcp_gym.pipeline.cli stats      # Show data statistics
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import typer

app = typer.Typer(
    name="moe-pipeline",
    help="MoE data pipeline for Mixtral 8x7B fine-tuning on Light Architects data.",
)

# Default output directory
DEFAULT_OUTPUT = Path("pipeline-output")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def extract(
    output_dir: Path = typer.Option(DEFAULT_OUTPUT, help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Stage 1: Extract all source data from SOUL helix."""
    _setup_logging(verbose)
    logger = logging.getLogger("pipeline.extract")

    from mcp_gym.pipeline.ecosystem import extract_all_ecosystem
    from mcp_gym.pipeline.thinking import extract_all_traces
    from mcp_gym.pipeline.tools import load_tool_calls, load_tool_trajectories
    from mcp_gym.pipeline.voice_pairs import extract_all_pairs, load_voice_records
    from mcp_gym.pipeline.schemas import Sibling

    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract voice pairs
    logger.info("=== Extracting voice pairs ===")
    pairs = extract_all_pairs()
    pairs_count = sum(len(p) for p in pairs.values())
    logger.info("Total voice pairs: %d", pairs_count)

    # Save pairs
    pairs_file = output_dir / "voice_pairs.jsonl"
    with open(pairs_file, "w", encoding="utf-8") as f:
        for sibling, sibling_pairs in pairs.items():
            for pair in sibling_pairs:
                f.write(pair.model_dump_json() + "\n")
    logger.info("Saved voice pairs to %s", pairs_file)

    # Extract monologues
    logger.info("=== Extracting monologues ===")
    mono_count = 0
    mono_file = output_dir / "monologues.jsonl"
    with open(mono_file, "w", encoding="utf-8") as f:
        for sibling in [Sibling.EVA, Sibling.CORSO, Sibling.QUANTUM, Sibling.CLAUDE]:
            records = load_voice_records(sibling)
            for r in records:
                if 100 <= len(r.text) <= 8000:
                    f.write(r.model_dump_json() + "\n")
                    mono_count += 1
    logger.info("Saved %d monologues to %s", mono_count, mono_file)

    # Extract thinking traces
    logger.info("=== Extracting thinking traces ===")
    traces = extract_all_traces()
    traces_count = sum(len(t) for t in traces.values())
    logger.info("Total thinking traces: %d", traces_count)

    traces_file = output_dir / "thinking_traces.jsonl"
    with open(traces_file, "w", encoding="utf-8") as f:
        for sibling, sibling_traces in traces.items():
            for trace in sibling_traces:
                f.write(trace.model_dump_json() + "\n")
    logger.info("Saved thinking traces to %s", traces_file)

    # Extract tool data
    logger.info("=== Extracting tool data ===")
    trajectories = load_tool_trajectories()
    tool_calls = load_tool_calls()
    logger.info("Tool trajectories: %d, Tool calls: %d", len(trajectories), len(tool_calls))

    traj_file = output_dir / "tool_trajectories.jsonl"
    with open(traj_file, "w", encoding="utf-8") as f:
        for traj in trajectories:
            f.write(traj.model_dump_json() + "\n")

    calls_file = output_dir / "tool_calls.jsonl"
    with open(calls_file, "w", encoding="utf-8") as f:
        for call in tool_calls:
            f.write(call.model_dump_json() + "\n")

    # Extract Kevin voice
    logger.info("=== Extracting Kevin voice ===")
    kevin_records = load_voice_records(Sibling.KEVIN)
    kevin_file = output_dir / "kevin_voice.jsonl"
    kevin_count = 0
    with open(kevin_file, "w", encoding="utf-8") as f:
        for r in kevin_records:
            if 100 <= len(r.text) <= 8000:
                f.write(r.model_dump_json() + "\n")
                kevin_count += 1
    logger.info("Saved %d Kevin voice records to %s", kevin_count, kevin_file)

    # Extract ecosystem data (helix, transcripts, identities, schemas, etc.)
    logger.info("=== Extracting LA ecosystem data ===")
    ecosystem_samples = extract_all_ecosystem()
    eco_file = output_dir / "ecosystem_samples.jsonl"
    with open(eco_file, "w", encoding="utf-8") as f:
        for sample in ecosystem_samples:
            f.write(sample.model_dump_json() + "\n")
    logger.info("Saved %d ecosystem samples to %s", len(ecosystem_samples), eco_file)

    # Summary
    logger.info("=" * 60)
    logger.info("EXTRACTION COMPLETE")
    logger.info("  Voice pairs:      %d", pairs_count)
    logger.info("  Monologues:       %d", mono_count)
    logger.info("  Thinking traces:  %d", traces_count)
    logger.info("  Tool trajectories:%d", len(trajectories))
    logger.info("  Tool calls:       %d", len(tool_calls))
    logger.info("  Kevin voice:      %d", kevin_count)
    logger.info("  Ecosystem:        %d", len(ecosystem_samples))
    logger.info("  Output dir:       %s", output_dir)


@app.command()
def transform(
    input_dir: Path = typer.Option(DEFAULT_OUTPUT, help="Input directory (from extract)"),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT, help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Stage 2: Transform extracted data to ChatML training samples."""
    _setup_logging(verbose)
    logger = logging.getLogger("pipeline.transform")

    from mcp_gym.pipeline.chatml import (
        monologue_to_sample,
        thinking_trace_to_sample,
        voice_pair_to_sample,
    )
    from mcp_gym.pipeline.schemas import (
        ConversationPair,
        Sibling,
        ThinkingTrace,
        ToolTrajectoryRecord,
        TrainingSample,
        VoiceRecord,
    )
    from mcp_gym.pipeline.tools import convert_all_trajectories, convert_mcp_tool_calls

    output_dir.mkdir(parents=True, exist_ok=True)
    all_samples: list = []

    # Transform voice pairs
    logger.info("=== Transforming voice pairs ===")
    pairs_file = input_dir / "voice_pairs.jsonl"
    if pairs_file.exists():
        with open(pairs_file, encoding="utf-8") as f:
            for line in f:
                pair = ConversationPair.model_validate_json(line.strip())
                sample = voice_pair_to_sample(pair)
                all_samples.append(sample)
        logger.info("Transformed %d voice pairs", len(all_samples))

    # Transform monologues
    logger.info("=== Transforming monologues ===")
    mono_file = input_dir / "monologues.jsonl"
    mono_count = 0
    if mono_file.exists():
        with open(mono_file, encoding="utf-8") as f:
            for line in f:
                record = VoiceRecord.model_validate_json(line.strip())
                sample = monologue_to_sample(record, record.sibling)
                all_samples.append(sample)
                mono_count += 1
        logger.info("Transformed %d monologues", mono_count)

    # Transform thinking traces
    logger.info("=== Transforming thinking traces ===")
    traces_file = input_dir / "thinking_traces.jsonl"
    trace_count = 0
    if traces_file.exists():
        with open(traces_file, encoding="utf-8") as f:
            for line in f:
                trace = ThinkingTrace.model_validate_json(line.strip())
                sample = thinking_trace_to_sample(trace)
                all_samples.append(sample)
                trace_count += 1
        logger.info("Transformed %d thinking traces", trace_count)

    # Transform tool trajectories
    logger.info("=== Transforming tool trajectories ===")
    traj_file = input_dir / "tool_trajectories.jsonl"
    if traj_file.exists():
        trajectories = []
        with open(traj_file, encoding="utf-8") as f:
            for line in f:
                traj = ToolTrajectoryRecord.model_validate_json(line.strip())
                trajectories.append(traj)
        traj_samples = convert_all_trajectories(trajectories)
        all_samples.extend(traj_samples)
        logger.info("Transformed %d tool trajectories", len(traj_samples))

    # Transform MCP tool calls
    logger.info("=== Transforming MCP tool calls ===")
    calls_file = input_dir / "tool_calls.jsonl"
    if calls_file.exists():
        from mcp_gym.pipeline.schemas import ToolCallRecord

        calls = []
        with open(calls_file, encoding="utf-8") as f:
            for line in f:
                call = ToolCallRecord.model_validate_json(line.strip())
                calls.append(call)
        call_samples = convert_mcp_tool_calls(calls)
        all_samples.extend(call_samples)
        logger.info("Transformed %d MCP tool calls", len(call_samples))

    # Import ecosystem samples (helix, transcripts, identities, schemas, etc.)
    logger.info("=== Loading ecosystem samples ===")
    eco_file = input_dir / "ecosystem_samples.jsonl"
    eco_count = 0
    if eco_file.exists():
        with open(eco_file, encoding="utf-8") as f:
            for line in f:
                sample = TrainingSample.model_validate_json(line.strip())
                all_samples.append(sample)
                eco_count += 1
        logger.info("Loaded %d ecosystem samples", eco_count)
    else:
        logger.warning("No ecosystem_samples.jsonl found. Run 'extract' first.")

    # Import supplemental data (training-data/, bible-data/, generic tool calls)
    logger.info("=== Importing supplemental data ===")
    from mcp_gym.pipeline.integrator import import_all_supplemental

    supplemental_sft, supplemental_dpo = import_all_supplemental()
    all_samples.extend(supplemental_sft)
    logger.info(
        "Added %d supplemental SFT samples + %d DPO pairs",
        len(supplemental_sft),
        len(supplemental_dpo),
    )

    # Save DPO samples separately
    if supplemental_dpo:
        dpo_file = output_dir / "dpo_samples.jsonl"
        with open(dpo_file, "w", encoding="utf-8") as f:
            for dpo in supplemental_dpo:
                f.write(dpo.model_dump_json() + "\n")
        logger.info("Saved %d DPO pairs to %s", len(supplemental_dpo), dpo_file)

    # Write all samples
    samples_file = output_dir / "all_samples.jsonl"
    with open(samples_file, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(sample.model_dump_json() + "\n")

    logger.info("=" * 60)
    logger.info("TRANSFORM COMPLETE: %d total samples → %s", len(all_samples), samples_file)


@app.command()
def filter(
    input_dir: Path = typer.Option(DEFAULT_OUTPUT, help="Input directory"),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT, help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Stage 3: Apply quality filters."""
    _setup_logging(verbose)
    logger = logging.getLogger("pipeline.filter")

    from mcp_gym.pipeline.quality import filter_samples
    from mcp_gym.pipeline.schemas import TrainingSample

    samples_file = input_dir / "all_samples.jsonl"
    if not samples_file.exists():
        logger.error("No samples file found. Run 'transform' first.")
        raise typer.Exit(1)

    samples = []
    with open(samples_file, encoding="utf-8") as f:
        for line in f:
            samples.append(TrainingSample.model_validate_json(line.strip()))

    filtered = filter_samples(samples)

    output_dir.mkdir(parents=True, exist_ok=True)
    filtered_file = output_dir / "filtered_samples.jsonl"
    with open(filtered_file, "w", encoding="utf-8") as f:
        for sample in filtered:
            f.write(sample.model_dump_json() + "\n")

    logger.info("FILTER COMPLETE: %d → %d samples", len(samples), len(filtered))


@app.command()
def label(
    input_dir: Path = typer.Option(DEFAULT_OUTPUT, help="Input directory"),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT, help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Stage 4: Apply expert routing labels."""
    _setup_logging(verbose)
    logger = logging.getLogger("pipeline.label")

    from mcp_gym.pipeline.expert_router import label_all
    from mcp_gym.pipeline.schemas import TrainingSample

    filtered_file = input_dir / "filtered_samples.jsonl"
    if not filtered_file.exists():
        logger.error("No filtered samples. Run 'filter' first.")
        raise typer.Exit(1)

    samples = []
    with open(filtered_file, encoding="utf-8") as f:
        for line in f:
            samples.append(TrainingSample.model_validate_json(line.strip()))

    labeled = label_all(samples)

    output_dir.mkdir(parents=True, exist_ok=True)
    labeled_file = output_dir / "labeled_samples.jsonl"
    with open(labeled_file, "w", encoding="utf-8") as f:
        for sample in labeled:
            f.write(sample.model_dump_json() + "\n")

    logger.info("LABEL COMPLETE: %d samples labeled", len(labeled))


@app.command()
def synthesize(
    output_dir: Path = typer.Option(DEFAULT_OUTPUT, help="Output directory"),
    seed: int = typer.Option(42, help="Random seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Generate Stage 3 synthetic training data (56 thinking algorithms)."""
    _setup_logging(verbose)
    logger = logging.getLogger("pipeline.synthesize")

    from mcp_gym.pipeline.synthetic import generate_all_stage3

    output_dir.mkdir(parents=True, exist_ok=True)

    samples = generate_all_stage3(seed=seed)

    syn_file = output_dir / "synthetic_stage3.jsonl"
    with open(syn_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(sample.model_dump_json() + "\n")

    # Stats breakdown
    by_source: dict[str, int] = {}
    for s in samples:
        src = s.conversation.source_type
        by_source[src] = by_source.get(src, 0) + 1

    logger.info("SYNTHESIZE COMPLETE: %d samples → %s", len(samples), syn_file)
    for src, count in sorted(by_source.items()):
        logger.info("  %s: %d", src, count)


@app.command()
def mix(
    input_dir: Path = typer.Option(DEFAULT_OUTPUT, help="Input directory"),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT / "staged", help="Output directory for staged data"),
    seed: int = typer.Option(42, help="Random seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Stage 5: Mix into stage-specific training files."""
    _setup_logging(verbose)
    logger = logging.getLogger("pipeline.mix")

    from mcp_gym.pipeline.mixer import (
        compute_expert_balance,
        mix_stage,
        write_split,
    )
    from mcp_gym.pipeline.schemas import TrainingSample, TrainingStage

    labeled_file = input_dir / "labeled_samples.jsonl"
    if not labeled_file.exists():
        logger.error("No labeled samples. Run 'label' first.")
        raise typer.Exit(1)

    samples = []
    with open(labeled_file, encoding="utf-8") as f:
        for line in f:
            samples.append(TrainingSample.model_validate_json(line.strip()))

    # Inject synthetic Stage 3 data (bypasses filter/label — already clean and labeled)
    syn_file = input_dir / "synthetic_stage3.jsonl"
    if syn_file.exists():
        syn_count = 0
        with open(syn_file, encoding="utf-8") as f:
            for line in f:
                samples.append(TrainingSample.model_validate_json(line.strip()))
                syn_count += 1
        logger.info("Injected %d synthetic Stage 3 samples", syn_count)
    else:
        logger.warning("No synthetic_stage3.jsonl found. Run 'synthesize' first for Stage 3 data.")

    output_dir.mkdir(parents=True, exist_ok=True)

    for stage in [
        TrainingStage.STAGE1_IDENTITY,
        TrainingStage.STAGE2_TOOLS,
        TrainingStage.STAGE3_INTEGRATION,
    ]:
        logger.info("=== Mixing %s ===", stage.value)
        mixed = mix_stage(samples, stage, seed=seed)
        paths = write_split(mixed, output_dir, stage, seed=seed)
        logger.info("  Files: %s", {k: str(v) for k, v in paths.items()})

    # Expert balance report
    balance = compute_expert_balance(samples)
    logger.info("Expert balance per stage:")
    for stage_name, experts in sorted(balance.items()):
        logger.info("  %s:", stage_name)
        for expert, count in sorted(experts.items(), key=lambda x: -x[1]):
            logger.info("    %s: %d", expert, count)


@app.command()
def stats(
    input_dir: Path = typer.Option(DEFAULT_OUTPUT, help="Pipeline output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Show data statistics for the pipeline output."""
    _setup_logging(verbose)

    typer.echo("=== Pipeline Data Statistics ===\n")

    for filename in [
        "voice_pairs.jsonl",
        "monologues.jsonl",
        "thinking_traces.jsonl",
        "tool_trajectories.jsonl",
        "tool_calls.jsonl",
        "kevin_voice.jsonl",
        "ecosystem_samples.jsonl",
        "synthetic_stage3.jsonl",
        "all_samples.jsonl",
        "filtered_samples.jsonl",
        "labeled_samples.jsonl",
    ]:
        filepath = input_dir / filename
        if filepath.exists():
            count = sum(1 for _ in open(filepath))
            size_mb = filepath.stat().st_size / (1024 * 1024)
            typer.echo(f"  {filename}: {count:,} records ({size_mb:.1f} MB)")
        else:
            typer.echo(f"  {filename}: not found")

    # Check staged output
    staged_dir = input_dir / "staged"
    if staged_dir.exists():
        typer.echo("\n=== Staged Training Data ===\n")
        for path in sorted(staged_dir.glob("*.jsonl")):
            count = sum(1 for _ in open(path))
            size_mb = path.stat().st_size / (1024 * 1024)
            typer.echo(f"  {path.name}: {count:,} records ({size_mb:.1f} MB)")


@app.command(name="run-all")
def run_all(
    output_dir: Path = typer.Option(DEFAULT_OUTPUT, help="Output directory"),
    seed: int = typer.Option(42, help="Random seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the complete pipeline: extract → transform → filter → label → mix."""
    _setup_logging(verbose)
    logger = logging.getLogger("pipeline")

    logger.info("=" * 60)
    logger.info("RUNNING FULL MoE DATA PIPELINE")
    logger.info("=" * 60)

    logger.info("\n[1/6] EXTRACT")
    extract(output_dir=output_dir, verbose=verbose)

    logger.info("\n[2/6] TRANSFORM")
    transform(input_dir=output_dir, output_dir=output_dir, verbose=verbose)

    logger.info("\n[3/6] FILTER")
    filter(input_dir=output_dir, output_dir=output_dir, verbose=verbose)

    logger.info("\n[4/6] LABEL")
    label(input_dir=output_dir, output_dir=output_dir, verbose=verbose)

    logger.info("\n[5/6] SYNTHESIZE (Stage 3)")
    synthesize(output_dir=output_dir, seed=seed, verbose=verbose)

    logger.info("\n[6/6] MIX")
    mix(input_dir=output_dir, output_dir=output_dir / "staged", seed=seed, verbose=verbose)

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    stats(input_dir=output_dir, verbose=verbose)


if __name__ == "__main__":
    app()
