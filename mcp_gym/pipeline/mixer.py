"""Data mixer for 3-stage SFT + DPO training.

Produces stage-specific JSONL files with the correct proportions:

Stage 1 — Expert Identity (52K target):
  40% voice pairs (persona speech patterns)
  25% monologues (extended persona examples)
  20% thinking traces (reasoning patterns)
  15% biblical foundation (from bible-data/)

Stage 2 — Tool Mastery (29K target):
  45% tool trajectories (multi-step tool use)
  25% MCP tool calls (individual tool invocations)
  15% tool schemas (API documentation)
  15% existing domain data (from training-data/)

Stage 3 — Integration (8K target):
  40% multi-expert scenarios (cross-domain routing)
  30% complex trajectories (multi-tool, multi-domain)
  20% SCRUM/review traces (squad collaboration)
  10% Kevin voice (architect's reasoning patterns)

DPO (1.7K pairs):
  40% biblical (biblical vs. worldly reasoning)
  30% tool quality (good vs. poor tool selection)
  30% security (safe vs. unsafe decisions)
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from mcp_gym.pipeline.schemas import (
    MoEExpert,
    TrainingSample,
    TrainingStage,
)

logger = logging.getLogger(__name__)

# Target sample counts per stage
STAGE_TARGETS = {
    TrainingStage.STAGE1_IDENTITY: 52000,
    TrainingStage.STAGE2_TOOLS: 29000,
    TrainingStage.STAGE3_INTEGRATION: 8000,
}

# Stage-specific source proportions
STAGE1_PROPORTIONS = {
    "voice_pair": 0.40,
    "monologue": 0.25,
    "thinking_trace": 0.20,
    "biblical": 0.15,
}

STAGE2_PROPORTIONS = {
    "tool_trajectory": 0.45,
    "tool_call": 0.25,
    "tool_schema": 0.15,
    "domain_existing": 0.15,
}

STAGE3_PROPORTIONS = {
    "multi_expert": 0.40,
    "complex_trajectory": 0.30,
    "scrum_trace": 0.20,
    "kevin_voice": 0.10,
}


def mix_stage(
    samples: list[TrainingSample],
    stage: TrainingStage,
    target_count: int | None = None,
    seed: int = 42,
) -> list[TrainingSample]:
    """Mix samples for a specific training stage.

    Groups samples by source_type, then draws proportionally from each
    group to hit the target count. Uses oversampling for small groups
    and undersampling for large groups.

    Args:
        samples: All available samples (will be filtered by stage).
        stage: Which training stage to mix for.
        target_count: Override target count. Defaults to STAGE_TARGETS.
        seed: Random seed for reproducibility.

    Returns:
        Mixed and shuffled list of TrainingSample objects.
    """
    rng = random.Random(seed)
    target = target_count or STAGE_TARGETS.get(stage, len(samples))

    # Get proportions for this stage
    proportions = _get_proportions(stage)

    # Filter samples to only those for this stage
    stage_samples = [s for s in samples if s.conversation.stage == stage]

    # Group by source_type
    by_source: dict[str, list[TrainingSample]] = {}
    for s in stage_samples:
        src = s.conversation.source_type
        if src not in by_source:
            by_source[src] = []
        by_source[src].append(s)

    logger.info(
        "Mixing stage %s: %d samples available across %d source types",
        stage.value,
        len(stage_samples),
        len(by_source),
    )
    for src, group in sorted(by_source.items()):
        logger.info("  %s: %d samples", src, len(group))

    # Draw proportionally
    mixed: list[TrainingSample] = []
    remaining_target = target

    for source_type, proportion in sorted(proportions.items(), key=lambda x: -x[1]):
        source_target = int(target * proportion)
        available = by_source.get(source_type, [])

        if not available:
            logger.warning(
                "  No samples for source '%s' (expected %.0f%% = %d)",
                source_type,
                proportion * 100,
                source_target,
            )
            continue

        if len(available) >= source_target:
            # Undersample: random selection
            selected = rng.sample(available, source_target)
        else:
            # Oversample: include all + random duplicates
            selected = list(available)
            while len(selected) < source_target:
                selected.append(rng.choice(available))

        mixed.extend(selected)
        remaining_target -= len(selected)
        logger.info(
            "  %s: drew %d/%d (%.1f%% of stage)",
            source_type,
            len(selected),
            len(available),
            len(selected) / target * 100,
        )

    # Fill remaining with any available samples (if we have leftover quota)
    if remaining_target > 0 and stage_samples:
        extras = rng.choices(stage_samples, k=min(remaining_target, len(stage_samples)))
        mixed.extend(extras)
        logger.info("  Filled %d remaining slots from all sources", len(extras))

    rng.shuffle(mixed)
    logger.info(
        "Stage %s mixed: %d samples (target was %d)",
        stage.value,
        len(mixed),
        target,
    )
    return mixed


def write_stage_output(
    samples: list[TrainingSample],
    output_dir: Path,
    stage: TrainingStage,
    format: str = "sharegpt",
) -> Path:
    """Write mixed samples to a JSONL file for training.

    Args:
        samples: Mixed training samples for this stage.
        output_dir: Directory to write output files.
        stage: Training stage (for filename).
        format: Output format ('sharegpt' or 'alpaca').

    Returns:
        Path to the output file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stage.value}.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            if format == "sharegpt":
                record = sample.to_sharegpt()
            else:
                record = sample.to_alpaca()
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        "Wrote %d samples to %s (format: %s)",
        len(samples),
        output_path,
        format,
    )
    return output_path


def write_split(
    samples: list[TrainingSample],
    output_dir: Path,
    stage: TrainingStage,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    seed: int = 42,
    format: str = "sharegpt",
) -> dict[str, Path]:
    """Write train/val/test splits for a stage.

    Args:
        samples: Mixed training samples.
        output_dir: Directory to write output files.
        stage: Training stage (for filename prefix).
        train_ratio: Proportion for training set.
        val_ratio: Proportion for validation set.
        seed: Random seed.
        format: Output format.

    Returns:
        Dict of split name → file path.
    """
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for split_name, split_samples in splits.items():
        path = output_dir / f"{stage.value}_{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for sample in split_samples:
                if format == "sharegpt":
                    record = sample.to_sharegpt()
                else:
                    record = sample.to_alpaca()
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        paths[split_name] = path
        logger.info("  %s: %d samples → %s", split_name, len(split_samples), path)

    return paths


def compute_expert_balance(samples: list[TrainingSample]) -> dict[str, dict[str, int]]:
    """Compute expert distribution per stage for diagnostics.

    Returns:
        Nested dict: stage → expert_name → count.
    """
    distribution: dict[str, dict[str, int]] = {}

    for sample in samples:
        stage = sample.conversation.stage.value
        expert = sample.conversation.expert_label.primary.name

        if stage not in distribution:
            distribution[stage] = {}
        distribution[stage][expert] = distribution[stage].get(expert, 0) + 1

    return distribution


def _get_proportions(stage: TrainingStage) -> dict[str, float]:
    """Get source type proportions for a training stage."""
    if stage == TrainingStage.STAGE1_IDENTITY:
        return STAGE1_PROPORTIONS
    elif stage == TrainingStage.STAGE2_TOOLS:
        return STAGE2_PROPORTIONS
    elif stage == TrainingStage.STAGE3_INTEGRATION:
        return STAGE3_PROPORTIONS
    else:
        return {}
