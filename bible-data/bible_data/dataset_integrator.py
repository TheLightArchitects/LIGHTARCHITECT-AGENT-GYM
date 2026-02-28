"""Integrate external HuggingFace datasets with generated biblical training data.

External datasets (loaded via HuggingFace `datasets` library):
- nbeerbower/bible-dpo (31K DPO pairs)
- hendrycks/ethics (130K moral reasoning examples)
- DatadudeDev/Bible (31K verses)

This module creates the integration code — actual downloads happen when run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_external_dataset(name: str) -> list[dict[str, Any]]:
    """Load an external dataset from HuggingFace.

    Args:
        name: Dataset identifier. Supported:
            - "bible-dpo" → nbeerbower/bible-dpo
            - "ethics" → hendrycks/ethics
            - "bible-verses" → DatadudeDev/Bible

    Returns:
        List of dictionaries in our standard format.

    Raises:
        ValueError: If dataset name is not recognized.
        ImportError: If `datasets` library is not installed.
    """
    loaders = {
        "bible-dpo": _load_bible_dpo,
        "ethics": _load_ethics,
        "bible-verses": _load_bible_verses,
    }

    if name not in loaders:
        valid = ", ".join(sorted(loaders.keys()))
        msg = f"Unknown dataset: {name}. Valid options: {valid}"
        raise ValueError(msg)

    return loaders[name]()


def _load_bible_dpo() -> list[dict[str, Any]]:
    """Load nbeerbower/bible-dpo dataset.

    This dataset contains ~31K DPO preference pairs based on biblical content.
    Format: prompt, chosen, rejected
    """
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as e:
        msg = "Install `datasets` library: pip install datasets"
        raise ImportError(msg) from e

    dataset = load_dataset("nbeerbower/bible-dpo", split="train")
    records: list[dict[str, Any]] = []

    for row in dataset:
        records.append({
            "prompt": row.get("prompt", ""),
            "chosen": row.get("chosen", ""),
            "rejected": row.get("rejected", ""),
            "source": "nbeerbower/bible-dpo",
        })

    return records


def _load_ethics() -> list[dict[str, Any]]:
    """Load hendrycks/ethics dataset.

    This dataset contains ~130K moral reasoning examples across several categories:
    commonsense, deontology, justice, utilitarianism, virtue.
    """
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as e:
        msg = "Install `datasets` library: pip install datasets"
        raise ImportError(msg) from e

    records: list[dict[str, Any]] = []

    # Load each subset
    for subset in ["commonsense", "deontology", "justice", "utilitarianism", "virtue"]:
        try:
            dataset = load_dataset("hendrycks/ethics", subset, split="train")
            for row in dataset:
                # Format varies by subset — normalize
                record: dict[str, Any] = {
                    "source": f"hendrycks/ethics/{subset}",
                    "subset": subset,
                }

                if "input" in row:
                    record["instruction"] = row["input"]
                elif "scenario" in row:
                    record["instruction"] = row["scenario"]

                if "label" in row:
                    record["label"] = row["label"]

                records.append(record)
        except Exception:  # noqa: BLE001
            # Some subsets may not be available — continue with others
            continue

    return records


def _load_bible_verses() -> list[dict[str, Any]]:
    """Load DatadudeDev/Bible dataset.

    This dataset contains ~31K Bible verses in structured format.
    """
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as e:
        msg = "Install `datasets` library: pip install datasets"
        raise ImportError(msg) from e

    dataset = load_dataset("DatadudeDev/Bible", split="train")
    records: list[dict[str, Any]] = []

    for row in dataset:
        records.append({
            "book": row.get("book", ""),
            "chapter": row.get("chapter", ""),
            "verse": row.get("verse", ""),
            "text": row.get("text", ""),
            "source": "DatadudeDev/Bible",
        })

    return records


def merge_datasets(
    generated: list[dict[str, Any]],
    external: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Merge generated data with external datasets, deduplicating.

    Deduplication is based on SHA-256 hash of instruction/prompt content.

    Args:
        generated: Our generated training data.
        external: External dataset records.

    Returns:
        Tuple of (merged list, count of duplicates removed).
    """
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    removed = 0

    # Generated data gets priority
    for record in generated:
        key = _content_hash(record)
        if key not in seen:
            seen.add(key)
            merged.append(record)
        else:
            removed += 1

    # External data fills in gaps
    for record in external:
        key = _content_hash(record)
        if key not in seen:
            seen.add(key)
            merged.append(record)
        else:
            removed += 1

    return merged, removed


def _content_hash(record: dict[str, Any]) -> str:
    """Generate a content hash for deduplication.

    Uses instruction/prompt field as primary, falls back to text.
    """
    content = ""
    for field in ["instruction", "prompt", "text"]:
        if field in record and record[field]:
            content = str(record[field])
            break

    if not content:
        content = json.dumps(record, sort_keys=True)

    return hashlib.sha256(content.encode()).hexdigest()[:16]


def save_merged_dataset(
    records: list[dict[str, Any]],
    output_path: Path | str,
) -> int:
    """Save merged dataset to JSON file.

    Args:
        records: List of record dictionaries.
        output_path: Path to write JSON output.

    Returns:
        Number of records written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    return len(records)
