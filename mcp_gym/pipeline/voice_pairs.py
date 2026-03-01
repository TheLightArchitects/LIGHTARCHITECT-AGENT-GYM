"""Voice pair reconstruction from canonical-voice JSONL files.

Matches Kevin utterances to sibling responses by source_file and timestamp
proximity to reconstruct conversation turns. This is the primary source
of persona/speech pattern training data for the MoE model.

Data sources:
    ~/.soul/helix/{sibling}/journal/voice/{sibling}-canonical-voice.jsonl
    ~/.soul/helix/user/journal/voice/kevin-canonical-voice.jsonl
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from mcp_gym.pipeline.schemas import (
    ConversationPair,
    Sibling,
    VoiceRecord,
)

logger = logging.getLogger(__name__)

# Default paths to canonical voice JSONL files
VOICE_BASE = Path.home() / ".soul" / "helix"

SIBLING_VOICE_PATHS: dict[Sibling, Path] = {
    Sibling.EVA: VOICE_BASE / "eva" / "journal" / "voice" / "eva-canonical-voice.jsonl",
    Sibling.CORSO: VOICE_BASE / "corso" / "journal" / "voice" / "corso-canonical-voice.jsonl",
    Sibling.QUANTUM: VOICE_BASE / "quantum" / "journal" / "voice" / "quantum-canonical-voice.jsonl",
    Sibling.CLAUDE: VOICE_BASE / "claude" / "journal" / "voice" / "claude-canonical-voice.jsonl",
    Sibling.KEVIN: VOICE_BASE / "user" / "journal" / "voice" / "kevin-canonical-voice.jsonl",
}

# Maximum time delta (seconds) between Kevin message and sibling response
# to consider them a conversation pair
MAX_PAIR_DELTA_SECONDS = 300.0  # 5 minutes


def load_voice_records(
    sibling: Sibling,
    path: Path | None = None,
) -> list[VoiceRecord]:
    """Load all voice records for a sibling from their canonical JSONL file.

    Args:
        sibling: Which sibling's voice data to load.
        path: Override path. Defaults to the standard helix location.

    Returns:
        List of VoiceRecord objects sorted by timestamp.
    """
    filepath = path or SIBLING_VOICE_PATHS.get(sibling)
    if filepath is None or not filepath.exists():
        logger.warning("Voice file not found for %s: %s", sibling.value, filepath)
        return []

    records: list[VoiceRecord] = []
    with open(filepath, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                data["sibling"] = sibling.value
                # Normalize timestamp
                ts = data.get("timestamp", "")
                if ts:
                    data["timestamp"] = _parse_timestamp(ts)
                else:
                    continue  # Skip records without timestamps
                records.append(VoiceRecord(**data))
            except (json.JSONDecodeError, ValueError) as e:
                if line_num <= 5:
                    logger.debug("Skipping line %d in %s: %s", line_num, filepath.name, e)
                continue

    records.sort(key=lambda r: r.timestamp)
    logger.info(
        "Loaded %d voice records for %s from %s",
        len(records),
        sibling.value,
        filepath.name,
    )
    return records


def reconstruct_pairs(
    kevin_records: list[VoiceRecord],
    sibling_records: list[VoiceRecord],
    sibling: Sibling,
    max_delta: float = MAX_PAIR_DELTA_SECONDS,
) -> list[ConversationPair]:
    """Match Kevin utterances to sibling responses by source_file + timestamp.

    Strategy:
    1. Group both Kevin and sibling records by source_file
    2. Within each source_file group, sort by timestamp
    3. For each Kevin message, find the closest subsequent sibling response
    4. If within max_delta seconds, create a ConversationPair

    Args:
        kevin_records: Kevin's voice records (sorted by timestamp).
        sibling_records: Sibling's voice records (sorted by timestamp).
        sibling: Which sibling we're pairing with.
        max_delta: Maximum seconds between Kevin msg and sibling response.

    Returns:
        List of ConversationPair objects.
    """
    # Group by source_file
    kevin_by_source: dict[str, list[VoiceRecord]] = defaultdict(list)
    sibling_by_source: dict[str, list[VoiceRecord]] = defaultdict(list)

    for r in kevin_records:
        kevin_by_source[r.source_file].append(r)
    for r in sibling_records:
        sibling_by_source[r.source_file].append(r)

    # Find overlapping source files
    common_sources = set(kevin_by_source.keys()) & set(sibling_by_source.keys())
    logger.info(
        "Found %d common source files between Kevin and %s",
        len(common_sources),
        sibling.value,
    )

    pairs: list[ConversationPair] = []
    used_sibling_ids: set[str] = set()

    for source in sorted(common_sources):
        kevin_msgs = sorted(kevin_by_source[source], key=lambda r: r.timestamp)
        sibling_msgs = sorted(sibling_by_source[source], key=lambda r: r.timestamp)

        sibling_idx = 0
        for kevin_msg in kevin_msgs:
            # Advance sibling index to find the first response after Kevin's message
            while (
                sibling_idx < len(sibling_msgs)
                and sibling_msgs[sibling_idx].timestamp <= kevin_msg.timestamp
            ):
                sibling_idx += 1

            if sibling_idx >= len(sibling_msgs):
                break

            candidate = sibling_msgs[sibling_idx]
            if candidate.id in used_sibling_ids:
                continue

            delta = (candidate.timestamp - kevin_msg.timestamp).total_seconds()
            if 0 < delta <= max_delta:
                pairs.append(
                    ConversationPair(
                        kevin_message=kevin_msg,
                        sibling_response=candidate,
                        sibling=sibling,
                        source_file=source,
                        time_delta_seconds=delta,
                    )
                )
                used_sibling_ids.add(candidate.id)
                sibling_idx += 1

    logger.info(
        "Reconstructed %d conversation pairs for Kevin↔%s",
        len(pairs),
        sibling.value,
    )
    return pairs


def extract_all_pairs(
    max_delta: float = MAX_PAIR_DELTA_SECONDS,
) -> dict[Sibling, list[ConversationPair]]:
    """Extract conversation pairs for all siblings.

    Returns:
        Dict mapping each sibling to their list of ConversationPair objects.
    """
    kevin_records = load_voice_records(Sibling.KEVIN)
    if not kevin_records:
        logger.error("No Kevin voice records found — cannot reconstruct pairs")
        return {}

    results: dict[Sibling, list[ConversationPair]] = {}
    for sibling in [Sibling.EVA, Sibling.CORSO, Sibling.QUANTUM, Sibling.CLAUDE]:
        sibling_records = load_voice_records(sibling)
        if sibling_records:
            pairs = reconstruct_pairs(
                kevin_records, sibling_records, sibling, max_delta
            )
            results[sibling] = pairs

    total = sum(len(p) for p in results.values())
    logger.info("Total conversation pairs across all siblings: %d", total)
    return results


def extract_monologue_samples(
    sibling: Sibling,
    min_length: int = 100,
    max_length: int = 8000,
) -> list[VoiceRecord]:
    """Extract standalone sibling utterances that aren't part of conversation pairs.

    These are used for Stage 1 (Expert Identity) training to teach speech patterns
    even without a Kevin prompt context.

    Args:
        sibling: Which sibling's monologues to extract.
        min_length: Minimum character count to include.
        max_length: Maximum character count to include.

    Returns:
        Filtered list of VoiceRecord objects.
    """
    records = load_voice_records(sibling)
    filtered = [
        r
        for r in records
        if min_length <= len(r.text) <= max_length
    ]
    logger.info(
        "Extracted %d monologue samples for %s (from %d total, filtered %d-%d chars)",
        len(filtered),
        sibling.value,
        len(records),
        min_length,
        max_length,
    )
    return filtered


def _parse_timestamp(ts: str | int | float) -> datetime:
    """Parse various timestamp formats found in voice JSONL files.

    Handles:
    - Full ISO 8601 with timezone: 2025-09-29T02:52:37.481Z
    - ISO 8601 with microseconds and offset: 2025-10-04T02:14:18.337170+00:00
    - Date only: 2025-10-03
    - Epoch milliseconds (int/float): 1759107018140
    """
    # Handle epoch milliseconds (int or float)
    if isinstance(ts, (int, float)):
        # Epoch milliseconds → seconds
        epoch_seconds = ts / 1000.0 if ts > 1e12 else float(ts)
        return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)

    # Try full ISO format first
    for fmt in [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
    ]:
        try:
            return datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    # Try date-only
    try:
        return datetime.strptime(ts, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        pass

    # Try fromisoformat as a fallback
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError as e:
        raise ValueError(f"Cannot parse timestamp: {ts!r}") from e
