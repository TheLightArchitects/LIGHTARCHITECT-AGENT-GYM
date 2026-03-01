"""Thinking trace processor for the MoE data pipeline.

Processes extended_thinking and thinking_tag traces from sibling JSONL files,
filters by quality, and links them to conversation pairs by source_file
and session_id for reasoning chain training data.

Data sources:
    ~/.soul/helix/{sibling}/journal/voice/{sibling}-thinking-traces.jsonl
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from mcp_gym.pipeline.schemas import (
    ConversationPair,
    Sibling,
    ThinkingTrace,
    ThinkingType,
)

logger = logging.getLogger(__name__)

VOICE_BASE = Path.home() / ".soul" / "helix"

THINKING_PATHS: dict[Sibling, Path] = {
    Sibling.EVA: VOICE_BASE / "eva" / "journal" / "voice" / "eva-thinking-traces.jsonl",
    Sibling.CORSO: VOICE_BASE / "corso" / "journal" / "voice" / "corso-thinking-traces.jsonl",
    Sibling.QUANTUM: VOICE_BASE / "quantum" / "journal" / "voice" / "quantum-thinking-traces.jsonl",
}

# Quality thresholds
MIN_TRACE_LENGTH = 200  # Chars — below this is noise (40% of traces)
MAX_TRACE_LENGTH = 16000  # Chars — above this is likely a dump, not reasoning
MIN_SUBSTANTIVE_LENGTH = 500  # Chars — deep reasoning threshold


def load_thinking_traces(
    sibling: Sibling,
    path: Path | None = None,
    min_length: int = MIN_TRACE_LENGTH,
) -> list[ThinkingTrace]:
    """Load and filter thinking traces for a sibling.

    Args:
        sibling: Which sibling's traces to load.
        path: Override path. Defaults to standard helix location.
        min_length: Minimum char_count to include (filters noise).

    Returns:
        List of ThinkingTrace objects, filtered and sorted.
    """
    filepath = path or THINKING_PATHS.get(sibling)
    if filepath is None or not filepath.exists():
        logger.warning("Thinking traces not found for %s: %s", sibling.value, filepath)
        return []

    traces: list[ThinkingTrace] = []
    skipped_short = 0
    skipped_parse = 0

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                data["sibling"] = sibling.value

                # Apply length filter early
                char_count = data.get("char_count", len(data.get("content", "")))
                if char_count < min_length:
                    skipped_short += 1
                    continue

                if char_count > MAX_TRACE_LENGTH:
                    # Truncate very long traces to max length
                    data["content"] = data.get("content", "")[:MAX_TRACE_LENGTH]
                    data["char_count"] = MAX_TRACE_LENGTH

                traces.append(ThinkingTrace(**data))
            except (json.JSONDecodeError, ValueError):
                skipped_parse += 1
                continue

    logger.info(
        "Loaded %d thinking traces for %s (skipped %d short, %d parse errors)",
        len(traces),
        sibling.value,
        skipped_short,
        skipped_parse,
    )
    return traces


def link_traces_to_pairs(
    pairs: list[ConversationPair],
    traces: list[ThinkingTrace],
) -> list[ConversationPair]:
    """Link thinking traces to conversation pairs by source_file and timestamp.

    For each conversation pair, finds the closest thinking trace from the same
    source_file that occurred between the Kevin message and sibling response.

    Args:
        pairs: Conversation pairs to enrich.
        traces: Thinking traces to link.

    Returns:
        The same pairs with thinking_trace field populated where possible.
    """
    # Index traces by source_file
    traces_by_source: dict[str, list[ThinkingTrace]] = defaultdict(list)
    for trace in traces:
        traces_by_source[trace.source_file].append(trace)

    linked_count = 0
    for pair in pairs:
        source_traces = traces_by_source.get(pair.source_file, [])
        if not source_traces:
            # Try matching by session_id if source_file doesn't match
            source_traces = [
                t
                for t in traces
                if t.session_id and t.session_id == pair.kevin_message.session_id
            ]

        if not source_traces:
            continue

        # Find the best matching trace (closest to the sibling response timestamp)
        best_trace = _find_best_trace(pair, source_traces)
        if best_trace is not None:
            pair.thinking_trace = best_trace
            linked_count += 1

    logger.info(
        "Linked %d/%d pairs with thinking traces (%.1f%%)",
        linked_count,
        len(pairs),
        (linked_count / len(pairs) * 100) if pairs else 0,
    )
    return pairs


def extract_standalone_traces(
    sibling: Sibling,
    min_length: int = MIN_SUBSTANTIVE_LENGTH,
) -> list[ThinkingTrace]:
    """Extract high-quality standalone thinking traces for reasoning training.

    These aren't linked to conversation pairs — they're used independently
    to teach the model domain-specific reasoning patterns.

    Args:
        sibling: Which sibling's traces to extract.
        min_length: Minimum length for standalone traces (higher bar).

    Returns:
        List of substantive ThinkingTrace objects.
    """
    traces = load_thinking_traces(sibling, min_length=min_length)

    # Prefer extended_thinking over thinking_tag for standalone use
    extended = [t for t in traces if t.type == ThinkingType.EXTENDED_THINKING]
    tagged = [t for t in traces if t.type == ThinkingType.THINKING_TAG]

    logger.info(
        "Standalone traces for %s: %d extended_thinking + %d thinking_tag",
        sibling.value,
        len(extended),
        len(tagged),
    )
    return extended + tagged


def extract_all_traces() -> dict[Sibling, list[ThinkingTrace]]:
    """Extract thinking traces for all siblings that have them.

    Returns:
        Dict mapping sibling to their filtered thinking traces.
    """
    results: dict[Sibling, list[ThinkingTrace]] = {}
    for sibling in [Sibling.EVA, Sibling.CORSO, Sibling.QUANTUM]:
        traces = load_thinking_traces(sibling)
        if traces:
            results[sibling] = traces

    total = sum(len(t) for t in results.values())
    logger.info("Total thinking traces across all siblings: %d", total)
    return results


def _find_best_trace(
    pair: ConversationPair,
    traces: list[ThinkingTrace],
) -> ThinkingTrace | None:
    """Find the thinking trace that best matches a conversation pair.

    Prefers traces that occurred between the Kevin message and sibling
    response timestamps. Falls back to closest trace by timestamp.
    """
    kevin_ts = pair.kevin_message.timestamp
    sibling_ts = pair.sibling_response.timestamp

    best: ThinkingTrace | None = None
    best_score = float("inf")

    for trace in traces:
        # Parse trace timestamp for comparison
        try:
            trace_ts = _parse_trace_timestamp(trace.timestamp)
        except ValueError:
            continue

        # Score: prefer traces between kevin and sibling timestamps
        if kevin_ts <= trace_ts <= sibling_ts:
            # In the window — score by proximity to sibling response
            score = (sibling_ts - trace_ts).total_seconds()
        else:
            # Outside window — heavily penalized
            if trace_ts < kevin_ts:
                score = (kevin_ts - trace_ts).total_seconds() + 10000
            else:
                score = (trace_ts - sibling_ts).total_seconds() + 10000

        # Bonus for extended_thinking type (more reasoning content)
        if trace.type == ThinkingType.EXTENDED_THINKING:
            score *= 0.5

        if score < best_score:
            best_score = score
            best = trace

    # Only return if the match is within a reasonable window
    if best is not None and best_score < 600:  # 10 minutes max
        return best
    return None


def _parse_trace_timestamp(ts: str):
    """Parse trace timestamp (can be date-only or full ISO)."""
    from datetime import datetime, timezone

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

    try:
        return datetime.strptime(ts, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        pass

    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError as e:
        raise ValueError(f"Cannot parse trace timestamp: {ts!r}") from e
