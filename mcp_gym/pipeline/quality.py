"""Quality filtering for training data.

Applies length, deduplication, PII detection, and noise removal filters
to ensure clean training data for the MoE model.
"""

from __future__ import annotations

import hashlib
import logging
import re

from mcp_gym.pipeline.schemas import ChatMLMessage, ChatRole, TrainingSample

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

MIN_CONVERSATION_CHARS = 50  # Minimum total chars across all messages
MAX_CONVERSATION_CHARS = 32000  # Maximum total chars (context window limit)
MIN_ASSISTANT_CHARS = 20  # Minimum chars in any assistant message
MAX_ASSISTANT_CHARS = 16000  # Maximum chars in single assistant message
MIN_USER_CHARS = 5  # Minimum chars in user message

# PII patterns to detect (not remove — flag for review)
PII_PATTERNS = [
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # Email
    re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),  # US phone
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
    re.compile(r"sk-ant-api\d{2}-[a-zA-Z0-9_-]{95}"),  # Anthropic API key
    re.compile(r"api[_-]?key[\"\s:=]+[\"']?([a-zA-Z0-9_-]{32,})"),  # Generic API key
    re.compile(r"-----BEGIN (RSA |EC )?PRIVATE KEY-----"),  # Private key
    re.compile(r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),  # JWT
]

# Noise patterns — content that's pure noise, not useful for training
NOISE_PATTERNS = [
    re.compile(r"^\s*$"),  # Empty/whitespace only
    re.compile(r"^(ok|okay|yes|no|sure|thanks|thank you|got it)\s*[.!?]*\s*$", re.I),
    re.compile(r"^```\s*```$"),  # Empty code blocks
]


def filter_samples(
    samples: list[TrainingSample],
    dedup: bool = True,
    check_pii: bool = True,
    remove_noise: bool = True,
) -> list[TrainingSample]:
    """Apply all quality filters to a list of training samples.

    Args:
        samples: Input training samples.
        dedup: Whether to deduplicate by content hash.
        check_pii: Whether to flag samples containing PII.
        remove_noise: Whether to remove noisy/low-quality samples.

    Returns:
        Filtered list of TrainingSample objects.
    """
    initial_count = len(samples)
    filtered = samples

    # Length filter
    filtered = [s for s in filtered if _check_length(s)]
    after_length = len(filtered)

    # Noise filter
    if remove_noise:
        filtered = [s for s in filtered if not _is_noise(s)]
    after_noise = len(filtered)

    # PII check (flag, don't remove — log warnings)
    if check_pii:
        pii_count = 0
        clean = []
        for s in filtered:
            if _has_pii(s):
                pii_count += 1
                # Scrub PII instead of removing
                s = _scrub_pii(s)
            clean.append(s)
        filtered = clean
        if pii_count:
            logger.warning("Found PII in %d samples — scrubbed", pii_count)

    # Deduplication
    if dedup:
        filtered = _deduplicate(filtered)
    after_dedup = len(filtered)

    logger.info(
        "Quality filter: %d → %d samples "
        "(length: -%d, noise: -%d, dedup: -%d)",
        initial_count,
        after_dedup,
        initial_count - after_length,
        after_length - after_noise,
        after_noise - after_dedup,
    )
    return filtered


def _check_length(sample: TrainingSample) -> bool:
    """Check if sample meets length requirements."""
    messages = sample.conversation.messages
    total_chars = sum(len(m.content) for m in messages)

    if total_chars < MIN_CONVERSATION_CHARS or total_chars > MAX_CONVERSATION_CHARS:
        return False

    for msg in messages:
        if msg.role == ChatRole.ASSISTANT and len(msg.content) < MIN_ASSISTANT_CHARS:
            return False
        if msg.role == ChatRole.ASSISTANT and len(msg.content) > MAX_ASSISTANT_CHARS:
            return False
        if msg.role == ChatRole.USER and len(msg.content) < MIN_USER_CHARS:
            return False

    return True


def _is_noise(sample: TrainingSample) -> bool:
    """Check if sample is pure noise."""
    for msg in sample.conversation.messages:
        if msg.role in (ChatRole.USER, ChatRole.ASSISTANT):
            for pattern in NOISE_PATTERNS:
                if pattern.match(msg.content):
                    return True
    return False


def _has_pii(sample: TrainingSample) -> bool:
    """Check if any message contains PII patterns."""
    for msg in sample.conversation.messages:
        for pattern in PII_PATTERNS:
            if pattern.search(msg.content):
                return True
    return False


def _scrub_pii(sample: TrainingSample) -> TrainingSample:
    """Replace PII patterns with redaction markers."""
    new_messages: list[ChatMLMessage] = []
    for msg in sample.conversation.messages:
        content = msg.content
        for pattern in PII_PATTERNS:
            content = pattern.sub("[REDACTED]", content)
        new_messages.append(
            ChatMLMessage(role=msg.role, content=content, name=msg.name)
        )

    sample.conversation.messages = new_messages
    return sample


def _deduplicate(samples: list[TrainingSample]) -> list[TrainingSample]:
    """Remove duplicate samples by content hash."""
    seen: set[str] = set()
    unique: list[TrainingSample] = []

    for sample in samples:
        # Hash based on user + assistant content (ignore system prompt variations)
        content_parts = []
        for msg in sample.conversation.messages:
            if msg.role in (ChatRole.USER, ChatRole.ASSISTANT):
                content_parts.append(msg.content)

        content_hash = hashlib.sha256(
            "|".join(content_parts).encode()
        ).hexdigest()[:16]

        if content_hash not in seen:
            seen.add(content_hash)
            unique.append(sample)

    removed = len(samples) - len(unique)
    if removed:
        logger.info("Deduplicated: removed %d duplicates", removed)
    return unique
