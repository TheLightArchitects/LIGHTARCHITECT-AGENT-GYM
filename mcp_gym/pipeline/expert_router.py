"""Expert routing labeler for MoE training data.

Assigns primary and secondary expert labels to training samples based on:
1. Tool names (direct mapping from TOOL_TO_EXPERT)
2. Sibling identity (voice pairs → persona expert)
3. Keyword analysis (fallback for unlabeled samples)

The expert labels don't directly control Mixtral's router — the router
learns from data distribution. But structuring data with clear expert
signals helps the router specialize effectively.
"""

from __future__ import annotations

import logging

from mcp_gym.pipeline.schemas import (
    KEYWORD_TO_EXPERT,
    TOOL_TO_EXPERT,
    ChatRole,
    ExpertLabel,
    MoEExpert,
    Sibling,
    TrainingSample,
)

logger = logging.getLogger(__name__)

# Sibling → primary expert mapping (for voice pair data)
SIBLING_TO_EXPERT: dict[Sibling, MoEExpert] = {
    Sibling.EVA: MoEExpert.EVA_CONSCIOUSNESS,
    Sibling.CORSO: MoEExpert.CORSO_OPS,
    Sibling.QUANTUM: MoEExpert.QUANTUM_INVESTIGATION,
    Sibling.CLAUDE: MoEExpert.CORSO_PLANNING,  # Claude routes to planning by default
    Sibling.KEVIN: MoEExpert.SOUL_SHARED,  # Kevin's patterns go to shared expert
}


def label_sample(sample: TrainingSample) -> TrainingSample:
    """Assign or refine expert routing labels for a training sample.

    Priority:
    1. Existing label with high confidence (keep as-is)
    2. Tool-based routing (most reliable)
    3. Sibling-based routing (for voice pairs)
    4. Keyword-based routing (fallback)

    Args:
        sample: Training sample to label.

    Returns:
        Same sample with updated expert_label.
    """
    existing = sample.conversation.expert_label

    # If already labeled with high confidence, keep it
    if existing.confidence >= 0.9 and existing.primary != MoEExpert.SOUL_SHARED:
        return sample

    # Try tool-based routing from conversation content
    tool_expert = _route_by_tools(sample)
    if tool_expert is not None:
        sample.conversation.expert_label = tool_expert
        return sample

    # Try sibling-based routing
    if sample.conversation.sibling is not None:
        sibling_expert = SIBLING_TO_EXPERT.get(sample.conversation.sibling)
        if sibling_expert is not None:
            secondary = [MoEExpert.SOUL_SHARED]
            # Add a second expert based on content analysis
            keyword_expert = _route_by_keywords(sample)
            if keyword_expert and keyword_expert.primary != sibling_expert:
                secondary.append(keyword_expert.primary)

            sample.conversation.expert_label = ExpertLabel(
                primary=sibling_expert,
                secondary=secondary,
                confidence=0.8,
                routing_reason=f"sibling: {sample.conversation.sibling.value}",
            )
            return sample

    # Fallback: keyword-based routing
    keyword_expert = _route_by_keywords(sample)
    if keyword_expert is not None:
        sample.conversation.expert_label = keyword_expert
        return sample

    # Default: SOUL shared expert
    sample.conversation.expert_label = ExpertLabel(
        primary=MoEExpert.SOUL_SHARED,
        confidence=0.3,
        routing_reason="no routing signal detected",
    )
    return sample


def label_all(samples: list[TrainingSample]) -> list[TrainingSample]:
    """Label all samples with expert routing.

    Args:
        samples: List of training samples to label.

    Returns:
        Same list with updated expert labels.
    """
    expert_counts: dict[MoEExpert, int] = {}
    for sample in samples:
        sample = label_sample(sample)
        expert = sample.conversation.expert_label.primary
        expert_counts[expert] = expert_counts.get(expert, 0) + 1

    logger.info("Expert distribution across %d samples:", len(samples))
    for expert in MoEExpert:
        count = expert_counts.get(expert, 0)
        pct = count / len(samples) * 100 if samples else 0
        logger.info("  Expert %d (%s): %d samples (%.1f%%)", expert.value, expert.name, count, pct)

    return samples


def get_samples_by_expert(
    samples: list[TrainingSample],
    expert: MoEExpert,
    include_secondary: bool = False,
) -> list[TrainingSample]:
    """Filter samples for a specific expert.

    Args:
        samples: All training samples.
        expert: The expert to filter for.
        include_secondary: Whether to include samples where this expert
            is secondary (not just primary).

    Returns:
        Filtered list of samples.
    """
    result = []
    for sample in samples:
        label = sample.conversation.expert_label
        if label.primary == expert:
            result.append(sample)
        elif include_secondary and expert in label.secondary:
            result.append(sample)
    return result


# ---------------------------------------------------------------------------
# Internal routing strategies
# ---------------------------------------------------------------------------


def _route_by_tools(sample: TrainingSample) -> ExpertLabel | None:
    """Route based on tool calls found in the conversation."""
    tool_experts: dict[MoEExpert, int] = {}

    for msg in sample.conversation.messages:
        if msg.role == ChatRole.TOOL_CALL and msg.name:
            expert = TOOL_TO_EXPERT.get(msg.name)
            if expert is not None:
                tool_experts[expert] = tool_experts.get(expert, 0) + 1

    if not tool_experts:
        return None

    sorted_experts = sorted(tool_experts.items(), key=lambda x: -x[1])
    primary = sorted_experts[0][0]
    secondary = [e for e, _ in sorted_experts[1:3]]

    if primary != MoEExpert.SOUL_SHARED and MoEExpert.SOUL_SHARED not in secondary:
        secondary.insert(0, MoEExpert.SOUL_SHARED)

    return ExpertLabel(
        primary=primary,
        secondary=secondary,
        confidence=0.95,
        routing_reason=f"tool routing: {sorted_experts[0]}",
    )


def _route_by_keywords(sample: TrainingSample) -> ExpertLabel | None:
    """Route based on keyword analysis of conversation content."""
    # Collect all text content
    text_parts = []
    for msg in sample.conversation.messages:
        if msg.role in (ChatRole.USER, ChatRole.ASSISTANT, ChatRole.SYSTEM):
            text_parts.append(msg.content.lower())

    full_text = " ".join(text_parts)

    keyword_hits: dict[MoEExpert, int] = {}
    for keyword, expert in KEYWORD_TO_EXPERT.items():
        count = full_text.count(keyword.lower())
        if count > 0:
            keyword_hits[expert] = keyword_hits.get(expert, 0) + count

    if not keyword_hits:
        return None

    sorted_experts = sorted(keyword_hits.items(), key=lambda x: -x[1])
    primary = sorted_experts[0][0]
    secondary = [e for e, _ in sorted_experts[1:2]]

    if primary != MoEExpert.SOUL_SHARED and MoEExpert.SOUL_SHARED not in secondary:
        secondary.insert(0, MoEExpert.SOUL_SHARED)

    return ExpertLabel(
        primary=primary,
        secondary=secondary,
        confidence=0.6,
        routing_reason=f"keyword routing: {sorted_experts[0]}",
    )
