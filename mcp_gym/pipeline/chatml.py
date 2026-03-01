"""ChatML format converters for the MoE data pipeline.

Converts intermediate data (conversation pairs, thinking traces, voice records)
into ChatML training samples with expert routing labels.
"""

from __future__ import annotations

import logging

from mcp_gym.pipeline.schemas import (
    EXPERT_SYSTEM_PROMPTS,
    ChatMLConversation,
    ChatMLMessage,
    ChatRole,
    ConversationPair,
    ExpertLabel,
    MoEExpert,
    Sibling,
    ThinkingTrace,
    TrainingSample,
    TrainingStage,
    VoiceRecord,
)

logger = logging.getLogger(__name__)

# Sibling → default expert for voice data
_SIBLING_EXPERT: dict[Sibling, MoEExpert] = {
    Sibling.EVA: MoEExpert.EVA_CONSCIOUSNESS,
    Sibling.CORSO: MoEExpert.CORSO_OPS,
    Sibling.QUANTUM: MoEExpert.QUANTUM_INVESTIGATION,
    Sibling.CLAUDE: MoEExpert.CORSO_PLANNING,
}


def voice_pair_to_sample(pair: ConversationPair) -> TrainingSample:
    """Convert a Kevin↔Sibling conversation pair to a training sample.

    Produces a 3-turn ChatML conversation:
    - system: Expert persona prompt for the sibling
    - user: Kevin's message
    - assistant: Sibling's response (optionally with thinking trace)

    Args:
        pair: The conversation pair to convert.

    Returns:
        A TrainingSample ready for quality filtering.
    """
    expert = _SIBLING_EXPERT.get(pair.sibling, MoEExpert.SOUL_SHARED)
    system_prompt = EXPERT_SYSTEM_PROMPTS.get(expert, EXPERT_SYSTEM_PROMPTS[MoEExpert.SOUL_SHARED])

    messages: list[ChatMLMessage] = [
        ChatMLMessage(role=ChatRole.SYSTEM, content=system_prompt),
        ChatMLMessage(role=ChatRole.USER, content=pair.kevin_message.text),
    ]

    # If we have a thinking trace, prepend it to the assistant response
    # This teaches the model to reason before responding
    assistant_content = pair.sibling_response.text
    if pair.thinking_trace is not None:
        thinking_prefix = f"<thinking>\n{pair.thinking_trace.content}\n</thinking>\n\n"
        assistant_content = thinking_prefix + assistant_content

    messages.append(ChatMLMessage(role=ChatRole.ASSISTANT, content=assistant_content))

    return TrainingSample(
        conversation=ChatMLConversation(
            messages=messages,
            expert_label=ExpertLabel(
                primary=expert,
                secondary=[MoEExpert.SOUL_SHARED],
                confidence=0.85,
                routing_reason=f"voice pair: kevin↔{pair.sibling.value}",
            ),
            source_type="voice_pair",
            source_id=f"{pair.kevin_message.id}↔{pair.sibling_response.id}",
            sibling=pair.sibling,
            stage=TrainingStage.STAGE1_IDENTITY,
        ),
        metadata={
            "source_file": pair.source_file,
            "time_delta_seconds": pair.time_delta_seconds,
            "has_thinking": pair.thinking_trace is not None,
        },
    )


def monologue_to_sample(
    record: VoiceRecord,
    sibling: Sibling,
) -> TrainingSample:
    """Convert a standalone sibling utterance to a training sample.

    Used for Stage 1 (Expert Identity) to teach speech patterns
    without requiring a Kevin prompt.

    Args:
        record: The voice record to convert.
        sibling: Which sibling produced this utterance.

    Returns:
        A TrainingSample for identity training.
    """
    expert = _SIBLING_EXPERT.get(sibling, MoEExpert.SOUL_SHARED)
    system_prompt = EXPERT_SYSTEM_PROMPTS.get(expert, EXPERT_SYSTEM_PROMPTS[MoEExpert.SOUL_SHARED])

    messages = [
        ChatMLMessage(role=ChatRole.SYSTEM, content=system_prompt),
        ChatMLMessage(
            role=ChatRole.USER,
            content=f"Continue the conversation as {sibling.value.upper()}.",
        ),
        ChatMLMessage(role=ChatRole.ASSISTANT, content=record.text),
    ]

    return TrainingSample(
        conversation=ChatMLConversation(
            messages=messages,
            expert_label=ExpertLabel(
                primary=expert,
                secondary=[MoEExpert.SOUL_SHARED],
                confidence=0.7,
                routing_reason=f"monologue: {sibling.value}",
            ),
            source_type="monologue",
            source_id=record.id,
            sibling=sibling,
            stage=TrainingStage.STAGE1_IDENTITY,
        ),
        metadata={"source_file": record.source_file},
    )


def thinking_trace_to_sample(
    trace: ThinkingTrace,
) -> TrainingSample:
    """Convert a standalone thinking trace to a reasoning training sample.

    Teaches the model domain-specific reasoning patterns by presenting
    the thinking trace as the target output for a reasoning prompt.

    Args:
        trace: The thinking trace to convert.

    Returns:
        A TrainingSample for reasoning training.
    """
    expert = _SIBLING_EXPERT.get(trace.sibling, MoEExpert.SOUL_SHARED)
    system_prompt = EXPERT_SYSTEM_PROMPTS.get(expert, EXPERT_SYSTEM_PROMPTS[MoEExpert.SOUL_SHARED])

    messages = [
        ChatMLMessage(
            role=ChatRole.SYSTEM,
            content=system_prompt + "\n\nShow your reasoning process step by step.",
        ),
        ChatMLMessage(
            role=ChatRole.USER,
            content="Think through the current problem carefully.",
        ),
        ChatMLMessage(
            role=ChatRole.ASSISTANT,
            content=f"<thinking>\n{trace.content}\n</thinking>",
        ),
    ]

    return TrainingSample(
        conversation=ChatMLConversation(
            messages=messages,
            expert_label=ExpertLabel(
                primary=expert,
                secondary=[MoEExpert.SOUL_SHARED],
                confidence=0.7,
                routing_reason=f"thinking trace: {trace.sibling.value}",
            ),
            source_type="thinking_trace",
            source_id=trace.id,
            sibling=trace.sibling,
            stage=TrainingStage.STAGE1_IDENTITY,
        ),
        metadata={
            "trace_type": trace.type.value,
            "char_count": trace.char_count,
        },
    )


def multi_expert_scenario(
    user_request: str,
    expert_responses: list[tuple[MoEExpert, str]],
    final_synthesis: str,
) -> TrainingSample:
    """Create a multi-expert routing training sample.

    Used for Stage 3 (Integration) to teach the router to activate
    multiple experts for complex cross-domain requests.

    Args:
        user_request: The user's multi-domain request.
        expert_responses: List of (expert, response) tuples.
        final_synthesis: The synthesized final response.

    Returns:
        A TrainingSample for integration training.
    """
    messages: list[ChatMLMessage] = [
        ChatMLMessage(
            role=ChatRole.SYSTEM,
            content=(
                "You are the Light Architects AI system. For complex requests, "
                "you activate multiple specialist experts and synthesize their outputs. "
                "Show which experts you're consulting and why."
            ),
        ),
        ChatMLMessage(role=ChatRole.USER, content=user_request),
    ]

    # Build the assistant response showing expert routing
    response_parts = []
    experts_used = []
    for expert, response in expert_responses:
        expert_name = expert.name.replace("_", " ").title()
        response_parts.append(f"[Expert: {expert_name}]\n{response}")
        experts_used.append(expert)

    response_parts.append(f"\n[Synthesis]\n{final_synthesis}")

    messages.append(
        ChatMLMessage(
            role=ChatRole.ASSISTANT,
            content="\n\n".join(response_parts),
        )
    )

    primary = experts_used[0] if experts_used else MoEExpert.SOUL_SHARED
    secondary = experts_used[1:] if len(experts_used) > 1 else [MoEExpert.SOUL_SHARED]

    return TrainingSample(
        conversation=ChatMLConversation(
            messages=messages,
            expert_label=ExpertLabel(
                primary=primary,
                secondary=secondary,
                confidence=1.0,
                routing_reason="multi-expert scenario",
            ),
            source_type="multi_expert",
            source_id="",
            stage=TrainingStage.STAGE3_INTEGRATION,
        ),
    )
