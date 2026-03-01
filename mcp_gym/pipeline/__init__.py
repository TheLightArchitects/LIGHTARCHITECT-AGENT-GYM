"""MoE data pipeline for Mixtral 8x7B fine-tuning.

Extracts voice pairs, thinking traces, and tool trajectories from the SOUL helix,
transforms them into ChatML format with expert routing labels, and produces
staged training data for SFT + DPO.

Pipeline stages:
  0. schemas    — Pydantic models for all data formats
  1. extract    — Read JSONL sources (voice, thinking, tools)
  2. transform  — Reconstruct conversation pairs, link traces
  3. filter     — Quality gates (length, dedup, PII, noise)
  4. label      — Expert routing assignment (8 Mixtral experts)
  5. mix        — Stage-specific data blending (3 SFT + DPO)
"""

from mcp_gym.pipeline.schemas import (
    ChatMLConversation,
    ChatMLMessage,
    ConversationPair,
    ExpertLabel,
    MoEExpert,
    ThinkingTrace,
    ToolCallRecord,
    ToolTrajectoryRecord,
    TrainingSample,
    VoiceRecord,
)

__all__ = [
    "ChatMLConversation",
    "ChatMLMessage",
    "ConversationPair",
    "ExpertLabel",
    "MoEExpert",
    "ThinkingTrace",
    "ToolCallRecord",
    "ToolTrajectoryRecord",
    "TrainingSample",
    "VoiceRecord",
]
