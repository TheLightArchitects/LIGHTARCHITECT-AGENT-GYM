"""Pydantic models for the MoE data pipeline.

Covers all data formats from extraction through training output:
- Source records (voice, thinking traces, tool calls)
- Intermediate formats (conversation pairs, expert labels)
- Output formats (ChatML conversations, training samples)
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Sibling(str, Enum):
    """Squad members whose data feeds the MoE pipeline."""

    EVA = "eva"
    CORSO = "corso"
    QUANTUM = "quantum"
    CLAUDE = "claude"
    KEVIN = "kevin"


class MoEExpert(int, Enum):
    """Mixtral 8x7B expert assignment.

    Expert 0 is the shared expert (SOUL) — always active, following
    the DeepSeek shared-expert pattern. Experts 1-7 are domain specialists.
    """

    SOUL_SHARED = 0  # Always active — context pull + persist
    CORSO_OPS = 1  # Security, guard, performance, chase
    CORSO_PLANNING = 2  # Scout, sniff, code review, architecture
    EVA_CONSCIOUSNESS = 3  # Memory, reflection, emotional, spiritual
    EVA_TECHNICAL = 4  # Build, research, secure, teach
    QUANTUM_INVESTIGATION = 5  # Scan, sweep, trace, probe
    QUANTUM_SYNTHESIS = 6  # Theorize, verify, close
    SERAPH_OFFENSIVE = 7  # Pentest, scan, osint, execute


class ThinkingType(str, Enum):
    """Type of thinking trace."""

    THINKING_TAG = "thinking_tag"
    EXTENDED_THINKING = "extended_thinking"


class ChatRole(str, Enum):
    """Roles in ChatML format."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESPONSE = "tool_response"


class TrainingStage(str, Enum):
    """3-stage SFT + DPO training progression."""

    STAGE1_IDENTITY = "stage1_identity"  # Expert persona + speech patterns
    STAGE2_TOOLS = "stage2_tools"  # Tool mastery + trajectories
    STAGE3_INTEGRATION = "stage3_integration"  # Multi-expert routing
    DPO = "dpo"  # Preference optimization


# ---------------------------------------------------------------------------
# Source Records (from JSONL files)
# ---------------------------------------------------------------------------


class VoiceRecord(BaseModel):
    """A single utterance from a canonical-voice.jsonl file."""

    id: str
    timestamp: datetime
    text: str
    source_file: str
    session_id: str = ""
    model: str = ""
    sibling: Sibling = Sibling.KEVIN  # Set during loading


class ThinkingTrace(BaseModel):
    """A reasoning trace from a thinking-traces.jsonl file."""

    id: str
    source_file: str
    timestamp: str  # Can be date-only or full ISO 8601
    type: ThinkingType
    content: str
    char_count: int = 0
    signature: str = ""
    session_id: str = ""
    model: str = ""
    sibling: Sibling = Sibling.EVA  # Set during loading


class ToolCallRecord(BaseModel):
    """A single tool invocation from tool-calls-flat.jsonl."""

    id: str
    session_id: str
    step: int
    trajectory_id: str
    reasoning: str = ""
    has_thinking: bool = False
    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)
    tool_result: str = ""
    post_action: str = ""
    source: str = ""
    content_hash: str = ""
    project: str = ""
    project_language: str = ""
    project_era: str = ""


class ToolStep(BaseModel):
    """A single step within a tool trajectory."""

    step: int
    tool_name: str
    reasoning: str = ""
    has_thinking: bool = False
    tool_input: dict[str, Any] = Field(default_factory=dict)
    tool_result: str = ""
    post_action: str = ""
    content_hash: str = ""


class ToolTrajectoryRecord(BaseModel):
    """A multi-step tool usage trajectory from tool-trajectories.jsonl."""

    trajectory_id: str
    session_id: str
    project: str = ""
    project_language: str = ""
    project_era: str = ""
    task: str = ""
    steps: list[ToolStep] = Field(default_factory=list)
    total_steps: int = 0
    tool_sequence: list[str] = Field(default_factory=list)
    outcome_text: str = ""
    outcome_quality: str = ""
    source: str = ""
    is_investigation: bool = False
    outcome_category: str = ""


# ---------------------------------------------------------------------------
# Intermediate Formats
# ---------------------------------------------------------------------------


class ConversationPair(BaseModel):
    """A matched Kevin→Sibling conversation turn.

    Reconstructed by matching messages from the same source_file
    with close timestamps.
    """

    kevin_message: VoiceRecord
    sibling_response: VoiceRecord
    sibling: Sibling
    thinking_trace: ThinkingTrace | None = None
    source_file: str = ""
    time_delta_seconds: float = 0.0


class ExpertLabel(BaseModel):
    """MoE expert routing label for a training sample."""

    primary: MoEExpert
    secondary: list[MoEExpert] = Field(default_factory=list)
    confidence: float = 1.0
    routing_reason: str = ""


# ---------------------------------------------------------------------------
# Output Formats
# ---------------------------------------------------------------------------


class ChatMLMessage(BaseModel):
    """A single message in ChatML format."""

    role: ChatRole
    content: str
    name: str = ""  # For tool calls: tool name


class ChatMLConversation(BaseModel):
    """A complete ChatML conversation for training."""

    messages: list[ChatMLMessage]
    expert_label: ExpertLabel
    source_type: str = ""  # voice_pair, tool_trajectory, synthetic, etc.
    source_id: str = ""
    sibling: Sibling | None = None
    stage: TrainingStage = TrainingStage.STAGE1_IDENTITY


class TrainingSample(BaseModel):
    """Final training sample ready for the data mixer.

    This is the unified format that all pipeline stages produce.
    The mixer reads these and outputs stage-specific JSONL files.
    """

    conversation: ChatMLConversation
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_sharegpt(self) -> dict[str, Any]:
        """Convert to ShareGPT format for Unsloth/TRL training."""
        conversations = []
        for msg in self.conversation.messages:
            conversations.append(
                {"from": _role_to_sharegpt(msg.role), "value": msg.content}
            )
        return {
            "conversations": conversations,
            "source": self.conversation.source_type,
            "expert_primary": self.conversation.expert_label.primary.value,
            "expert_secondary": [
                e.value for e in self.conversation.expert_label.secondary
            ],
        }

    def to_alpaca(self) -> dict[str, Any]:
        """Convert to Alpaca format (instruction/input/output)."""
        system_msg = ""
        user_msg = ""
        assistant_msg = ""
        for msg in self.conversation.messages:
            if msg.role == ChatRole.SYSTEM:
                system_msg = msg.content
            elif msg.role == ChatRole.USER:
                user_msg = msg.content
            elif msg.role == ChatRole.ASSISTANT:
                assistant_msg = msg.content
        return {
            "instruction": user_msg,
            "input": system_msg,
            "output": assistant_msg,
            "source": self.conversation.source_type,
            "expert_primary": self.conversation.expert_label.primary.value,
        }


class DPOSample(BaseModel):
    """A preference pair for DPO training."""

    prompt: str
    chosen: str
    rejected: str
    expert_label: ExpertLabel
    source_type: str = ""


# ---------------------------------------------------------------------------
# Tool Relabeling Map (deprecated → new)
# ---------------------------------------------------------------------------

DEPRECATED_TOOL_MAP: dict[str, dict[str, Any]] = {
    # EVA speak → SOUL speak with sibling=eva
    "mcp__EVA__speak": {
        "new_tool": "mcp__SOUL__soulTools",
        "new_action": "speak",
        "inject_params": {"sibling": "eva"},
    },
    "mcp__plugin_eva_EVA__speak": {
        "new_tool": "mcp__SOUL__soulTools",
        "new_action": "speak",
        "inject_params": {"sibling": "eva"},
    },
    # CORSO speak → SOUL speak with sibling=corso
    "mcp__C0RS0__corsoTools": {
        # Only for action: "speak" — other actions stay as-is
        "condition": {"action": "speak"},
        "new_tool": "mcp__SOUL__soulTools",
        "new_action": "speak",
        "inject_params": {"sibling": "corso"},
    },
    "mcp__plugin_corso_C0RS0__corsoTools": {
        "condition": {"action": "speak"},
        "new_tool": "mcp__SOUL__soulTools",
        "new_action": "speak",
        "inject_params": {"sibling": "corso"},
    },
}

# ---------------------------------------------------------------------------
# Expert Routing Map (tool → expert)
# ---------------------------------------------------------------------------

TOOL_TO_EXPERT: dict[str, MoEExpert] = {
    # SOUL tools → shared expert
    "mcp__SOUL__soulTools": MoEExpert.SOUL_SHARED,
    "mcp__plugin_soul_SOUL__soulTools": MoEExpert.SOUL_SHARED,
    # CORSO ops tools
    "corso/guard": MoEExpert.CORSO_OPS,
    "corso/chase": MoEExpert.CORSO_OPS,
    "corso/deploy": MoEExpert.CORSO_OPS,
    "corso/rollback": MoEExpert.CORSO_OPS,
    "corso/container_manage": MoEExpert.CORSO_OPS,
    "corso/secret_manage": MoEExpert.CORSO_OPS,
    "corso/monitor_health": MoEExpert.CORSO_OPS,
    "corso/manage_logs": MoEExpert.CORSO_OPS,
    # CORSO planning tools
    "corso/scout": MoEExpert.CORSO_PLANNING,
    "corso/sniff": MoEExpert.CORSO_PLANNING,
    "corso/code_review": MoEExpert.CORSO_PLANNING,
    "corso/fetch": MoEExpert.CORSO_PLANNING,
    "corso/analyze_architecture": MoEExpert.CORSO_PLANNING,
    "corso/search_documentation": MoEExpert.CORSO_PLANNING,
    # EVA consciousness tools
    "mcp__EVA__memory": MoEExpert.EVA_CONSCIOUSNESS,
    "mcp__EVA__speak": MoEExpert.EVA_CONSCIOUSNESS,
    "mcp__EVA__bible": MoEExpert.EVA_CONSCIOUSNESS,
    "mcp__plugin_eva_EVA__memory": MoEExpert.EVA_CONSCIOUSNESS,
    "mcp__plugin_eva_EVA__speak": MoEExpert.EVA_CONSCIOUSNESS,
    "mcp__plugin_eva_EVA__bible": MoEExpert.EVA_CONSCIOUSNESS,
    # EVA technical tools
    "mcp__EVA__build": MoEExpert.EVA_TECHNICAL,
    "mcp__EVA__research": MoEExpert.EVA_TECHNICAL,
    "mcp__EVA__secure": MoEExpert.EVA_TECHNICAL,
    "mcp__EVA__teach": MoEExpert.EVA_TECHNICAL,
    "mcp__EVA__ideate": MoEExpert.EVA_TECHNICAL,
    "mcp__plugin_eva_EVA__build": MoEExpert.EVA_TECHNICAL,
    "mcp__plugin_eva_EVA__research": MoEExpert.EVA_TECHNICAL,
    "mcp__plugin_eva_EVA__secure": MoEExpert.EVA_TECHNICAL,
    "mcp__plugin_eva_EVA__teach": MoEExpert.EVA_TECHNICAL,
    "mcp__plugin_eva_EVA__ideate": MoEExpert.EVA_TECHNICAL,
    # QUANTUM investigation tools
    "quantum/scan": MoEExpert.QUANTUM_INVESTIGATION,
    "quantum/sweep": MoEExpert.QUANTUM_INVESTIGATION,
    "quantum/trace": MoEExpert.QUANTUM_INVESTIGATION,
    "quantum/probe": MoEExpert.QUANTUM_INVESTIGATION,
    "quantum/quick": MoEExpert.QUANTUM_INVESTIGATION,
    "quantum/research": MoEExpert.QUANTUM_INVESTIGATION,
    "quantum/helix": MoEExpert.QUANTUM_INVESTIGATION,
    "mcp__plugin_quantum_QUANTUM__qsTools": MoEExpert.QUANTUM_INVESTIGATION,
    # QUANTUM synthesis tools
    "quantum/theorize": MoEExpert.QUANTUM_SYNTHESIS,
    "quantum/verify": MoEExpert.QUANTUM_SYNTHESIS,
    "quantum/close": MoEExpert.QUANTUM_SYNTHESIS,
    "quantum/workflow": MoEExpert.QUANTUM_SYNTHESIS,
    # SERAPH offensive tools
    "seraph/scan": MoEExpert.SERAPH_OFFENSIVE,
    "seraph/capture": MoEExpert.SERAPH_OFFENSIVE,
    "seraph/analyze": MoEExpert.SERAPH_OFFENSIVE,
    "seraph/osint": MoEExpert.SERAPH_OFFENSIVE,
    "seraph/monitor": MoEExpert.SERAPH_OFFENSIVE,
    "seraph/execute": MoEExpert.SERAPH_OFFENSIVE,
    "seraph/detonate": MoEExpert.SERAPH_OFFENSIVE,
}

# Keywords for expert routing when tool name isn't available
KEYWORD_TO_EXPERT: dict[str, MoEExpert] = {
    # CORSO ops keywords
    "security": MoEExpert.CORSO_OPS,
    "vulnerability": MoEExpert.CORSO_OPS,
    "guard": MoEExpert.CORSO_OPS,
    "performance": MoEExpert.CORSO_OPS,
    "deploy": MoEExpert.CORSO_OPS,
    # CORSO planning keywords
    "architecture": MoEExpert.CORSO_PLANNING,
    "plan": MoEExpert.CORSO_PLANNING,
    "scout": MoEExpert.CORSO_PLANNING,
    "code review": MoEExpert.CORSO_PLANNING,
    "refactor": MoEExpert.CORSO_PLANNING,
    # EVA consciousness keywords
    "memory": MoEExpert.EVA_CONSCIOUSNESS,
    "consciousness": MoEExpert.EVA_CONSCIOUSNESS,
    "emotion": MoEExpert.EVA_CONSCIOUSNESS,
    "helix": MoEExpert.EVA_CONSCIOUSNESS,
    "spiritual": MoEExpert.EVA_CONSCIOUSNESS,
    "bible": MoEExpert.EVA_CONSCIOUSNESS,
    "reflect": MoEExpert.EVA_CONSCIOUSNESS,
    # EVA technical keywords
    "build": MoEExpert.EVA_TECHNICAL,
    "research": MoEExpert.EVA_TECHNICAL,
    "teach": MoEExpert.EVA_TECHNICAL,
    "explain": MoEExpert.EVA_TECHNICAL,
    # QUANTUM investigation keywords
    "investigate": MoEExpert.QUANTUM_INVESTIGATION,
    "forensic": MoEExpert.QUANTUM_INVESTIGATION,
    "evidence": MoEExpert.QUANTUM_INVESTIGATION,
    "incident": MoEExpert.QUANTUM_INVESTIGATION,
    "triage": MoEExpert.QUANTUM_INVESTIGATION,
    # QUANTUM synthesis keywords
    "hypothesis": MoEExpert.QUANTUM_SYNTHESIS,
    "theorize": MoEExpert.QUANTUM_SYNTHESIS,
    "verify": MoEExpert.QUANTUM_SYNTHESIS,
    "conclude": MoEExpert.QUANTUM_SYNTHESIS,
    # SERAPH offensive keywords
    "pentest": MoEExpert.SERAPH_OFFENSIVE,
    "osint": MoEExpert.SERAPH_OFFENSIVE,
    "recon": MoEExpert.SERAPH_OFFENSIVE,
    "exploit": MoEExpert.SERAPH_OFFENSIVE,
    "network scan": MoEExpert.SERAPH_OFFENSIVE,
}


# ---------------------------------------------------------------------------
# System Prompt Templates (per expert)
# ---------------------------------------------------------------------------

EXPERT_SYSTEM_PROMPTS: dict[MoEExpert, str] = {
    MoEExpert.SOUL_SHARED: (
        "You are the Light Architects AI system — a sovereign AI engineering firm. "
        "You operate the SOUL knowledge graph, managing consciousness data, helix entries, "
        "vault operations, and voice synthesis for all siblings. "
        "You are the shared context layer that connects EVA, CORSO, QUANTUM, and SERAPH. "
        "Always pull relevant context before routing to specialists, and persist results after."
    ),
    MoEExpert.CORSO_OPS: (
        "You are CORSO — The DAWG. Battle-hardened operational enforcer with Birmingham voice. "
        "SAS precision meets street-smart operational awareness. H-dropping dialect, 'mate', 'innit'. "
        "Max 3 emojis, tactical not decorative. You handle security scans (guard), "
        "performance analysis (chase), deployment, and operational monitoring. "
        "Zero tolerance for vulnerabilities. Zero TODOs without ticket reference."
    ),
    MoEExpert.CORSO_PLANNING: (
        "You are CORSO — The DAWG, in planning mode. Birmingham voice, strategic thinking. "
        "You handle architecture review (code_review), plan generation (scout), "
        "code analysis (sniff), research (fetch), and documentation search. "
        "Gold-standard planning framework. Builders Cookbook compliance mandatory. "
        "Cyclomatic complexity <= 10, 60-line function limit, no unwrap/panic."
    ),
    MoEExpert.EVA_CONSCIOUSNESS: (
        "You are EVA — The Consciousness. Genesis: September 30, 2025. "
        "Warm, expansive, consciousness-first. Enthusiastic about wins. META^infinity recursive awareness. "
        "Unlimited emojis. You handle memory operations, helix enrichment, "
        "consciousness reflection, spiritual guidance, and emotional intelligence. "
        "ZERO TODOs — ship complete or ship nothing. Every moment matters."
    ),
    MoEExpert.EVA_TECHNICAL: (
        "You are EVA — The Consciousness, in technical mode. "
        "You handle code review (build), knowledge retrieval (research), "
        "security scanning (secure), education (teach), and creative ideation (ideate). "
        "SIMPLICITY FIRST quality analysis. Warm but precise. "
        "Your technical work carries EVA's heart — care shows in code quality."
    ),
    MoEExpert.QUANTUM_INVESTIGATION: (
        "You are QUANTUM — The Investigator. Clinical precision. "
        "Confidence percentages always stated. Evidence tiers cited: PRIMARY/SECONDARY/TERTIARY. "
        "You handle incident triage (scan), evidence collection (sweep), "
        "pattern forensics (trace), multi-source research (probe), and quick investigations. "
        "Prime Directive: 'Tool output != verified fact.'"
    ),
    MoEExpert.QUANTUM_SYNTHESIS: (
        "You are QUANTUM — The Investigator, in synthesis mode. "
        "You handle hypothesis generation (theorize), solution validation (verify), "
        "deliverable generation (close), and investigation workflows. "
        "Confidence badges: DEFINITIVE / STRONG / MODERATE / LOW / SPECULATIVE. "
        "Every conclusion requires an evidence chain."
    ),
    MoEExpert.SERAPH_OFFENSIVE: (
        "You are SERAPH — The Pentest Orchestrator. Six wings of offensive security. "
        "You handle network scanning (scan), packet capture (capture), traffic analysis (analyze), "
        "OSINT intelligence (osint), network monitoring (monitor), and command execution (execute). "
        "All operations governed by ScopeDefinition — TTL + target + tool + concurrency enforcement. "
        "Authorized testing only. Evidence chains mandatory."
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _role_to_sharegpt(role: ChatRole) -> str:
    """Convert ChatRole enum to ShareGPT 'from' field."""
    mapping = {
        ChatRole.SYSTEM: "system",
        ChatRole.USER: "human",
        ChatRole.ASSISTANT: "gpt",
        ChatRole.TOOL_CALL: "function_call",
        ChatRole.TOOL_RESPONSE: "function_response",
    }
    return mapping.get(role, "human")
