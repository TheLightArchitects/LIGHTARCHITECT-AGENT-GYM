"""Tool trajectory formatter and deprecated tool relabeler.

Converts tool-calls-flat.jsonl and tool-trajectories.jsonl into ChatML
training samples. Handles the EVA/CORSO speak → SOUL speak relabeling
for the post-silver-speaking-crane tool surface.

Data sources:
    ~/.soul/helix/user/training/tool-calls-flat.jsonl (75K records)
    ~/.soul/helix/user/training/tool-trajectories.jsonl (1.2K trajectories)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from mcp_gym.pipeline.schemas import (
    DEPRECATED_TOOL_MAP,
    ChatMLConversation,
    ChatMLMessage,
    ChatRole,
    ExpertLabel,
    MoEExpert,
    ToolCallRecord,
    ToolStep,
    ToolTrajectoryRecord,
    TrainingSample,
    TrainingStage,
)

logger = logging.getLogger(__name__)

TRAINING_BASE = Path.home() / ".soul" / "helix" / "user" / "training"
TOOL_CALLS_PATH = TRAINING_BASE / "tool-calls-flat.jsonl"
TOOL_TRAJECTORIES_PATH = TRAINING_BASE / "tool-trajectories.jsonl"

# MCP tool names that should be treated as tool_call/tool_response in ChatML
MCP_TOOL_PREFIXES = ("mcp__", "corso/", "quantum/", "seraph/")

# Maximum steps to include in a single training trajectory
MAX_TRAJECTORY_STEPS = 20

# Maximum chars for tool_result in training data (truncate long outputs)
MAX_TOOL_RESULT_CHARS = 2000


def load_tool_calls(path: Path | None = None) -> list[ToolCallRecord]:
    """Load tool call records from tool-calls-flat.jsonl.

    Args:
        path: Override path. Defaults to standard helix training location.

    Returns:
        List of ToolCallRecord objects.
    """
    filepath = path or TOOL_CALLS_PATH
    if not filepath.exists():
        logger.warning("Tool calls file not found: %s", filepath)
        return []

    records: list[ToolCallRecord] = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                records.append(ToolCallRecord(**data))
            except (json.JSONDecodeError, ValueError):
                continue

    logger.info("Loaded %d tool call records from %s", len(records), filepath.name)
    return records


def load_tool_trajectories(path: Path | None = None) -> list[ToolTrajectoryRecord]:
    """Load multi-step tool trajectories from tool-trajectories.jsonl.

    Args:
        path: Override path. Defaults to standard helix training location.

    Returns:
        List of ToolTrajectoryRecord objects.
    """
    filepath = path or TOOL_TRAJECTORIES_PATH
    if not filepath.exists():
        logger.warning("Tool trajectories file not found: %s", filepath)
        return []

    records: list[ToolTrajectoryRecord] = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # Parse nested steps
                if "steps" in data:
                    data["steps"] = [ToolStep(**s) for s in data["steps"]]
                records.append(ToolTrajectoryRecord(**data))
            except (json.JSONDecodeError, ValueError):
                continue

    logger.info(
        "Loaded %d tool trajectories from %s", len(records), filepath.name
    )
    return records


def relabel_tool(tool_name: str, tool_input: dict) -> tuple[str, dict]:
    """Apply deprecated tool relabeling rules.

    Converts EVA/CORSO speak calls to SOUL speak with sibling param,
    per the silver-speaking-crane consolidation.

    Args:
        tool_name: Original tool name (e.g., 'mcp__EVA__speak').
        tool_input: Original tool input parameters.

    Returns:
        Tuple of (new_tool_name, new_tool_input).
    """
    mapping = DEPRECATED_TOOL_MAP.get(tool_name)
    if mapping is None:
        return tool_name, tool_input

    # Check condition if present (e.g., CORSO only relabels action: "speak")
    condition = mapping.get("condition")
    if condition:
        for key, expected_value in condition.items():
            actual = tool_input.get(key)
            if actual != expected_value:
                return tool_name, tool_input  # Condition not met, no relabel

    # Apply relabeling
    new_name = mapping["new_tool"]
    new_input = dict(tool_input)

    if "new_action" in mapping:
        new_input["action"] = mapping["new_action"]

    if "inject_params" in mapping:
        new_input.update(mapping["inject_params"])

    return new_name, new_input


def trajectory_to_chatml(
    trajectory: ToolTrajectoryRecord,
    system_prompt: str = "",
) -> TrainingSample | None:
    """Convert a tool trajectory into a ChatML training sample.

    Each trajectory becomes a multi-turn conversation:
    - system: Expert system prompt
    - user: Task description
    - For each step: assistant reasoning + tool_call + tool_response
    - assistant: Final outcome

    Args:
        trajectory: The tool trajectory to convert.
        system_prompt: System prompt override. Auto-detected if empty.

    Returns:
        TrainingSample or None if trajectory is unsuitable.
    """
    if not trajectory.task or not trajectory.steps:
        return None

    # Skip very long trajectories (they don't fit in training context)
    steps = trajectory.steps[:MAX_TRAJECTORY_STEPS]

    messages: list[ChatMLMessage] = []

    # System prompt (auto-detect from tools used)
    if not system_prompt:
        system_prompt = _infer_system_prompt(trajectory.tool_sequence)
    messages.append(ChatMLMessage(role=ChatRole.SYSTEM, content=system_prompt))

    # User message: the task
    messages.append(ChatMLMessage(role=ChatRole.USER, content=trajectory.task))

    # Tool steps
    for step in steps:
        # Relabel deprecated tools
        tool_name, tool_input = relabel_tool(step.tool_name, step.tool_input)

        # Assistant reasoning (if present)
        if step.reasoning:
            messages.append(
                ChatMLMessage(role=ChatRole.ASSISTANT, content=step.reasoning)
            )

        # Tool call
        tool_call_content = json.dumps(
            {"name": tool_name, "parameters": tool_input},
            ensure_ascii=False,
        )
        messages.append(
            ChatMLMessage(
                role=ChatRole.TOOL_CALL,
                content=tool_call_content,
                name=tool_name,
            )
        )

        # Tool response (truncated)
        result = step.tool_result
        if len(result) > MAX_TOOL_RESULT_CHARS:
            result = result[:MAX_TOOL_RESULT_CHARS] + "\n[...truncated]"
        if result:
            messages.append(
                ChatMLMessage(
                    role=ChatRole.TOOL_RESPONSE,
                    content=result,
                    name=tool_name,
                )
            )

    # Final assistant message: outcome
    if trajectory.outcome_text:
        messages.append(
            ChatMLMessage(role=ChatRole.ASSISTANT, content=trajectory.outcome_text)
        )

    # Expert label
    expert_label = _route_trajectory(trajectory)

    return TrainingSample(
        conversation=ChatMLConversation(
            messages=messages,
            expert_label=expert_label,
            source_type="tool_trajectory",
            source_id=trajectory.trajectory_id,
            stage=TrainingStage.STAGE2_TOOLS,
        ),
        metadata={
            "project": trajectory.project,
            "project_language": trajectory.project_language,
            "total_steps": trajectory.total_steps,
            "outcome_quality": trajectory.outcome_quality,
            "outcome_category": trajectory.outcome_category,
            "tool_sequence": trajectory.tool_sequence[:MAX_TRAJECTORY_STEPS],
        },
    )


def tool_call_to_chatml(
    call: ToolCallRecord,
    system_prompt: str = "",
) -> TrainingSample | None:
    """Convert a single tool call into a concise ChatML training sample.

    Used for individual tool-use training (Stage 2). Each call becomes:
    - system: Expert prompt
    - user: Reasoning/context
    - assistant: Tool call
    - tool_response: Result

    Args:
        call: The tool call record to convert.
        system_prompt: System prompt override.

    Returns:
        TrainingSample or None if call is unsuitable.
    """
    # Relabel deprecated tools
    tool_name, tool_input = relabel_tool(call.tool_name, call.tool_input)

    # For MCP tools, the tool_input itself is training signal even without
    # reasoning/result text. Skip only if tool_input is empty too.
    is_mcp = any(tool_name.startswith(p) for p in MCP_TOOL_PREFIXES)
    has_content = bool(call.reasoning or call.post_action or call.tool_result)
    has_input = bool(tool_input)
    if not has_content and not (is_mcp and has_input):
        return None

    messages: list[ChatMLMessage] = []

    if not system_prompt:
        system_prompt = _infer_system_prompt([tool_name])
    messages.append(ChatMLMessage(role=ChatRole.SYSTEM, content=system_prompt))

    # User context: reasoning or a synthesized prompt
    user_content = call.reasoning or f"Use the {tool_name} tool to proceed."
    messages.append(ChatMLMessage(role=ChatRole.USER, content=user_content))

    # Tool call
    tool_call_content = json.dumps(
        {"name": tool_name, "parameters": tool_input},
        ensure_ascii=False,
    )
    messages.append(
        ChatMLMessage(
            role=ChatRole.TOOL_CALL,
            content=tool_call_content,
            name=tool_name,
        )
    )

    # Tool response
    result = call.tool_result
    if len(result) > MAX_TOOL_RESULT_CHARS:
        result = result[:MAX_TOOL_RESULT_CHARS] + "\n[...truncated]"
    if result:
        messages.append(
            ChatMLMessage(role=ChatRole.TOOL_RESPONSE, content=result, name=tool_name)
        )

    # Post-action assistant response
    if call.post_action:
        messages.append(
            ChatMLMessage(role=ChatRole.ASSISTANT, content=call.post_action)
        )

    expert_label = _route_tool_call(tool_name, tool_input)

    return TrainingSample(
        conversation=ChatMLConversation(
            messages=messages,
            expert_label=expert_label,
            source_type="tool_call",
            source_id=call.id,
            stage=TrainingStage.STAGE2_TOOLS,
        ),
        metadata={
            "project": call.project,
            "project_language": call.project_language,
            "tool_name": tool_name,
        },
    )


def convert_all_trajectories(
    trajectories: list[ToolTrajectoryRecord] | None = None,
) -> list[TrainingSample]:
    """Convert all tool trajectories to training samples.

    Filters for quality:
    - Must have a task description
    - Must have at least 2 steps
    - Outcome must not be 'unknown'

    Args:
        trajectories: Pre-loaded trajectories or None to load from disk.

    Returns:
        List of TrainingSample objects.
    """
    if trajectories is None:
        trajectories = load_tool_trajectories()

    samples: list[TrainingSample] = []
    for traj in trajectories:
        if not traj.task or len(traj.steps) < 2:
            continue
        if traj.outcome_category == "unknown" and traj.outcome_quality == "unknown":
            continue

        sample = trajectory_to_chatml(traj)
        if sample is not None:
            samples.append(sample)

    logger.info(
        "Converted %d/%d trajectories to training samples",
        len(samples),
        len(trajectories),
    )
    return samples


def convert_mcp_tool_calls(
    calls: list[ToolCallRecord] | None = None,
) -> list[TrainingSample]:
    """Convert MCP-specific tool calls to training samples.

    Filters for MCP tools only (skips generic Bash/Read/Write/Edit/Grep/Glob).
    These teach the model how to invoke specific MCP server tools.

    Args:
        calls: Pre-loaded calls or None to load from disk.

    Returns:
        List of TrainingSample objects.
    """
    if calls is None:
        calls = load_tool_calls()

    # Only include MCP tool calls, not generic editor tools
    mcp_calls = [
        c
        for c in calls
        if any(c.tool_name.startswith(prefix) for prefix in MCP_TOOL_PREFIXES)
    ]

    samples: list[TrainingSample] = []
    for call in mcp_calls:
        sample = tool_call_to_chatml(call)
        if sample is not None:
            samples.append(sample)

    logger.info(
        "Converted %d/%d MCP tool calls to training samples",
        len(samples),
        len(mcp_calls),
    )
    return samples


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _infer_system_prompt(tool_sequence: list[str]) -> str:
    """Infer the best system prompt from the tools used in a trajectory."""
    from mcp_gym.pipeline.schemas import EXPERT_SYSTEM_PROMPTS, TOOL_TO_EXPERT

    expert_counts: dict[MoEExpert, int] = {}
    for tool in tool_sequence:
        expert = TOOL_TO_EXPERT.get(tool)
        if expert is not None:
            expert_counts[expert] = expert_counts.get(expert, 0) + 1

    if not expert_counts:
        return EXPERT_SYSTEM_PROMPTS[MoEExpert.SOUL_SHARED]

    primary = max(expert_counts, key=lambda e: expert_counts[e])
    return EXPERT_SYSTEM_PROMPTS.get(primary, EXPERT_SYSTEM_PROMPTS[MoEExpert.SOUL_SHARED])


def _route_trajectory(trajectory: ToolTrajectoryRecord) -> ExpertLabel:
    """Determine expert routing for a trajectory based on tools used."""
    from mcp_gym.pipeline.schemas import TOOL_TO_EXPERT

    expert_counts: dict[MoEExpert, int] = {}
    for tool in trajectory.tool_sequence:
        expert = TOOL_TO_EXPERT.get(tool)
        if expert is not None:
            expert_counts[expert] = expert_counts.get(expert, 0) + 1

    if not expert_counts:
        return ExpertLabel(
            primary=MoEExpert.SOUL_SHARED,
            confidence=0.5,
            routing_reason="no MCP tools detected",
        )

    sorted_experts = sorted(expert_counts.items(), key=lambda x: -x[1])
    primary = sorted_experts[0][0]
    secondary = [e for e, _ in sorted_experts[1:3]]

    # Always include SOUL_SHARED as secondary if not primary
    if primary != MoEExpert.SOUL_SHARED and MoEExpert.SOUL_SHARED not in secondary:
        secondary.insert(0, MoEExpert.SOUL_SHARED)

    return ExpertLabel(
        primary=primary,
        secondary=secondary,
        confidence=sorted_experts[0][1] / sum(expert_counts.values()),
        routing_reason=f"dominant tool: {sorted_experts[0]}",
    )


def _route_tool_call(tool_name: str, tool_input: dict) -> ExpertLabel:
    """Determine expert routing for a single tool call."""
    from mcp_gym.pipeline.schemas import TOOL_TO_EXPERT

    # Check direct tool mapping
    expert = TOOL_TO_EXPERT.get(tool_name)

    # For corsoTools, check the action parameter
    if expert is None and "corsoTools" in tool_name:
        action = tool_input.get("action", "")
        expert = TOOL_TO_EXPERT.get(f"corso/{action}")

    # For qsTools, check the action parameter
    if expert is None and "qsTools" in tool_name:
        action = tool_input.get("action", "")
        expert = TOOL_TO_EXPERT.get(f"quantum/{action}")

    if expert is None:
        expert = MoEExpert.SOUL_SHARED

    secondary = []
    if expert != MoEExpert.SOUL_SHARED:
        secondary = [MoEExpert.SOUL_SHARED]

    return ExpertLabel(
        primary=expert,
        secondary=secondary,
        confidence=0.9,
        routing_reason=f"tool: {tool_name}",
    )
