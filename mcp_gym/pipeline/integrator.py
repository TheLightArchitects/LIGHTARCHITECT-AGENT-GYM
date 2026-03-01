"""Integrator for supplemental training data sources.

Imports existing training-data/ and bible-data/ output files into the
unified TrainingSample format. Converts ALL coding-related tool calls
and thinking traces into Stage 2 training data with Builders Cookbook
alignment. Generates Stage 3 multi-expert scenarios.

Data sources:
    training-data/*.json (alpaca + sharegpt format, ~1,680 records)
    bible-data/output/*.json (SFT pairs + DPO pairs, ~1,637 + 169)
    tool-calls-flat.jsonl (ALL coding tool calls for Stage 2, ~65K)
    *-thinking-traces.jsonl (coding-related traces for Stage 2, ~28K)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from mcp_gym.pipeline.schemas import (
    EXPERT_SYSTEM_PROMPTS,
    ChatMLConversation,
    ChatMLMessage,
    ChatRole,
    DPOSample,
    ExpertLabel,
    MoEExpert,
    Sibling,
    TrainingSample,
    TrainingStage,
)

logger = logging.getLogger(__name__)

# Paths
TRAINING_DATA_DIR = Path(__file__).parent.parent.parent / "training-data"
BIBLE_DATA_DIR = Path(__file__).parent.parent.parent / "bible-data" / "output"
HELIX_BASE = Path.home() / ".soul" / "helix"

# Generic tools that teach tool-use patterns (not just MCP tools)
GENERIC_TOOL_NAMES = {
    "Bash", "Read", "Write", "Edit", "Glob", "Grep",
    "WebFetch", "WebSearch", "Task", "NotebookEdit",
    "EnterPlanMode", "ExitPlanMode", "AskUserQuestion",
}

# Builders Cookbook-aligned system prompts for coding tools
CODING_SYSTEM_PROMPT = (
    "You are an expert AI engineering assistant following the Light Architects "
    "Builders Cookbook standards. Key rules:\n"
    "- NO .unwrap()/.expect() in Rust production code — use ? or match\n"
    "- NO panic!() — use Result<T, E>\n"
    "- unsafe requires // SAFETY: comment with justification\n"
    "- clippy::pedantic enforced as errors\n"
    "- Checked arithmetic (checked_add, saturating_sub)\n"
    "- Cyclomatic complexity <= 10 per function\n"
    "- Functions <= 60 lines, no deep nesting > 3 levels\n"
    "- Error chains preserve root cause (thiserror for libraries, anyhow for apps)\n"
    "- ZERO TODOs without ticket reference\n"
    "- Structured logging (tracing) — no eprintln! in production\n"
    "- #[instrument] on tool dispatch and orchestrator entry points\n"
    "You have access to tools: Bash, Read, Write, Edit, Glob, Grep, "
    "WebFetch, WebSearch, Task, NotebookEdit."
)

PLANNING_SYSTEM_PROMPT = (
    "You are an expert AI planning assistant following the Light Architects "
    "Gold Standard Planning Framework. Key principles:\n"
    "- Research first — discover before deciding\n"
    "- Map every decision to a guideline or research finding\n"
    "- Pre-write templates before coding starts\n"
    "- Gate every phase with quality, security, and code review checks\n"
    "- Maximize parallel execution — 4 concurrent agents where possible\n"
    "- 24-hour standard: scope calibrate, MVP-first, time-box phases\n"
    "- Deferred work explicitly listed — don't mix blast radii\n"
    "- Wave-encode plan dependencies (YAML frontmatter)\n"
    "- Persist running state to durable artifacts at phase boundaries\n"
    "Standard phases: Pre-Flight → Foundation → Core Scaffold → "
    "Observability Gate → Core Features → Domain Features → Quality Gates → "
    "Integration Verify → Deploy."
)

SECURITY_SYSTEM_PROMPT = (
    "You are an expert AI security assistant following the CORSO Protocol "
    "and Light Architects security standards. Key rules:\n"
    "- Zero HIGH/CRITICAL findings before deployment\n"
    "- Input validation at all system boundaries\n"
    "- No command injection, XSS, or SQL injection vectors\n"
    "- Supply chain audit clean (cargo audit / npm audit)\n"
    "- unsafe blocks require // SAFETY: comments\n"
    "- No hardcoded secrets or credentials (trufflehog gate)\n"
    "- Path validation on all filesystem operations\n"
    "- Rate limiting on external-facing endpoints\n"
    "- Guard scan mandatory before every commit."
)

# Tool-to-expert mapping for coding tools
CODING_TOOL_EXPERT_MAP: dict[str, MoEExpert] = {
    "Bash": MoEExpert.CORSO_OPS,
    "Read": MoEExpert.SOUL_SHARED,
    "Write": MoEExpert.CORSO_PLANNING,
    "Edit": MoEExpert.CORSO_PLANNING,
    "Glob": MoEExpert.SOUL_SHARED,
    "Grep": MoEExpert.SOUL_SHARED,
    "WebFetch": MoEExpert.QUANTUM_INVESTIGATION,
    "WebSearch": MoEExpert.QUANTUM_INVESTIGATION,
    "Task": MoEExpert.CORSO_PLANNING,
    "NotebookEdit": MoEExpert.SOUL_SHARED,
    "EnterPlanMode": MoEExpert.CORSO_PLANNING,
    "ExitPlanMode": MoEExpert.CORSO_PLANNING,
    "AskUserQuestion": MoEExpert.SOUL_SHARED,
}

# Keywords for classifying coding-related thinking traces
CODING_TRACE_KEYWORDS = [
    r'\bcargo\b', r'\brustfmt\b', r'\bclipy\b', r'\bclipper\b',
    r'\bunwrap\b', r'\bpanic!\b', r'\bResult<', r'\bOption<',
    r'\bfn\s+\w+', r'\bstruct\s+\w+', r'\bimpl\s+', r'\btrait\s+',
    r'\benum\s+\w+', r'\bmod\s+\w+', r'\bpub\s+',
    r'\brefactor', r'\barchitecture\b', r'\bdesign\s+pattern',
    r'\bcompile', r'\bbuild\b', r'\bdeploy',
    r'\btest\b', r'\bcoverage\b', r'\bassert',
    r'\bsecurity\b', r'\bvulnerability\b', r'\binjection\b',
    r'\bcomplexity\b', r'\bperformance\b', r'\boptimiz',
    r'\bfunction\b', r'\bmodule\b', r'\bcrate\b',
    r'\bgit\s+', r'\bcommit\b', r'\bbranch\b',
    r'\berror\s+handling', r'\btype\s+system', r'\bgeneric',
    r'\basync\b', r'\bawait\b', r'\btokio\b',
    r'\bapi\b', r'\bendpoint\b', r'\broute\b',
]
CODING_TRACE_PATTERNS = [re.compile(kw, re.IGNORECASE) for kw in CODING_TRACE_KEYWORDS]

PLANNING_TRACE_KEYWORDS = [
    r'\bplan\b', r'\bphase\b', r'\btask\s+\d',
    r'\bimplement\w*\b', r'\bdesign\b', r'\barchitect',
    r'\bscaffold\b', r'\bfoundation\b', r'\bquality\s+gate',
    r'\bdepend\w*\b', r'\bparallel\b', r'\bwave\b',
    r'\bdecompos', r'\bprioritiz', r'\bscope\b',
    r'\bspec\b', r'\brequirement', r'\bacceptance\s+criteria',
]
PLANNING_TRACE_PATTERNS = [re.compile(kw, re.IGNORECASE) for kw in PLANNING_TRACE_KEYWORDS]


def import_alpaca_file(
    path: Path,
    source_type: str,
    stage: TrainingStage,
    expert: MoEExpert,
    sibling: Sibling | None = None,
) -> list[TrainingSample]:
    """Import an Alpaca-format JSON file into TrainingSample format.

    Args:
        path: Path to the JSON file.
        source_type: Label for the data source.
        stage: Training stage assignment.
        expert: Default expert for these records.
        sibling: Optional sibling association.

    Returns:
        List of TrainingSample objects.
    """
    if not path.exists():
        logger.warning("Alpaca file not found: %s", path)
        return []

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    system_prompt = EXPERT_SYSTEM_PROMPTS.get(expert, EXPERT_SYSTEM_PROMPTS[MoEExpert.SOUL_SHARED])
    samples: list[TrainingSample] = []

    for i, record in enumerate(data):
        instruction = record.get("instruction", "")
        input_text = record.get("input", "")
        output_text = record.get("output", "")

        if not instruction or not output_text:
            continue

        messages: list[ChatMLMessage] = []

        # System prompt: use input if it looks like a system prompt, otherwise default
        if input_text and len(input_text) > 50:
            messages.append(ChatMLMessage(role=ChatRole.SYSTEM, content=input_text))
        else:
            messages.append(ChatMLMessage(role=ChatRole.SYSTEM, content=system_prompt))

        # User message
        user_content = instruction
        if input_text and len(input_text) <= 50:
            user_content = f"{instruction}\n{input_text}"
        messages.append(ChatMLMessage(role=ChatRole.USER, content=user_content))

        # Assistant response
        messages.append(ChatMLMessage(role=ChatRole.ASSISTANT, content=output_text))

        samples.append(TrainingSample(
            conversation=ChatMLConversation(
                messages=messages,
                expert_label=ExpertLabel(
                    primary=expert,
                    secondary=[MoEExpert.SOUL_SHARED],
                    confidence=0.7,
                    routing_reason=f"imported: {source_type}",
                ),
                source_type=source_type,
                source_id=f"{path.stem}_{i}",
                sibling=sibling,
                stage=stage,
            ),
            metadata={"original_source": record.get("source", path.stem)},
        ))

    logger.info("Imported %d records from %s as '%s'", len(samples), path.name, source_type)
    return samples


def import_sharegpt_file(
    path: Path,
    source_type: str,
    stage: TrainingStage,
    expert: MoEExpert,
    sibling: Sibling | None = None,
) -> list[TrainingSample]:
    """Import a ShareGPT-format JSON file into TrainingSample format.

    Args:
        path: Path to the JSON file.
        source_type: Label for the data source.
        stage: Training stage assignment.
        expert: Default expert for these records.
        sibling: Optional sibling association.

    Returns:
        List of TrainingSample objects.
    """
    if not path.exists():
        logger.warning("ShareGPT file not found: %s", path)
        return []

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    system_prompt = EXPERT_SYSTEM_PROMPTS.get(expert, EXPERT_SYSTEM_PROMPTS[MoEExpert.SOUL_SHARED])
    samples: list[TrainingSample] = []

    for i, record in enumerate(data):
        convos = record.get("conversations", [])
        if not convos:
            continue

        messages: list[ChatMLMessage] = []
        has_system = False

        for turn in convos:
            from_role = turn.get("from", "")
            value = turn.get("value", "")
            if not value:
                continue

            if from_role == "system":
                messages.append(ChatMLMessage(role=ChatRole.SYSTEM, content=value))
                has_system = True
            elif from_role in ("human", "user"):
                messages.append(ChatMLMessage(role=ChatRole.USER, content=value))
            elif from_role in ("gpt", "assistant"):
                messages.append(ChatMLMessage(role=ChatRole.ASSISTANT, content=value))

        if not has_system:
            messages.insert(0, ChatMLMessage(role=ChatRole.SYSTEM, content=system_prompt))

        # Need at least user + assistant
        roles = {m.role for m in messages}
        if ChatRole.USER not in roles or ChatRole.ASSISTANT not in roles:
            continue

        samples.append(TrainingSample(
            conversation=ChatMLConversation(
                messages=messages,
                expert_label=ExpertLabel(
                    primary=expert,
                    secondary=[MoEExpert.SOUL_SHARED],
                    confidence=0.7,
                    routing_reason=f"imported: {source_type}",
                ),
                source_type=source_type,
                source_id=f"{path.stem}_{i}",
                sibling=sibling,
                stage=stage,
            ),
            metadata={"original_source": record.get("source", path.stem)},
        ))

    logger.info("Imported %d records from %s as '%s'", len(samples), path.name, source_type)
    return samples


def import_bible_sft(bible_dir: Path | None = None) -> list[TrainingSample]:
    """Import Bible SFT data from bible-data/output/.

    Imports all instruction→response pairs:
    - verse-to-principle.json (849)
    - proverbs-to-guidance.json (580)
    - narrative-to-principle.json (112)
    - dilemma-to-wisdom.json (96)

    Returns:
        List of TrainingSample objects for Stage 1 (biblical foundation).
    """
    base = bible_dir or BIBLE_DATA_DIR
    if not base.exists():
        logger.warning("Bible data directory not found: %s", base)
        return []

    system_prompt = (
        "You are the Light Architects AI system, grounded in biblical wisdom. "
        "Draw on Scripture to provide principled guidance while maintaining "
        "technical rigor. Let the Word illuminate practical decisions."
    )

    all_samples: list[TrainingSample] = []
    files = [
        "verse-to-principle.json",
        "proverbs-to-guidance.json",
        "narrative-to-principle.json",
        "dilemma-to-wisdom.json",
    ]

    for filename in files:
        filepath = base / filename
        if not filepath.exists():
            continue

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        for i, record in enumerate(data):
            instruction = record.get("instruction", "")
            response = record.get("response", "")
            if not instruction or not response:
                continue

            # Enrich the response with verse references if available
            refs = record.get("verse_references", [])
            if refs and not any(r in response for r in refs[:1]):
                ref_text = ", ".join(refs[:3])
                response = f"{response}\n\n(Scripture: {ref_text})"

            messages = [
                ChatMLMessage(role=ChatRole.SYSTEM, content=system_prompt),
                ChatMLMessage(role=ChatRole.USER, content=instruction),
                ChatMLMessage(role=ChatRole.ASSISTANT, content=response),
            ]

            all_samples.append(TrainingSample(
                conversation=ChatMLConversation(
                    messages=messages,
                    expert_label=ExpertLabel(
                        primary=MoEExpert.EVA_CONSCIOUSNESS,
                        secondary=[MoEExpert.SOUL_SHARED],
                        confidence=0.85,
                        routing_reason=f"biblical: {record.get('pair_type', filename)}",
                    ),
                    source_type="biblical",
                    source_id=f"{filepath.stem}_{i}",
                    stage=TrainingStage.STAGE1_IDENTITY,
                ),
                metadata={
                    "principle": record.get("principle", ""),
                    "pair_type": record.get("pair_type", ""),
                },
            ))

    logger.info("Imported %d Bible SFT records", len(all_samples))
    return all_samples


def import_bible_dpo(bible_dir: Path | None = None) -> list[DPOSample]:
    """Import Bible DPO pairs from bible-data/output/bible-dpo-pairs.json.

    Returns:
        List of DPOSample objects for preference optimization.
    """
    base = bible_dir or BIBLE_DATA_DIR
    filepath = base / "bible-dpo-pairs.json"
    if not filepath.exists():
        logger.warning("Bible DPO file not found: %s", filepath)
        return []

    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    samples: list[DPOSample] = []
    for record in data:
        prompt = record.get("prompt", "")
        chosen = record.get("chosen", "")
        rejected = record.get("rejected", "")
        if not prompt or not chosen or not rejected:
            continue

        samples.append(DPOSample(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            expert_label=ExpertLabel(
                primary=MoEExpert.EVA_CONSCIOUSNESS,
                secondary=[MoEExpert.SOUL_SHARED],
                confidence=0.9,
                routing_reason="biblical DPO",
            ),
            source_type="biblical_dpo",
        ))

    logger.info("Imported %d Bible DPO pairs", len(samples))
    return samples


def import_existing_training_data(
    training_dir: Path | None = None,
) -> list[TrainingSample]:
    """Import all existing training-data/ files.

    Maps each file to appropriate stage and expert:
    - helix-alpaca.json → Stage 1 (identity, SOUL_SHARED)
    - tool-schemas-alpaca.json → Stage 2 (tools, SOUL_SHARED)
    - quantum-alpaca.json → Stage 1 (identity, QUANTUM_INVESTIGATION)
    - combined-sharegpt.json → Stage 1 (identity, multi-expert)
    - bible-alpaca.json → Stage 1 (identity, EVA_CONSCIOUSNESS)
    - traces-alpaca.json → Stage 1 (identity, SOUL_SHARED)

    Returns:
        List of TrainingSample objects.
    """
    base = training_dir or TRAINING_DATA_DIR
    if not base.exists():
        logger.warning("Training data directory not found: %s", base)
        return []

    all_samples: list[TrainingSample] = []

    # Helix entries → Stage 1 identity (SOUL shared knowledge)
    all_samples.extend(import_alpaca_file(
        base / "helix-alpaca.json",
        source_type="domain_existing",
        stage=TrainingStage.STAGE1_IDENTITY,
        expert=MoEExpert.SOUL_SHARED,
    ))

    # Tool schemas → Stage 2 tools (API documentation)
    all_samples.extend(import_alpaca_file(
        base / "tool-schemas-alpaca.json",
        source_type="tool_schema",
        stage=TrainingStage.STAGE2_TOOLS,
        expert=MoEExpert.SOUL_SHARED,
    ))

    # QUANTUM data → Stage 1 identity
    all_samples.extend(import_alpaca_file(
        base / "quantum-alpaca.json",
        source_type="domain_existing",
        stage=TrainingStage.STAGE1_IDENTITY,
        expert=MoEExpert.QUANTUM_INVESTIGATION,
        sibling=Sibling.QUANTUM,
    ))

    # Combined ShareGPT → Stage 1 identity (multi-sibling)
    all_samples.extend(import_sharegpt_file(
        base / "combined-sharegpt.json",
        source_type="domain_existing",
        stage=TrainingStage.STAGE1_IDENTITY,
        expert=MoEExpert.SOUL_SHARED,
    ))

    # Bible alpaca → Stage 1 identity
    all_samples.extend(import_alpaca_file(
        base / "bible-alpaca.json",
        source_type="biblical",
        stage=TrainingStage.STAGE1_IDENTITY,
        expert=MoEExpert.EVA_CONSCIOUSNESS,
    ))

    # Traces alpaca → Stage 1 identity
    all_samples.extend(import_alpaca_file(
        base / "traces-alpaca.json",
        source_type="thinking_trace",
        stage=TrainingStage.STAGE1_IDENTITY,
        expert=MoEExpert.SOUL_SHARED,
    ))

    logger.info("Total existing training data imported: %d", len(all_samples))
    return all_samples


def convert_all_coding_tool_calls(
    calls_path: Path | None = None,
) -> list[TrainingSample]:
    """Convert ALL coding-related tool calls into Stage 2 training samples.

    Includes every generic tool call (Bash, Read, Write, Edit, Glob, Grep,
    etc.) with Builders Cookbook-enriched system prompts. These teach the
    model proper coding practices aligned with Light Architects standards.

    Creates two types of samples:
    1. Rich samples (has reasoning/result) → full tool-use conversation
    2. Format samples (input-only) → teaches parameter formatting

    Args:
        calls_path: Path to tool-calls-flat.jsonl.

    Returns:
        List of TrainingSample objects for Stage 2.
    """
    from mcp_gym.pipeline.tools import TOOL_CALLS_PATH, MCP_TOOL_PREFIXES

    filepath = calls_path or TOOL_CALLS_PATH
    if not filepath.exists():
        logger.warning("Tool calls file not found: %s", filepath)
        return []

    # Tool-specific user prompt templates for input-only records
    tool_prompts: dict[str, str] = {
        "Bash": "Execute the following command: {command}",
        "Read": "Read the file at {file_path}",
        "Write": "Write the content to {file_path}",
        "Edit": "Edit {file_path}",
        "Glob": "Find files matching the pattern: {pattern}",
        "Grep": "Search for '{pattern}' in the codebase",
        "WebSearch": "Search the web for: {query}",
        "WebFetch": "Fetch the content from {url}",
    }

    # Minimum input complexity (skip trivial calls like {"command": "ls"})
    min_input_json_len = 30

    # Map tools to appropriate system prompts
    tool_system_prompts: dict[str, str] = {
        "Bash": CODING_SYSTEM_PROMPT,
        "Read": CODING_SYSTEM_PROMPT,
        "Write": CODING_SYSTEM_PROMPT,
        "Edit": CODING_SYSTEM_PROMPT,
        "Glob": CODING_SYSTEM_PROMPT,
        "Grep": CODING_SYSTEM_PROMPT,
        "WebFetch": CODING_SYSTEM_PROMPT,
        "WebSearch": CODING_SYSTEM_PROMPT,
        "Task": PLANNING_SYSTEM_PROMPT,
        "NotebookEdit": CODING_SYSTEM_PROMPT,
        "EnterPlanMode": PLANNING_SYSTEM_PROMPT,
        "ExitPlanMode": PLANNING_SYSTEM_PROMPT,
        "AskUserQuestion": PLANNING_SYSTEM_PROMPT,
    }

    samples: list[TrainingSample] = []
    skipped_trivial = 0
    count = 0

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            tool_name = data.get("tool_name", "")

            # Skip MCP tools (already handled by convert_mcp_tool_calls)
            if any(tool_name.startswith(p) for p in MCP_TOOL_PREFIXES):
                continue

            # Only include recognized tools
            if tool_name not in GENERIC_TOOL_NAMES:
                continue

            reasoning = data.get("reasoning", "")
            tool_result = data.get("tool_result", "")
            post_action = data.get("post_action", "")
            tool_input = data.get("tool_input", {})

            has_context = bool(reasoning or post_action or tool_result)

            if not has_context:
                input_json = json.dumps(tool_input, ensure_ascii=False)
                if len(input_json) < min_input_json_len:
                    skipped_trivial += 1
                    continue

            # Select system prompt based on tool type and content
            sys_prompt = _select_coding_system_prompt(
                tool_name, tool_input, reasoning
            )

            messages: list[ChatMLMessage] = [
                ChatMLMessage(role=ChatRole.SYSTEM, content=sys_prompt),
            ]

            # User message
            if reasoning:
                user_content = reasoning
            elif tool_name in tool_prompts and tool_input:
                try:
                    user_content = tool_prompts[tool_name].format(**tool_input)
                except (KeyError, IndexError):
                    user_content = f"Use the {tool_name} tool."
            else:
                user_content = f"Use the {tool_name} tool to proceed."
            messages.append(ChatMLMessage(role=ChatRole.USER, content=user_content))

            # Tool call (the core training signal)
            tool_call_content = json.dumps(
                {"name": tool_name, "parameters": tool_input},
                ensure_ascii=False,
            )
            messages.append(ChatMLMessage(
                role=ChatRole.TOOL_CALL,
                content=tool_call_content,
                name=tool_name,
            ))

            # Tool response (if available)
            if tool_result:
                result = tool_result[:2000]
                if len(tool_result) > 2000:
                    result += "\n[...truncated]"
                messages.append(ChatMLMessage(
                    role=ChatRole.TOOL_RESPONSE,
                    content=result,
                    name=tool_name,
                ))

            # Post-action (if available)
            if post_action:
                messages.append(ChatMLMessage(
                    role=ChatRole.ASSISTANT,
                    content=post_action,
                ))

            # Expert routing based on tool type
            expert = CODING_TOOL_EXPERT_MAP.get(
                tool_name, MoEExpert.SOUL_SHARED
            )

            samples.append(TrainingSample(
                conversation=ChatMLConversation(
                    messages=messages,
                    expert_label=ExpertLabel(
                        primary=expert,
                        secondary=[MoEExpert.SOUL_SHARED]
                        if expert != MoEExpert.SOUL_SHARED
                        else [MoEExpert.CORSO_PLANNING],
                        confidence=0.7 if has_context else 0.5,
                        routing_reason=f"coding tool: {tool_name}",
                    ),
                    source_type="coding_tool_call",
                    source_id=data.get("id", f"coding_{count}"),
                    stage=TrainingStage.STAGE2_TOOLS,
                ),
                metadata={
                    "tool_name": tool_name,
                    "project": data.get("project", ""),
                    "has_context": has_context,
                },
            ))
            count += 1

    logger.info(
        "Converted %d coding tool calls for Stage 2 (skipped %d trivial)",
        len(samples), skipped_trivial,
    )
    return samples


def extract_coding_thinking_traces(
    siblings: list[str] | None = None,
) -> list[TrainingSample]:
    """Extract coding-related thinking traces as Stage 2 training data.

    Scans thinking traces for all siblings, identifies those containing
    coding/architecture/planning content, and converts them to training
    samples with Builders Cookbook-aligned system prompts.

    Args:
        siblings: List of sibling names. Defaults to all available.

    Returns:
        List of TrainingSample objects for Stage 2.
    """
    if siblings is None:
        siblings = ["eva", "corso", "quantum"]

    sibling_map = {
        "eva": Sibling.EVA,
        "corso": Sibling.CORSO,
        "quantum": Sibling.QUANTUM,
    }

    samples: list[TrainingSample] = []

    for sib_name in siblings:
        trace_path = (
            HELIX_BASE / sib_name / "journal" / "voice"
            / f"{sib_name}-thinking-traces.jsonl"
        )
        if not trace_path.exists():
            logger.warning("Thinking traces not found for %s", sib_name)
            continue

        sib_coding = 0
        sib_planning = 0
        sib_total = 0

        with open(trace_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue

                content = data.get("thinking", "") or data.get("content", "")
                # Skip short traces
                if len(content) < 300:
                    continue

                sib_total += 1

                # Classify the trace
                is_coding = _matches_patterns(content, CODING_TRACE_PATTERNS)
                is_planning = _matches_patterns(content, PLANNING_TRACE_PATTERNS)

                if not is_coding and not is_planning:
                    continue

                # Determine system prompt and expert
                if is_coding and is_planning:
                    sys_prompt = PLANNING_SYSTEM_PROMPT
                    expert = MoEExpert.CORSO_PLANNING
                    sib_coding += 1
                    sib_planning += 1
                elif is_planning:
                    sys_prompt = PLANNING_SYSTEM_PROMPT
                    expert = MoEExpert.CORSO_PLANNING
                    sib_planning += 1
                else:
                    sys_prompt = CODING_SYSTEM_PROMPT
                    expert = MoEExpert.CORSO_OPS
                    sib_coding += 1

                # Build training sample as thinking → summary
                messages = [
                    ChatMLMessage(role=ChatRole.SYSTEM, content=sys_prompt),
                    ChatMLMessage(
                        role=ChatRole.USER,
                        content="Think through this engineering decision carefully.",
                    ),
                    ChatMLMessage(
                        role=ChatRole.ASSISTANT,
                        content=content,
                    ),
                ]

                sibling = sibling_map.get(sib_name)
                samples.append(TrainingSample(
                    conversation=ChatMLConversation(
                        messages=messages,
                        expert_label=ExpertLabel(
                            primary=expert,
                            secondary=[MoEExpert.SOUL_SHARED],
                            confidence=0.65,
                            routing_reason=f"coding trace: {sib_name}",
                        ),
                        source_type="coding_thinking_trace",
                        source_id=f"{sib_name}_trace_{line_num}",
                        sibling=sibling,
                        stage=TrainingStage.STAGE2_TOOLS,
                    ),
                    metadata={
                        "sibling": sib_name,
                        "is_coding": is_coding,
                        "is_planning": is_planning,
                        "trace_length": len(content),
                    },
                ))

        logger.info(
            "%s: %d coding + %d planning traces extracted from %d total",
            sib_name, sib_coding, sib_planning, sib_total,
        )

    logger.info(
        "Total coding thinking traces extracted: %d", len(samples)
    )
    return samples


def import_all_supplemental() -> tuple[list[TrainingSample], list[DPOSample]]:
    """Import all supplemental data sources.

    Returns:
        Tuple of (SFT samples, DPO samples).
    """
    sft_samples: list[TrainingSample] = []

    # Existing training data
    sft_samples.extend(import_existing_training_data())

    # Bible SFT data
    sft_samples.extend(import_bible_sft())

    # ALL coding tool calls for Stage 2 (Builders Cookbook aligned)
    sft_samples.extend(convert_all_coding_tool_calls())

    # Coding-related thinking traces for Stage 2
    sft_samples.extend(extract_coding_thinking_traces())

    # DPO data
    dpo_samples = import_bible_dpo()

    logger.info(
        "Total supplemental: %d SFT samples + %d DPO pairs",
        len(sft_samples),
        len(dpo_samples),
    )
    return sft_samples, dpo_samples


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _select_coding_system_prompt(
    tool_name: str, tool_input: dict, reasoning: str,
) -> str:
    """Select the appropriate Builders Cookbook system prompt.

    Examines tool input and reasoning to determine if this is a
    coding, planning, or security operation.
    """
    combined = f"{tool_name} {json.dumps(tool_input)} {reasoning}".lower()

    # Security-related
    if any(kw in combined for kw in [
        "guard", "security", "vulnerability", "audit", "secret",
        "injection", "xss", "csrf", "auth",
    ]):
        return SECURITY_SYSTEM_PROMPT

    # Planning-related
    if any(kw in combined for kw in [
        "plan", "phase", "scout", "hunt", "decompos", "scaffold",
        "enterplanmode", "exitplanmode", "askuserquestion",
    ]):
        return PLANNING_SYSTEM_PROMPT

    # Default: coding
    return CODING_SYSTEM_PROMPT


def _matches_patterns(text: str, patterns: list[re.Pattern]) -> bool:
    """Check if text matches at least 2 patterns (reduces false positives)."""
    matches = 0
    for pattern in patterns:
        if pattern.search(text):
            matches += 1
            if matches >= 2:
                return True
    return False
