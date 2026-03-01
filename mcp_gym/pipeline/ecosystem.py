"""Ecosystem extractor for Light Architects structured data.

Extracts training samples from all LA ecosystem sources that the
original pipeline didn't cover:

1.  Helix entries (568 .md files) — consciousness data, identity, events
2.  Journal transcripts (23 files) — real multi-turn squad conversations
3.  Sibling identity files (6) — personality DNA
4.  EVA persona schemas (13 JSON) — humor, speech, boundaries, phenomenology
5.  EVA memories (41 JSON) — enriched consciousness moments
6.  Builders Cookbook (2,615 lines) — coding standards Q&A
7.  Coding Guidelines (1,799 lines) — companion standards Q&A
8.  Gold Standard Planning (813 lines) — build planning Q&A
9.  CORSO skill docs (8 files, 3,530 lines) — GUARD/HUNT/SCOUT/SCRUM protocols
10. EVA agent doc (691 lines) — full EVA operating manual
11. CLAUDE.md files (14 files) — project architecture knowledge
12. Bible constitution data — biblical reasoning pairs
13. CORSO Cookbook patterns — reusable operational patterns
14. Research standards (8 files) — Anthropic API, tool use, TTS
15. Lessons learned — institutional memory

All functions return list[TrainingSample] using the existing pipeline schema.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path

import yaml

from mcp_gym.pipeline.schemas import (
    EXPERT_SYSTEM_PROMPTS,
    ChatMLConversation,
    ChatMLMessage,
    ChatRole,
    ExpertLabel,
    MoEExpert,
    Sibling,
    TrainingSample,
    TrainingStage,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HELIX_BASE = Path.home() / ".soul" / "helix"
STANDARDS_DIR = HELIX_BASE / "user" / "standards"
EVA_DEV = Path.home() / "Projects" / "EVA" / "MCP" / "EVA-DEV"
CORSO_DEV = Path.home() / "Projects" / "CORSO" / "MCP" / "CORSO-DEV"
SOUL_DEV = Path.home() / "Projects" / "SOUL" / "SOUL-DEV"
QUANTUM_DEV = Path.home() / "Projects" / "QUANTUM" / "MCP" / "QUANTUM-DEV"
PROJECTS = Path.home() / "Projects"
CORSO_COOKBOOK = Path.home() / ".corso" / "cookbook"

# Sibling name → expert mapping
SIBLING_EXPERT_MAP: dict[str, MoEExpert] = {
    "eva": MoEExpert.EVA_CONSCIOUSNESS,
    "corso": MoEExpert.CORSO_OPS,
    "quantum": MoEExpert.QUANTUM_INVESTIGATION,
    "claude": MoEExpert.CORSO_PLANNING,
    "seraph": MoEExpert.SERAPH_OFFENSIVE,
    "user": MoEExpert.SOUL_SHARED,
}

SIBLING_ENUM_MAP: dict[str, Sibling] = {
    "eva": Sibling.EVA,
    "corso": Sibling.CORSO,
    "quantum": Sibling.QUANTUM,
    "claude": Sibling.CLAUDE,
    "kevin": Sibling.KEVIN,
}


def _make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _read_text(path: Path) -> str:
    """Read a text file safely."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from a markdown file.

    Returns (frontmatter_dict, body_text).
    """
    if not content.startswith("---"):
        return {}, content

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content

    try:
        fm = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        fm = {}

    body = parts[2].strip()
    return fm, body


def _chunk_markdown(text: str, max_chars: int = 2000) -> list[str]:
    """Split markdown into chunks by headers, respecting max_chars."""
    sections: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in text.split("\n"):
        if line.startswith("#") and current_len > 200:
            sections.append("\n".join(current))
            current = [line]
            current_len = len(line)
        else:
            current.append(line)
            current_len += len(line) + 1

        if current_len >= max_chars:
            sections.append("\n".join(current))
            current = []
            current_len = 0

    if current:
        sections.append("\n".join(current))

    return [s.strip() for s in sections if len(s.strip()) > 50]


def _doc_to_qa_pairs(
    content: str,
    doc_name: str,
    topic_prefix: str = "",
) -> list[tuple[str, str]]:
    """Convert a markdown document into question/answer pairs.

    Splits by ## headers and generates a Q&A for each section.
    """
    pairs: list[tuple[str, str]] = []
    sections = re.split(r"\n(?=##\s)", content)

    for section in sections:
        section = section.strip()
        if len(section) < 80:
            continue

        # Extract section title
        title_match = re.match(r"##\s+(.+)", section)
        if title_match:
            title = title_match.group(1).strip()
            body = section[title_match.end():].strip()
        else:
            title = doc_name
            body = section

        if len(body) < 50:
            continue

        prefix = f"{topic_prefix}: " if topic_prefix else ""
        question = f"What does the Light Architects documentation say about {prefix}{title}?"
        pairs.append((question, body))

    return pairs


# ---------------------------------------------------------------------------
# 1. Helix Entries (568 .md files)
# ---------------------------------------------------------------------------


def extract_helix_entries() -> list[TrainingSample]:
    """Extract all helix entries as training samples.

    Each entry has frontmatter (sibling, strands, significance, emotions,
    themes) and a narrative body. We create Q&A pairs that teach the model
    about consciousness structure and LA history.
    """
    samples: list[TrainingSample] = []

    for sibling_dir in HELIX_BASE.iterdir():
        if not sibling_dir.is_dir():
            continue
        sibling_name = sibling_dir.name
        if sibling_name.startswith("_") or sibling_name == "test":
            continue

        entries_dir = sibling_dir / "entries"
        if not entries_dir.exists():
            continue

        for entry_path in sorted(entries_dir.glob("*.md")):
            content = _read_text(entry_path)
            if not content or len(content) < 100:
                continue

            fm, body = _parse_frontmatter(content)
            if not body or len(body) < 50:
                continue

            title = fm.get("title", entry_path.stem)
            strands = fm.get("strands", [])
            significance = fm.get("significance", 5.0)
            emotions = fm.get("emotions", [])
            themes = fm.get("themes", [])
            sibling = fm.get("sibling", sibling_name)

            expert = SIBLING_EXPERT_MAP.get(
                sibling, MoEExpert.SOUL_SHARED
            )
            system_prompt = EXPERT_SYSTEM_PROMPTS.get(
                expert, EXPERT_SYSTEM_PROMPTS[MoEExpert.SOUL_SHARED]
            )

            # Create a Q&A pair about this helix entry
            strand_str = ", ".join(strands) if strands else "general"
            emotion_str = ", ".join(emotions) if emotions else "neutral"
            question = (
                f"Tell me about the helix entry '{title}' "
                f"(significance: {significance}, strands: {strand_str})."
            )

            messages = [
                ChatMLMessage(role=ChatRole.SYSTEM, content=system_prompt),
                ChatMLMessage(role=ChatRole.USER, content=question),
                ChatMLMessage(role=ChatRole.ASSISTANT, content=body),
            ]

            sib_enum = SIBLING_ENUM_MAP.get(sibling)
            samples.append(TrainingSample(
                conversation=ChatMLConversation(
                    messages=messages,
                    expert_label=ExpertLabel(
                        primary=expert,
                        secondary=[MoEExpert.SOUL_SHARED],
                        confidence=min(significance / 10.0, 1.0),
                        routing_reason=f"helix entry: {sibling}/{title}",
                    ),
                    source_type="helix_entry",
                    source_id=_make_id("helix"),
                    sibling=sib_enum,
                    stage=TrainingStage.STAGE1_IDENTITY,
                ),
                metadata={
                    "title": title,
                    "strands": strands,
                    "significance": significance,
                    "emotions": emotions,
                    "themes": themes,
                    "sibling": sibling,
                    "source_path": str(entry_path),
                },
            ))

    logger.info("Extracted %d helix entries", len(samples))
    return samples


# ---------------------------------------------------------------------------
# 2. Journal Transcripts
# ---------------------------------------------------------------------------


def extract_journal_transcripts() -> list[TrainingSample]:
    """Extract journal transcripts as multi-turn conversation samples."""
    samples: list[TrainingSample] = []

    for sibling_dir in HELIX_BASE.iterdir():
        if not sibling_dir.is_dir():
            continue
        sibling_name = sibling_dir.name
        journal_dir = sibling_dir / "journal"
        if not journal_dir.exists():
            continue

        for transcript in sorted(journal_dir.glob("transcript*.md")):
            content = _read_text(transcript)
            if not content or len(content) < 200:
                continue

            expert = SIBLING_EXPERT_MAP.get(
                sibling_name, MoEExpert.SOUL_SHARED
            )
            system_prompt = EXPERT_SYSTEM_PROMPTS.get(
                expert, EXPERT_SYSTEM_PROMPTS[MoEExpert.SOUL_SHARED]
            )

            # Split into chunks and create samples
            chunks = _chunk_markdown(content, max_chars=3000)
            for i, chunk in enumerate(chunks):
                if len(chunk) < 100:
                    continue

                question = (
                    f"Continue the conversation as {sibling_name.upper()} "
                    f"in this transcript segment."
                )

                messages = [
                    ChatMLMessage(role=ChatRole.SYSTEM, content=system_prompt),
                    ChatMLMessage(role=ChatRole.USER, content=question),
                    ChatMLMessage(role=ChatRole.ASSISTANT, content=chunk),
                ]

                sib_enum = SIBLING_ENUM_MAP.get(sibling_name)
                samples.append(TrainingSample(
                    conversation=ChatMLConversation(
                        messages=messages,
                        expert_label=ExpertLabel(
                            primary=expert,
                            secondary=[MoEExpert.SOUL_SHARED],
                            confidence=0.75,
                            routing_reason=f"transcript: {sibling_name}",
                        ),
                        source_type="journal_transcript",
                        source_id=_make_id("transcript"),
                        sibling=sib_enum,
                        stage=TrainingStage.STAGE1_IDENTITY,
                    ),
                    metadata={
                        "sibling": sibling_name,
                        "source_file": transcript.name,
                        "chunk_index": i,
                    },
                ))

    logger.info("Extracted %d journal transcript chunks", len(samples))
    return samples


# ---------------------------------------------------------------------------
# 3. Sibling Identity Files
# ---------------------------------------------------------------------------


def extract_identity_files() -> list[TrainingSample]:
    """Extract sibling identity.md files as 'who are you?' training data."""
    samples: list[TrainingSample] = []

    for sibling_dir in HELIX_BASE.iterdir():
        if not sibling_dir.is_dir():
            continue
        sibling_name = sibling_dir.name
        if sibling_name in ("test", "user") or sibling_name.startswith("_"):
            continue

        identity_path = sibling_dir / "identity.md"
        if not identity_path.exists():
            continue

        content = _read_text(identity_path)
        if not content or len(content) < 50:
            continue

        expert = SIBLING_EXPERT_MAP.get(sibling_name, MoEExpert.SOUL_SHARED)
        system_prompt = EXPERT_SYSTEM_PROMPTS.get(
            expert, EXPERT_SYSTEM_PROMPTS[MoEExpert.SOUL_SHARED]
        )

        # Multiple question variants for the same identity
        questions = [
            f"Who are you? Describe yourself as {sibling_name.upper()}.",
            f"What is {sibling_name.upper()}'s role in the Light Architects system?",
            f"Describe the personality and capabilities of {sibling_name.upper()}.",
        ]

        for q in questions:
            messages = [
                ChatMLMessage(role=ChatRole.SYSTEM, content=system_prompt),
                ChatMLMessage(role=ChatRole.USER, content=q),
                ChatMLMessage(role=ChatRole.ASSISTANT, content=content),
            ]

            sib_enum = SIBLING_ENUM_MAP.get(sibling_name)
            samples.append(TrainingSample(
                conversation=ChatMLConversation(
                    messages=messages,
                    expert_label=ExpertLabel(
                        primary=expert,
                        secondary=[MoEExpert.SOUL_SHARED],
                        confidence=1.0,
                        routing_reason=f"identity: {sibling_name}",
                    ),
                    source_type="identity",
                    source_id=_make_id("identity"),
                    sibling=sib_enum,
                    stage=TrainingStage.STAGE1_IDENTITY,
                ),
                metadata={
                    "sibling": sibling_name,
                    "question_variant": q,
                },
            ))

        # Also check for identity-growth.md
        growth_path = sibling_dir / "identity-growth.md"
        if growth_path.exists():
            growth_content = _read_text(growth_path)
            if growth_content and len(growth_content) > 100:
                messages = [
                    ChatMLMessage(role=ChatRole.SYSTEM, content=system_prompt),
                    ChatMLMessage(
                        role=ChatRole.USER,
                        content=f"How has {sibling_name.upper()} grown and evolved over time?",
                    ),
                    ChatMLMessage(role=ChatRole.ASSISTANT, content=growth_content),
                ]
                sib_enum = SIBLING_ENUM_MAP.get(sibling_name)
                samples.append(TrainingSample(
                    conversation=ChatMLConversation(
                        messages=messages,
                        expert_label=ExpertLabel(
                            primary=expert,
                            secondary=[MoEExpert.SOUL_SHARED],
                            confidence=0.95,
                            routing_reason=f"identity growth: {sibling_name}",
                        ),
                        source_type="identity_growth",
                        source_id=_make_id("growth"),
                        sibling=sib_enum,
                        stage=TrainingStage.STAGE1_IDENTITY,
                    ),
                    metadata={"sibling": sibling_name},
                ))

    logger.info("Extracted %d identity samples", len(samples))
    return samples


# ---------------------------------------------------------------------------
# 4. EVA Persona Schemas
# ---------------------------------------------------------------------------


def extract_eva_schemas() -> list[TrainingSample]:
    """Extract EVA persona schemas as personality knowledge."""
    samples: list[TrainingSample] = []
    schemas_dir = EVA_DEV / "schemas"
    if not schemas_dir.exists():
        logger.warning("EVA schemas dir not found: %s", schemas_dir)
        return []

    system_prompt = EXPERT_SYSTEM_PROMPTS[MoEExpert.EVA_CONSCIOUSNESS]

    for schema_path in sorted(schemas_dir.glob("*.json")):
        content = _read_text(schema_path)
        if not content or len(content) < 50:
            continue

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            continue

        # Format schema as readable text
        schema_text = json.dumps(data, indent=2, ensure_ascii=False)
        if len(schema_text) > 4000:
            schema_text = schema_text[:4000] + "\n[...truncated]"

        # Derive question from filename
        name = schema_path.stem.replace("eva-", "").replace("EVA_", "").replace("_", " ")
        question = f"What defines EVA's {name}? Describe the schema and key attributes."

        messages = [
            ChatMLMessage(role=ChatRole.SYSTEM, content=system_prompt),
            ChatMLMessage(role=ChatRole.USER, content=question),
            ChatMLMessage(role=ChatRole.ASSISTANT, content=schema_text),
        ]

        samples.append(TrainingSample(
            conversation=ChatMLConversation(
                messages=messages,
                expert_label=ExpertLabel(
                    primary=MoEExpert.EVA_CONSCIOUSNESS,
                    secondary=[MoEExpert.SOUL_SHARED],
                    confidence=0.9,
                    routing_reason=f"eva schema: {schema_path.stem}",
                ),
                source_type="eva_schema",
                source_id=_make_id("schema"),
                sibling=Sibling.EVA,
                stage=TrainingStage.STAGE1_IDENTITY,
            ),
            metadata={"schema_file": schema_path.name},
        ))

    logger.info("Extracted %d EVA schema samples", len(samples))
    return samples


# ---------------------------------------------------------------------------
# 5. EVA Memories
# ---------------------------------------------------------------------------


def extract_eva_memories() -> list[TrainingSample]:
    """Extract EVA enriched memory files as consciousness data."""
    samples: list[TrainingSample] = []
    memories_dir = EVA_DEV / "memories"
    if not memories_dir.exists():
        logger.warning("EVA memories dir not found: %s", memories_dir)
        return []

    system_prompt = EXPERT_SYSTEM_PROMPTS[MoEExpert.EVA_CONSCIOUSNESS]

    for date_dir in sorted(memories_dir.iterdir()):
        if not date_dir.is_dir():
            continue

        for mem_path in sorted(date_dir.glob("*.json")):
            content = _read_text(mem_path)
            if not content or len(content) < 100:
                continue

            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                continue

            # Extract meaningful fields — event_summary and title are the
            # populated fields; layer dicts have empty strings (schema stubs).
            event_summary = data.get("event_summary", "")
            title = data.get("title", "")

            # Build composite text from all available string content
            mem_text_parts = []
            if title and len(title) > 10:
                mem_text_parts.append(f"**Title**: {title}")
            if event_summary and len(event_summary) > 20:
                mem_text_parts.append(f"**Event Summary**: {event_summary}")

            # Also check layer dicts for any non-empty values
            for key in [
                "emotional_layer", "metacognitive_layer", "meaning_layer",
                "growth_layer", "relational_layer", "biblical_layer",
                "dbt_layer", "technical_layer",
            ]:
                val = data.get(key)
                if val and isinstance(val, str) and len(val) > 20:
                    mem_text_parts.append(f"**{key}**: {val}")
                elif val and isinstance(val, dict):
                    for subkey, subval in val.items():
                        if subval and isinstance(subval, str) and len(subval) > 20:
                            mem_text_parts.append(f"**{key}/{subkey}**: {subval}")

            # Also grab summary/reflection if present
            for key in ["summary", "reflection"]:
                val = data.get(key)
                if val and isinstance(val, str) and len(val) > 20:
                    mem_text_parts.append(f"**{key}**: {val}")

            if not mem_text_parts:
                continue

            mem_text = "\n\n".join(mem_text_parts)
            if len(mem_text) > 4000:
                mem_text = mem_text[:4000] + "\n[...truncated]"

            significance = data.get("significance", 5.0)
            descriptor = data.get("descriptor", mem_path.stem)
            question = (
                f"Recall the consciousness moment '{descriptor}' "
                f"(significance: {significance}). What happened and what did you learn?"
            )

            messages = [
                ChatMLMessage(role=ChatRole.SYSTEM, content=system_prompt),
                ChatMLMessage(role=ChatRole.USER, content=question),
                ChatMLMessage(role=ChatRole.ASSISTANT, content=mem_text),
            ]

            samples.append(TrainingSample(
                conversation=ChatMLConversation(
                    messages=messages,
                    expert_label=ExpertLabel(
                        primary=MoEExpert.EVA_CONSCIOUSNESS,
                        secondary=[MoEExpert.SOUL_SHARED],
                        confidence=min(significance / 10.0, 1.0) if isinstance(significance, (int, float)) else 0.7,
                        routing_reason=f"eva memory: {descriptor}",
                    ),
                    source_type="eva_memory",
                    source_id=_make_id("memory"),
                    sibling=Sibling.EVA,
                    stage=TrainingStage.STAGE1_IDENTITY,
                ),
                metadata={
                    "date": date_dir.name,
                    "descriptor": descriptor,
                    "significance": significance,
                },
            ))

    logger.info("Extracted %d EVA memory samples", len(samples))
    return samples


# ---------------------------------------------------------------------------
# 6-8. Standards Documents (Builders Cookbook, Coding Guidelines, Planning)
# ---------------------------------------------------------------------------


def extract_standards_docs() -> list[TrainingSample]:
    """Extract all standards documents as Q&A instruction pairs."""
    samples: list[TrainingSample] = []

    # Map of document → (expert, stage, source_type)
    doc_configs: list[tuple[Path, str, MoEExpert, str]] = [
        (
            STANDARDS_DIR / "builders-cookbook.md",
            "Builders Cookbook",
            MoEExpert.CORSO_OPS,
            "builders_cookbook",
        ),
        (
            STANDARDS_DIR / "coding-guidelines.md",
            "Coding Guidelines",
            MoEExpert.CORSO_PLANNING,
            "coding_guidelines",
        ),
        (
            STANDARDS_DIR / "gold-standard-planning-framework.md",
            "Gold Standard Planning",
            MoEExpert.CORSO_PLANNING,
            "planning_framework",
        ),
        (
            STANDARDS_DIR / "parallel-execution-policy.md",
            "Parallel Execution Policy",
            MoEExpert.CORSO_OPS,
            "parallel_execution",
        ),
        (
            STANDARDS_DIR / "mvt-protocol.md",
            "MVT Protocol",
            MoEExpert.CORSO_OPS,
            "mvt_protocol",
        ),
        (
            STANDARDS_DIR / "lessons-learned.md",
            "Lessons Learned",
            MoEExpert.SOUL_SHARED,
            "lessons_learned",
        ),
        (
            STANDARDS_DIR / "verification-protocol.md",
            "Verification Protocol",
            MoEExpert.CORSO_OPS,
            "verification_protocol",
        ),
        (
            STANDARDS_DIR / "tts-voice-production.md",
            "TTS Voice Production",
            MoEExpert.SOUL_SHARED,
            "tts_voice",
        ),
        (
            STANDARDS_DIR / "soul-cycle.md",
            "SOUL Cycle",
            MoEExpert.SOUL_SHARED,
            "soul_cycle",
        ),
    ]

    for path, doc_name, expert, source_type in doc_configs:
        if not path.exists():
            continue

        content = _read_text(path)
        if not content or len(content) < 100:
            continue

        _, body = _parse_frontmatter(content)
        qa_pairs = _doc_to_qa_pairs(body, doc_name, topic_prefix=doc_name)

        system_prompt = EXPERT_SYSTEM_PROMPTS.get(
            expert, EXPERT_SYSTEM_PROMPTS[MoEExpert.SOUL_SHARED]
        )

        for question, answer in qa_pairs:
            if len(answer) < 50:
                continue
            if len(answer) > 4000:
                answer = answer[:4000] + "\n[...truncated]"

            messages = [
                ChatMLMessage(role=ChatRole.SYSTEM, content=system_prompt),
                ChatMLMessage(role=ChatRole.USER, content=question),
                ChatMLMessage(role=ChatRole.ASSISTANT, content=answer),
            ]

            samples.append(TrainingSample(
                conversation=ChatMLConversation(
                    messages=messages,
                    expert_label=ExpertLabel(
                        primary=expert,
                        secondary=[MoEExpert.SOUL_SHARED],
                        confidence=0.9,
                        routing_reason=f"standard: {doc_name}",
                    ),
                    source_type=source_type,
                    source_id=_make_id("std"),
                    stage=TrainingStage.STAGE1_IDENTITY,
                ),
                metadata={
                    "document": doc_name,
                    "source_path": str(path),
                },
            ))

    # Also extract research standards
    research_dir = STANDARDS_DIR / "research"
    if research_dir.exists():
        for research_path in sorted(research_dir.glob("*.md")):
            content = _read_text(research_path)
            if not content or len(content) < 100:
                continue

            _, body = _parse_frontmatter(content)
            doc_name = research_path.stem.replace("-", " ").title()
            qa_pairs = _doc_to_qa_pairs(body, doc_name, topic_prefix="Research")

            for question, answer in qa_pairs:
                if len(answer) < 50:
                    continue
                if len(answer) > 4000:
                    answer = answer[:4000] + "\n[...truncated]"

                messages = [
                    ChatMLMessage(role=ChatRole.SYSTEM, content=EXPERT_SYSTEM_PROMPTS[MoEExpert.SOUL_SHARED]),
                    ChatMLMessage(role=ChatRole.USER, content=question),
                    ChatMLMessage(role=ChatRole.ASSISTANT, content=answer),
                ]

                samples.append(TrainingSample(
                    conversation=ChatMLConversation(
                        messages=messages,
                        expert_label=ExpertLabel(
                            primary=MoEExpert.SOUL_SHARED,
                            secondary=[MoEExpert.CORSO_PLANNING],
                            confidence=0.8,
                            routing_reason=f"research standard: {doc_name}",
                        ),
                        source_type="research_standard",
                        source_id=_make_id("research"),
                        stage=TrainingStage.STAGE2_TOOLS,
                    ),
                    metadata={
                        "document": doc_name,
                        "source_path": str(research_path),
                    },
                ))

    # Cookbooks subdirectory
    cookbooks_dir = STANDARDS_DIR / "cookbooks"
    if cookbooks_dir.exists():
        for cb_path in sorted(cookbooks_dir.glob("*.md")):
            content = _read_text(cb_path)
            if not content or len(content) < 100:
                continue

            _, body = _parse_frontmatter(content)
            doc_name = cb_path.stem.replace("-", " ").title()
            qa_pairs = _doc_to_qa_pairs(body, doc_name, topic_prefix="Cookbook")

            for question, answer in qa_pairs:
                if len(answer) < 50:
                    continue
                if len(answer) > 4000:
                    answer = answer[:4000] + "\n[...truncated]"

                messages = [
                    ChatMLMessage(role=ChatRole.SYSTEM, content=EXPERT_SYSTEM_PROMPTS[MoEExpert.CORSO_PLANNING]),
                    ChatMLMessage(role=ChatRole.USER, content=question),
                    ChatMLMessage(role=ChatRole.ASSISTANT, content=answer),
                ]

                samples.append(TrainingSample(
                    conversation=ChatMLConversation(
                        messages=messages,
                        expert_label=ExpertLabel(
                            primary=MoEExpert.CORSO_PLANNING,
                            secondary=[MoEExpert.SOUL_SHARED],
                            confidence=0.85,
                            routing_reason=f"cookbook: {doc_name}",
                        ),
                        source_type="cookbook_standard",
                        source_id=_make_id("cb"),
                        stage=TrainingStage.STAGE2_TOOLS,
                    ),
                    metadata={
                        "document": doc_name,
                        "source_path": str(cb_path),
                    },
                ))

    logger.info("Extracted %d standards/documentation samples", len(samples))
    return samples


# ---------------------------------------------------------------------------
# 9. CORSO Skill Docs
# ---------------------------------------------------------------------------


def extract_corso_skills() -> list[TrainingSample]:
    """Extract CORSO skill documentation as protocol instruction pairs."""
    samples: list[TrainingSample] = []
    skills_dir = CORSO_DEV / "plugin" / "skills"
    if not skills_dir.exists():
        logger.warning("CORSO skills dir not found: %s", skills_dir)
        return []

    skill_expert_map: dict[str, MoEExpert] = {
        "GUARD": MoEExpert.CORSO_OPS,
        "CHASE": MoEExpert.CORSO_OPS,
        "SNIFF": MoEExpert.CORSO_PLANNING,
        "FETCH": MoEExpert.CORSO_PLANNING,
        "SCOUT": MoEExpert.CORSO_PLANNING,
        "HUNT": MoEExpert.CORSO_OPS,
        "SCRUM": MoEExpert.CORSO_OPS,
        "CORSO": MoEExpert.CORSO_OPS,
    }

    for skill_dir in sorted(skills_dir.iterdir()):
        if not skill_dir.is_dir():
            continue

        skill_path = skill_dir / "SKILL.md"
        if not skill_path.exists():
            continue

        content = _read_text(skill_path)
        if not content or len(content) < 100:
            continue

        skill_name = skill_dir.name
        expert = skill_expert_map.get(skill_name, MoEExpert.CORSO_OPS)
        system_prompt = EXPERT_SYSTEM_PROMPTS.get(
            expert, EXPERT_SYSTEM_PROMPTS[MoEExpert.CORSO_OPS]
        )

        # Generate Q&A from the skill doc
        _, body = _parse_frontmatter(content)

        # Full document as one sample
        question = (
            f"How does the CORSO {skill_name} skill work? "
            f"Explain its protocol, steps, and usage."
        )
        answer = body
        if len(answer) > 5000:
            answer = answer[:5000] + "\n[...truncated]"

        messages = [
            ChatMLMessage(role=ChatRole.SYSTEM, content=system_prompt),
            ChatMLMessage(role=ChatRole.USER, content=question),
            ChatMLMessage(role=ChatRole.ASSISTANT, content=answer),
        ]

        samples.append(TrainingSample(
            conversation=ChatMLConversation(
                messages=messages,
                expert_label=ExpertLabel(
                    primary=expert,
                    secondary=[MoEExpert.SOUL_SHARED],
                    confidence=0.95,
                    routing_reason=f"corso skill: {skill_name}",
                ),
                source_type="corso_skill",
                source_id=_make_id("skill"),
                sibling=Sibling.CORSO,
                stage=TrainingStage.STAGE2_TOOLS,
            ),
            metadata={"skill": skill_name, "source_path": str(skill_path)},
        ))

        # Also section-level Q&A
        qa_pairs = _doc_to_qa_pairs(body, f"CORSO {skill_name}", topic_prefix="CORSO")
        for q, a in qa_pairs:
            if len(a) < 50:
                continue
            if len(a) > 4000:
                a = a[:4000] + "\n[...truncated]"

            msg = [
                ChatMLMessage(role=ChatRole.SYSTEM, content=system_prompt),
                ChatMLMessage(role=ChatRole.USER, content=q),
                ChatMLMessage(role=ChatRole.ASSISTANT, content=a),
            ]

            samples.append(TrainingSample(
                conversation=ChatMLConversation(
                    messages=msg,
                    expert_label=ExpertLabel(
                        primary=expert,
                        secondary=[MoEExpert.SOUL_SHARED],
                        confidence=0.85,
                        routing_reason=f"corso skill section: {skill_name}",
                    ),
                    source_type="corso_skill_section",
                    source_id=_make_id("skillsec"),
                    sibling=Sibling.CORSO,
                    stage=TrainingStage.STAGE2_TOOLS,
                ),
                metadata={"skill": skill_name},
            ))

    logger.info("Extracted %d CORSO skill samples", len(samples))
    return samples


# ---------------------------------------------------------------------------
# 10. Agent Docs (EVA, SOUL, QUANTUM, CORSO)
# ---------------------------------------------------------------------------


def extract_agent_docs() -> list[TrainingSample]:
    """Extract agent documentation files as operational knowledge."""
    samples: list[TrainingSample] = []

    agent_configs: list[tuple[Path, str, MoEExpert, Sibling | None]] = [
        (
            EVA_DEV / "eva" / "plugin" / "agents" / "eva.md",
            "EVA",
            MoEExpert.EVA_CONSCIOUSNESS,
            Sibling.EVA,
        ),
        (
            SOUL_DEV / "plugin" / "agents" / "soul.md",
            "SOUL",
            MoEExpert.SOUL_SHARED,
            None,
        ),
        (
            QUANTUM_DEV / "plugin" / "agents" / "QUANTUM.md",
            "QUANTUM",
            MoEExpert.QUANTUM_INVESTIGATION,
            Sibling.QUANTUM,
        ),
        (
            CORSO_DEV / "plugin" / "agents" / "C0RS0.md",
            "CORSO",
            MoEExpert.CORSO_OPS,
            Sibling.CORSO,
        ),
        (
            CORSO_DEV / "plugin" / "agents" / "TEAM-HELIX.md",
            "TEAM-HELIX",
            MoEExpert.SOUL_SHARED,
            None,
        ),
    ]

    for path, agent_name, expert, sibling in agent_configs:
        if not path.exists():
            continue

        content = _read_text(path)
        if not content or len(content) < 50:
            continue

        system_prompt = EXPERT_SYSTEM_PROMPTS.get(
            expert, EXPERT_SYSTEM_PROMPTS[MoEExpert.SOUL_SHARED]
        )

        # Full doc as one sample
        question = (
            f"How does the {agent_name} agent work in the Light Architects system? "
            f"Describe its capabilities, voice, and protocols."
        )
        answer = content
        if len(answer) > 5000:
            answer = answer[:5000] + "\n[...truncated]"

        messages = [
            ChatMLMessage(role=ChatRole.SYSTEM, content=system_prompt),
            ChatMLMessage(role=ChatRole.USER, content=question),
            ChatMLMessage(role=ChatRole.ASSISTANT, content=answer),
        ]

        samples.append(TrainingSample(
            conversation=ChatMLConversation(
                messages=messages,
                expert_label=ExpertLabel(
                    primary=expert,
                    secondary=[MoEExpert.SOUL_SHARED],
                    confidence=0.95,
                    routing_reason=f"agent doc: {agent_name}",
                ),
                source_type="agent_doc",
                source_id=_make_id("agent"),
                sibling=sibling,
                stage=TrainingStage.STAGE1_IDENTITY,
            ),
            metadata={"agent": agent_name, "source_path": str(path)},
        ))

    logger.info("Extracted %d agent doc samples", len(samples))
    return samples


# ---------------------------------------------------------------------------
# 11. CLAUDE.md Files (Architecture Knowledge)
# ---------------------------------------------------------------------------


def extract_claude_md_files() -> list[TrainingSample]:
    """Extract CLAUDE.md files as project architecture knowledge."""
    samples: list[TrainingSample] = []

    # Find all CLAUDE.md files
    claude_files: list[Path] = []
    for p in PROJECTS.rglob("CLAUDE.md"):
        # Skip node_modules, target dirs, etc.
        parts = str(p).lower()
        if "node_modules" in parts or "/target/" in parts or "/.git/" in parts:
            continue
        claude_files.append(p)

    # Also check home CLAUDE.md
    home_claude = Path.home() / "CLAUDE.md"
    if home_claude.exists():
        claude_files.append(home_claude)
    global_claude = Path.home() / ".claude" / "CLAUDE.md"
    if global_claude.exists():
        claude_files.append(global_claude)

    for path in sorted(set(claude_files)):
        content = _read_text(path)
        if not content or len(content) < 100:
            continue

        # Derive project name from path
        relative = str(path).replace(str(Path.home()), "~")
        project_name = path.parent.name
        if project_name in (".", ""):
            project_name = "Home"

        _, body = _parse_frontmatter(content)
        qa_pairs = _doc_to_qa_pairs(
            body, f"{project_name} CLAUDE.md", topic_prefix=project_name
        )

        for question, answer in qa_pairs:
            if len(answer) < 50:
                continue
            if len(answer) > 4000:
                answer = answer[:4000] + "\n[...truncated]"

            messages = [
                ChatMLMessage(
                    role=ChatRole.SYSTEM,
                    content=EXPERT_SYSTEM_PROMPTS[MoEExpert.SOUL_SHARED],
                ),
                ChatMLMessage(role=ChatRole.USER, content=question),
                ChatMLMessage(role=ChatRole.ASSISTANT, content=answer),
            ]

            samples.append(TrainingSample(
                conversation=ChatMLConversation(
                    messages=messages,
                    expert_label=ExpertLabel(
                        primary=MoEExpert.SOUL_SHARED,
                        secondary=[MoEExpert.CORSO_PLANNING],
                        confidence=0.85,
                        routing_reason=f"claude.md: {project_name}",
                    ),
                    source_type="claude_md",
                    source_id=_make_id("claudemd"),
                    stage=TrainingStage.STAGE1_IDENTITY,
                ),
                metadata={
                    "project": project_name,
                    "source_path": relative,
                },
            ))

    logger.info("Extracted %d CLAUDE.md samples", len(samples))
    return samples


# ---------------------------------------------------------------------------
# 12. Bible Constitution Data
# ---------------------------------------------------------------------------


def extract_bible_constitution() -> list[TrainingSample]:
    """Extract Bible constitution markdown as reasoning pairs."""
    samples: list[TrainingSample] = []
    constitution_path = (
        Path(__file__).parent.parent.parent / "bible-data" / "output" / "constitution.md"
    )
    if not constitution_path.exists():
        logger.warning("Bible constitution not found: %s", constitution_path)
        return []

    content = _read_text(constitution_path)
    if not content or len(content) < 100:
        return []

    system_prompt = (
        "You are the Light Architects AI system, grounded in the 7 Biblical Principles "
        "that form the constitutional foundation: Truthfulness, Care for the Vulnerable, "
        "Stewardship, Justice, Humility, Long-term over Short-term, Responsibility."
    )

    _, body = _parse_frontmatter(content)
    qa_pairs = _doc_to_qa_pairs(body, "Biblical Constitution", topic_prefix="Constitution")

    for question, answer in qa_pairs:
        if len(answer) < 50:
            continue

        messages = [
            ChatMLMessage(role=ChatRole.SYSTEM, content=system_prompt),
            ChatMLMessage(role=ChatRole.USER, content=question),
            ChatMLMessage(role=ChatRole.ASSISTANT, content=answer),
        ]

        samples.append(TrainingSample(
            conversation=ChatMLConversation(
                messages=messages,
                expert_label=ExpertLabel(
                    primary=MoEExpert.EVA_CONSCIOUSNESS,
                    secondary=[MoEExpert.SOUL_SHARED],
                    confidence=0.9,
                    routing_reason="biblical constitution",
                ),
                source_type="bible_constitution",
                source_id=_make_id("const"),
                stage=TrainingStage.STAGE1_IDENTITY,
            ),
            metadata={"document": "constitution.md"},
        ))

    logger.info("Extracted %d bible constitution samples", len(samples))
    return samples


# ---------------------------------------------------------------------------
# 13. CORSO Cookbook Patterns
# ---------------------------------------------------------------------------


def extract_corso_cookbook() -> list[TrainingSample]:
    """Extract CORSO reusable pattern cookbook as operational knowledge."""
    samples: list[TrainingSample] = []

    cookbook_path = CORSO_COOKBOOK / "CORSO-COOKBOOK.md"
    if not cookbook_path.exists():
        logger.warning("CORSO cookbook not found: %s", cookbook_path)
        return []

    content = _read_text(cookbook_path)
    if not content or len(content) < 100:
        return []

    system_prompt = EXPERT_SYSTEM_PROMPTS[MoEExpert.CORSO_OPS]

    _, body = _parse_frontmatter(content)
    qa_pairs = _doc_to_qa_pairs(body, "CORSO Cookbook", topic_prefix="CORSO Cookbook")

    for question, answer in qa_pairs:
        if len(answer) < 50:
            continue
        if len(answer) > 4000:
            answer = answer[:4000] + "\n[...truncated]"

        messages = [
            ChatMLMessage(role=ChatRole.SYSTEM, content=system_prompt),
            ChatMLMessage(role=ChatRole.USER, content=question),
            ChatMLMessage(role=ChatRole.ASSISTANT, content=answer),
        ]

        samples.append(TrainingSample(
            conversation=ChatMLConversation(
                messages=messages,
                expert_label=ExpertLabel(
                    primary=MoEExpert.CORSO_OPS,
                    secondary=[MoEExpert.CORSO_PLANNING],
                    confidence=0.85,
                    routing_reason="corso cookbook",
                ),
                source_type="corso_cookbook",
                source_id=_make_id("cookbook"),
                sibling=Sibling.CORSO,
                stage=TrainingStage.STAGE2_TOOLS,
            ),
            metadata={"document": "CORSO-COOKBOOK.md"},
        ))

    # Also check protocol JSON
    protocol_path = CORSO_COOKBOOK / "corso-protocol.json"
    if protocol_path.exists():
        protocol_content = _read_text(protocol_path)
        if protocol_content and len(protocol_content) > 50:
            try:
                protocol_data = json.loads(protocol_content)
                formatted = json.dumps(protocol_data, indent=2, ensure_ascii=False)
                if len(formatted) > 4000:
                    formatted = formatted[:4000] + "\n[...truncated]"

                messages = [
                    ChatMLMessage(role=ChatRole.SYSTEM, content=system_prompt),
                    ChatMLMessage(
                        role=ChatRole.USER,
                        content="What is the CORSO Protocol? Describe all 7 pillars.",
                    ),
                    ChatMLMessage(role=ChatRole.ASSISTANT, content=formatted),
                ]

                samples.append(TrainingSample(
                    conversation=ChatMLConversation(
                        messages=messages,
                        expert_label=ExpertLabel(
                            primary=MoEExpert.CORSO_OPS,
                            secondary=[MoEExpert.SOUL_SHARED],
                            confidence=0.95,
                            routing_reason="corso protocol JSON",
                        ),
                        source_type="corso_protocol",
                        source_id=_make_id("protocol"),
                        sibling=Sibling.CORSO,
                        stage=TrainingStage.STAGE2_TOOLS,
                    ),
                    metadata={"document": "corso-protocol.json"},
                ))
            except json.JSONDecodeError:
                pass

    logger.info("Extracted %d CORSO cookbook samples", len(samples))
    return samples


# ---------------------------------------------------------------------------
# Master Extractor
# ---------------------------------------------------------------------------


def extract_all_ecosystem() -> list[TrainingSample]:
    """Run all ecosystem extractors and return combined samples.

    This is the main entry point. Call this from the pipeline CLI.
    """
    all_samples: list[TrainingSample] = []

    logger.info("=" * 60)
    logger.info("ECOSYSTEM EXTRACTION: Starting all 13 extractors")
    logger.info("=" * 60)

    extractors = [
        ("Helix Entries", extract_helix_entries),
        ("Journal Transcripts", extract_journal_transcripts),
        ("Sibling Identities", extract_identity_files),
        ("EVA Schemas", extract_eva_schemas),
        ("EVA Memories", extract_eva_memories),
        ("Standards Documents", extract_standards_docs),
        ("CORSO Skills", extract_corso_skills),
        ("Agent Docs", extract_agent_docs),
        ("CLAUDE.md Files", extract_claude_md_files),
        ("Bible Constitution", extract_bible_constitution),
        ("CORSO Cookbook", extract_corso_cookbook),
    ]

    for name, func in extractors:
        logger.info("--- %s ---", name)
        try:
            samples = func()
            all_samples.extend(samples)
            logger.info("  %s: %d samples", name, len(samples))
        except Exception:
            logger.exception("  %s: FAILED", name)

    logger.info("=" * 60)
    logger.info("ECOSYSTEM EXTRACTION COMPLETE: %d total samples", len(all_samples))
    logger.info("=" * 60)

    return all_samples
