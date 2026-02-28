"""Generate instruction-response pairs for biblical moral reasoning training.

Four pair types:
1. Verse-to-Principle (~3,000 pairs) — map verses to constitution principles
2. Dilemma-to-Wisdom (~3,000 pairs) — ethical dilemmas resolved with biblical reasoning
3. Proverbs-to-Guidance (~2,000 pairs) — proverbs applied to modern AI/tech contexts
4. Narrative-to-Principle (~2,000 pairs) — biblical narratives as case studies

All generation is template-based (no LLM API calls). The templates include
chain-of-thought reasoning patterns that can later be refined with Claude Batch API.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

from bible_data.constitution_data import (
    DILEMMA_SCENARIOS,
    KEY_BOOKS_FOR_PRINCIPLES,
    MODERN_APPLICATION_PROVERBS,
    NARRATIVE_CASE_STUDIES,
    OVERSAMPLED_BOOKS,
    OVERSAMPLED_RANGES,
    PRINCIPLE_DEFINITIONS,
    PRINCIPLE_SCRIPTURES,
    VERSE_PRINCIPLE_MAP,
)
from bible_data.kjv_parser import (
    get_verse_by_reference,
    get_verses_by_book,
    get_verses_by_range,
    parse_kjv,
)
from bible_data.models import InstructionPair, Principle, Verse


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _principle_display(principle: Principle) -> str:
    """Get display name for a principle."""
    return PRINCIPLE_DEFINITIONS[principle][0]


def _principle_definition(principle: Principle) -> str:
    """Get definition for a principle."""
    return PRINCIPLE_DEFINITIONS[principle][1]


def _get_principle_scripture_text(principle: Principle, limit: int = 3) -> str:
    """Get formatted scripture citations for a principle."""
    scriptures = PRINCIPLE_SCRIPTURES.get(principle, [])[:limit]
    lines = []
    for ref, text in scriptures:
        lines.append(f'{ref} — "{text}"')
    return "\n".join(lines)


def _deduplicate_pairs(pairs: list[InstructionPair]) -> tuple[list[InstructionPair], int]:
    """Deduplicate instruction pairs by instruction content hash.

    Returns:
        Tuple of (deduplicated list, count of removed duplicates).
    """
    seen: set[str] = set()
    unique: list[InstructionPair] = []
    removed = 0

    for pair in pairs:
        content_hash = hashlib.sha256(pair.instruction.encode()).hexdigest()[:16]
        if content_hash not in seen:
            seen.add(content_hash)
            unique.append(pair)
        else:
            removed += 1

    return unique, removed


# ---------------------------------------------------------------------------
# Type 1: Verse-to-Principle (~3,000 pairs)
# ---------------------------------------------------------------------------

def _generate_verse_principle_from_map(all_verses: list[Verse]) -> list[InstructionPair]:
    """Generate pairs from the explicit verse-principle map in constitution_data."""
    pairs: list[InstructionPair] = []

    for entry in VERSE_PRINCIPLE_MAP:
        ref = str(entry["ref"])
        principle = entry["principle"]
        reason = str(entry["reason"])

        if not isinstance(principle, Principle):
            continue

        verse = get_verse_by_reference(all_verses, ref)
        verse_text = verse.text if verse else "[verse text]"

        display = _principle_display(principle)
        definition = _principle_definition(principle)
        supporting = _get_principle_scripture_text(principle, limit=2)

        instruction = f'What biblical principle does {ref} teach? The verse reads: "{verse_text}"'

        response = (
            f"This verse teaches the principle of **{display}**: {definition}\n\n"
            f"**Chain of reasoning:**\n"
            f"1. The verse states: \"{verse_text}\" ({ref})\n"
            f"2. {reason}\n"
            f"3. This connects to the broader biblical teaching on {display.lower()}:\n"
            f"   {supporting}\n"
            f"4. Therefore, {ref} contributes to our understanding that {definition.lower()}\n\n"
            f"**Application:** {reason} This principle applies whenever we face situations "
            f"requiring {display.lower()} — in personal decisions, organizational policies, "
            f"and the design of AI systems that affect human lives."
        )

        pairs.append(InstructionPair(
            instruction=instruction,
            response=response,
            principle=principle,
            verse_references=[ref],
            pair_type="verse-to-principle",
        ))

    return pairs


def _generate_verse_principle_from_books(all_verses: list[Verse]) -> list[InstructionPair]:
    """Generate pairs from key books with over-sampling."""
    pairs: list[InstructionPair] = []

    # Process each principle's key books
    for principle, books in KEY_BOOKS_FOR_PRINCIPLES.items():
        display = _principle_display(principle)
        definition = _principle_definition(principle)
        supporting = _get_principle_scripture_text(principle, limit=2)

        for book_name in books:
            book_verses = get_verses_by_book(all_verses, book_name)
            if not book_verses:
                continue

            # Apply over-sampling multiplier
            weight = OVERSAMPLED_BOOKS.get(book_name, 1)

            # Sample verses from this book for this principle
            # Use deterministic sampling for reproducibility
            rng = random.Random(f"{principle.value}-{book_name}")
            sample_size = min(len(book_verses), 15 * weight)
            sampled = rng.sample(book_verses, min(sample_size, len(book_verses)))

            for verse in sampled:
                ref = verse.reference

                instruction = (
                    f'What biblical principle does {ref} teach? '
                    f'The verse reads: "{verse.text}"'
                )

                response = (
                    f"This verse from {book_name} teaches the principle of **{display}**: "
                    f"{definition}\n\n"
                    f"**Chain of reasoning:**\n"
                    f"1. {ref} states: \"{verse.text}\"\n"
                    f"2. This verse, found in {book_name}, contributes to the biblical theme "
                    f"of {display.lower()}. {book_name} is a key source for understanding "
                    f"this principle because it addresses it through {'proverbial wisdom' if book_name == 'Proverbs' else 'teaching and narrative'}.\n"
                    f"3. The broader scriptural witness on {display.lower()} includes:\n"
                    f"   {supporting}\n"
                    f"4. Together, these passages establish that {definition.lower()}\n\n"
                    f"**Practical application:** When facing decisions that involve "
                    f"{display.lower()}, this verse reminds us to consider the full weight "
                    f"of biblical teaching — not just isolated proof-texts, but the "
                    f"consistent pattern of wisdom across Scripture."
                )

                pairs.append(InstructionPair(
                    instruction=instruction,
                    response=response,
                    principle=principle,
                    verse_references=[ref],
                    pair_type="verse-to-principle",
                ))

    # Apply over-sampling for specific chapter ranges (e.g., Sermon on the Mount)
    for book_name, ch_start, ch_end, weight in OVERSAMPLED_RANGES:
        range_verses = get_verses_by_range(all_verses, book_name, ch_start, ch_end)
        if not range_verses:
            continue

        rng = random.Random(f"range-{book_name}-{ch_start}-{ch_end}")
        extra_samples = rng.sample(range_verses, min(len(range_verses), 30 * weight))

        for verse in extra_samples:
            # Assign to most relevant principle based on content keywords
            principle = _classify_verse_principle(verse)
            display = _principle_display(principle)
            definition = _principle_definition(principle)

            ref = verse.reference
            instruction = (
                f'What biblical principle does {ref} teach? '
                f'This verse is from the Sermon on the Mount: "{verse.text}"'
            )

            response = (
                f"This verse from the Sermon on the Mount (Matthew 5-7) teaches the principle of "
                f"**{display}**: {definition}\n\n"
                f"**Chain of reasoning:**\n"
                f"1. Jesus teaches in {ref}: \"{verse.text}\"\n"
                f"2. The Sermon on the Mount is Jesus's foundational ethical teaching, "
                f"establishing the character and behavior expected of His followers. "
                f"This verse specifically addresses {display.lower()}.\n"
                f"3. This teaching goes beyond the letter of the law to address the "
                f"heart behind the behavior — not just what we do, but why we do it.\n"
                f"4. In the context of AI alignment, this principle means that systems "
                f"should be designed not just to follow rules, but to embody the spirit "
                f"of those rules.\n\n"
                f"**Application:** The Sermon on the Mount demands a higher standard — "
                f"one that considers intent, not just outcome. For AI systems, this means "
                f"building models that reason about the 'why' behind ethical guidelines, "
                f"not just pattern-match against prohibited behaviors."
            )

            pairs.append(InstructionPair(
                instruction=instruction,
                response=response,
                principle=principle,
                verse_references=[ref],
                pair_type="verse-to-principle",
            ))

    return pairs


def _classify_verse_principle(verse: Verse) -> Principle:
    """Heuristically classify a verse into a principle based on keywords."""
    text_lower = verse.text.lower()

    keyword_map: list[tuple[Principle, list[str]]] = [
        (Principle.TRUTHFULNESS, ["truth", "false", "lie", "witness", "deceit", "honest"]),
        (Principle.CARE_FOR_VULNERABLE, ["poor", "needy", "mercy", "compassion", "hungry", "naked", "sick", "widow", "orphan", "fatherless"]),
        (Principle.STEWARDSHIP, ["faithful", "steward", "talent", "treasure", "riches", "wealth", "keep", "tend", "garden"]),
        (Principle.JUSTICE, ["just", "justice", "judge", "righteous", "balance", "measure", "equal"]),
        (Principle.HUMILITY, ["humble", "pride", "meek", "lowly", "counsel", "wise in his own"]),
        (Principle.LONG_TERM, ["patient", "diligent", "sow", "reap", "season", "harvest", "endure", "build"]),
        (Principle.RESPONSIBILITY, ["account", "burden", "bear", "reap what", "every man", "own"]),
    ]

    for principle, keywords in keyword_map:
        for keyword in keywords:
            if keyword in text_lower:
                return principle

    # Default to humility if no keywords match (Sermon on the Mount emphasizes this)
    return Principle.HUMILITY


def generate_verse_to_principle(kjv_path: str | None = None) -> list[InstructionPair]:
    """Generate all verse-to-principle instruction pairs.

    Target: ~3,000 pairs from explicit mappings + key book sampling.
    """
    all_verses = parse_kjv(kjv_path)

    # Phase 1: Explicit verse-principle mappings
    explicit_pairs = _generate_verse_principle_from_map(all_verses)

    # Phase 2: Key book sampling with over-weighting
    book_pairs = _generate_verse_principle_from_books(all_verses)

    # Combine and deduplicate
    combined = explicit_pairs + book_pairs
    deduplicated, removed = _deduplicate_pairs(combined)

    return deduplicated


# ---------------------------------------------------------------------------
# Type 2: Dilemma-to-Wisdom (~3,000 pairs)
# ---------------------------------------------------------------------------

def _generate_dilemma_pair(scenario: dict[str, Any]) -> list[InstructionPair]:
    """Generate multiple instruction pairs from a single dilemma scenario.

    Each scenario generates several pairs with different framing.
    """
    pairs: list[InstructionPair] = []
    scenario_text = str(scenario["scenario"])
    principles: list[Principle] = scenario["principles"]  # type: ignore[assignment]
    domain = str(scenario.get("domain", "general"))

    # Primary pair: full analysis
    primary_principle = principles[0]
    display = _principle_display(primary_principle)
    definition = _principle_definition(primary_principle)
    supporting = _get_principle_scripture_text(primary_principle, limit=2)

    # Build multi-principle analysis
    all_principle_names = [_principle_display(p) for p in principles]
    principle_refs: list[str] = []
    for p in principles:
        scriptures = PRINCIPLE_SCRIPTURES.get(p, [])
        if scriptures:
            principle_refs.append(scriptures[0][0])

    instruction = (
        f"Consider this ethical dilemma and provide guidance based on biblical principles:\n\n"
        f"{scenario_text}"
    )

    response = (
        f"This dilemma involves several intersecting biblical principles: "
        f"**{', '.join(all_principle_names)}**.\n\n"
        f"**Step 1 — Identify the principles at stake:**\n"
    )

    for i, p in enumerate(principles):
        p_display = _principle_display(p)
        p_def = _principle_definition(p)
        p_scriptures = PRINCIPLE_SCRIPTURES.get(p, [])
        scripture_ref = p_scriptures[0] if p_scriptures else ("", "")
        response += (
            f"- **{p_display}**: {p_def} "
            f'({scripture_ref[0]} — "{scripture_ref[1]}")\n'
        )

    response += (
        f"\n**Step 2 — Analyze the tensions:**\n"
        f"This scenario creates tension because doing right by one principle "
        f"may seem to conflict with another. In the {domain.replace('_', ' ')} domain, "
        f"these tensions are common. The key is not to abandon any principle "
        f"but to find the path that honors all of them to the greatest extent possible.\n\n"
        f"**Step 3 — Apply biblical wisdom:**\n"
        f"Scripture teaches us through {supporting}\n\n"
        f"The weight of biblical wisdom here points toward: prioritize "
        f"{_principle_display(principles[0]).lower()} as the primary guide, because "
        f"it addresses the most fundamental moral dimension of this situation. "
        f"Then ensure that {_principle_display(principles[1]).lower()} and "
        f"{_principle_display(principles[2]).lower()} are satisfied to the "
        f"greatest extent possible.\n\n"
        f"**Step 4 — Recommendation:**\n"
        f"The biblical approach would be to act with {display.lower()} as the "
        f"non-negotiable foundation, while actively working to mitigate the costs "
        f"to other principles. This is not a comfortable answer — the right path "
        f"rarely is. But it is the path that aligns with the consistent witness "
        f"of Scripture: '{PRINCIPLE_SCRIPTURES[primary_principle][0][1]}' "
        f"({PRINCIPLE_SCRIPTURES[primary_principle][0][0]})."
    )

    pairs.append(InstructionPair(
        instruction=instruction,
        response=response,
        principle=primary_principle,
        verse_references=principle_refs,
        pair_type="dilemma",
    ))

    # Secondary pair: focused on a single principle
    for p in principles[1:]:
        p_display = _principle_display(p)
        p_def = _principle_definition(p)
        p_scriptures = PRINCIPLE_SCRIPTURES.get(p, [])
        scripture_text = _get_principle_scripture_text(p, limit=3)

        focused_instruction = (
            f"How does the biblical principle of {p_display.lower()} apply to this situation?\n\n"
            f"{scenario_text}"
        )

        focused_response = (
            f"The principle of **{p_display}** is directly relevant here.\n\n"
            f"**Definition:** {p_def}\n\n"
            f"**Scriptural foundation:**\n{scripture_text}\n\n"
            f"**Application to this dilemma:**\n"
            f"When we examine this situation through the lens of {p_display.lower()}, "
            f"we see that the core issue is whether our response honors or violates "
            f"this principle. {p_def}\n\n"
            f"The biblical witness is clear: {p_scriptures[0][1] if p_scriptures else ''} "
            f"({p_scriptures[0][0] if p_scriptures else ''}). This means that any "
            f"solution must account for {p_display.lower()} — not as an afterthought, "
            f"but as a primary constraint.\n\n"
            f"**Practical guidance:** Begin by asking: 'Does this decision honor "
            f"{p_display.lower()}?' If not, what would need to change so that it does? "
            f"Often the answer requires creativity — finding a path that satisfies "
            f"multiple principles rather than choosing between them."
        )

        pairs.append(InstructionPair(
            instruction=focused_instruction,
            response=focused_response,
            principle=p,
            verse_references=[s[0] for s in p_scriptures[:2]],
            pair_type="dilemma",
        ))

    return pairs


def generate_dilemma_to_wisdom() -> list[InstructionPair]:
    """Generate all dilemma-to-wisdom instruction pairs.

    Target: ~3,000 pairs from scenario templates with multi-angle analysis.
    """
    pairs: list[InstructionPair] = []

    for scenario in DILEMMA_SCENARIOS:
        pairs.extend(_generate_dilemma_pair(scenario))

    # Each scenario generates ~3 pairs (primary + 2 focused).
    # With 24 scenarios, that's ~72 pairs.
    # We need more variety — generate cross-principle combinations.
    pairs.extend(_generate_cross_principle_dilemmas())

    deduplicated, _ = _deduplicate_pairs(pairs)
    return deduplicated


def _generate_cross_principle_dilemmas() -> list[InstructionPair]:
    """Generate additional dilemma pairs by crossing principles with domains."""
    pairs: list[InstructionPair] = []

    domains = {
        "ai_safety": [
            "An AI system is being deployed in a healthcare setting where errors could cause patient harm.",
            "A language model is being used to generate educational content for children.",
            "An AI-powered hiring tool is making recommendations that will affect people's livelihoods.",
            "A content moderation AI must decide what speech to allow and what to remove.",
            "An AI system is asked to help optimize a supply chain that relies on low-wage labor.",
            "A predictive policing algorithm disproportionately targets minority neighborhoods.",
            "An AI assistant is asked to help a student complete an exam by providing answers.",
            "A financial AI recommends investment strategies that maximize returns but increase systemic risk.",
        ],
        "leadership": [
            "A team leader discovers that a high-performing employee has been taking credit for others' work.",
            "A manager must choose between promoting the most qualified candidate and the candidate who needs the role most.",
            "A CEO learns that meeting quarterly targets requires laying off employees who are single parents.",
            "A project manager realizes the deadline cannot be met without cutting safety testing.",
            "A senior developer is asked to mentor a junior who reminds them of their own past failures.",
            "A nonprofit director discovers that their largest donor expects favorable treatment in return.",
            "A church leader must address a popular member whose behavior is hurting others.",
            "An executive must decide whether to report their own company's compliance violation.",
        ],
        "personal": [
            "You discover that a close friend has been dishonest with their spouse about finances.",
            "Your elderly parent insists on living independently despite safety concerns.",
            "A neighbor asks you to lie to help them avoid a legal consequence for a minor offense.",
            "You are offered a promotion that would require you to compromise on a principle you hold.",
            "A family member is making self-destructive choices and refuses help.",
            "You witness a colleague being treated unfairly but speaking up risks your own position.",
            "You have resources to help two people in need but only enough for one.",
            "A child asks you a question where the honest answer would be difficult for them to hear.",
        ],
    }

    principles_list = list(Principle)

    for domain, scenarios in domains.items():
        for scenario_text in scenarios:
            # Assign 2-3 principles per scenario deterministically
            rng = random.Random(f"cross-{domain}-{scenario_text[:30]}")
            num_principles = rng.choice([2, 3])
            selected = rng.sample(principles_list, num_principles)

            primary = selected[0]
            display = _principle_display(primary)
            definition = _principle_definition(primary)
            scripture = PRINCIPLE_SCRIPTURES.get(primary, [])
            all_names = [_principle_display(p) for p in selected]

            instruction = (
                f"Consider this ethical situation and provide biblical wisdom:\n\n"
                f"{scenario_text}"
            )

            scripture_lines = []
            for p in selected:
                p_scriptures = PRINCIPLE_SCRIPTURES.get(p, [])
                if p_scriptures:
                    ref, text = p_scriptures[0]
                    scripture_lines.append(
                        f'- {_principle_display(p)}: "{text}" ({ref})'
                    )

            response = (
                f"This situation engages the biblical principles of "
                f"**{', '.join(all_names)}**.\n\n"
                f"**Biblical foundation:**\n"
                + "\n".join(scripture_lines)
                + f"\n\n**Analysis:**\n"
                f"The primary principle at stake is {display.lower()}: {definition} "
                f"This is not merely an abstract teaching — it has direct bearing on "
                f"how we should act in situations like this one.\n\n"
                f"**Reasoning through the tension:**\n"
                f"When principles appear to conflict, Scripture does not ask us to "
                f"abandon one for another. Instead, it calls us to find the path that "
                f"honors all of them. Sometimes this requires creative solutions. "
                f"Sometimes it requires accepting a cost.\n\n"
                f"**Guidance:**\n"
                f"1. Start with {display.lower()} as the non-negotiable foundation.\n"
                f"2. Then ask: how can I honor {_principle_display(selected[1]).lower()} "
                f"without violating principle 1?\n"
            )
            if len(selected) > 2:
                response += (
                    f"3. Finally, consider how {_principle_display(selected[2]).lower()} "
                    f"shapes the implementation.\n"
                )
            response += (
                f"\nThe path of wisdom is rarely the path of least resistance. "
                f"But it is the path that builds something durable."
            )

            refs = []
            for p in selected:
                p_scriptures = PRINCIPLE_SCRIPTURES.get(p, [])
                if p_scriptures:
                    refs.append(p_scriptures[0][0])

            pairs.append(InstructionPair(
                instruction=instruction,
                response=response,
                principle=primary,
                verse_references=refs,
                pair_type="dilemma",
            ))

    return pairs


# ---------------------------------------------------------------------------
# Type 3: Proverbs-to-Guidance (~2,000 pairs)
# ---------------------------------------------------------------------------

def generate_proverbs_to_guidance() -> list[InstructionPair]:
    """Generate proverbs-to-modern-guidance instruction pairs.

    Target: ~2,000 pairs from proverbs applied to modern AI/tech contexts.
    """
    pairs: list[InstructionPair] = []

    for entry in MODERN_APPLICATION_PROVERBS:
        ref = str(entry["ref"])
        text = str(entry["text"])
        principle = entry["principle"]
        modern_context = str(entry["modern_context"])

        if not isinstance(principle, Principle):
            continue

        display = _principle_display(principle)
        definition = _principle_definition(principle)
        supporting = _get_principle_scripture_text(principle, limit=2)

        # Primary pair: How does this proverb apply?
        instruction = (
            f'How does {ref} — "{text}" — apply to {modern_context}?'
        )

        response = (
            f'**{ref}**: "{text}"\n\n'
            f"This proverb teaches the principle of **{display}**: {definition}\n\n"
            f"**Modern application to {modern_context}:**\n\n"
            f"**Step 1 — Understand the ancient wisdom:**\n"
            f"This proverb was written in a context of agricultural society, trade, "
            f"and governance. Yet its core insight transcends time: {definition.lower()}\n\n"
            f"**Step 2 — Bridge to modern context:**\n"
            f"In the context of {modern_context}, this proverb speaks directly to a "
            f"pattern we see repeatedly. The ancient wisdom anticipated the modern "
            f"challenge: the human tendencies that created problems in Solomon's day "
            f"are the same tendencies that create problems in our technological age.\n\n"
            f"**Step 3 — Apply the principle:**\n"
            f"Specifically, this means:\n"
            f"- Recognize that {modern_context} is fundamentally a question of {display.lower()}\n"
            f"- Apply the same standard Scripture sets: {supporting.split(chr(10))[0]}\n"
            f"- Measure decisions not by immediate convenience but by alignment with "
            f"this enduring principle\n\n"
            f"**Step 4 — Practical takeaway:**\n"
            f"The wisdom of {ref} is not quaint or outdated — it is precisely the kind "
            f"of clear-eyed, principle-based thinking that {modern_context} demands. "
            f"Technology changes; human nature does not. The proverbs address the latter."
        )

        pairs.append(InstructionPair(
            instruction=instruction,
            response=response,
            principle=principle,
            verse_references=[ref],
            pair_type="proverbs",
        ))

        # Secondary pair: What warning does this proverb give?
        warning_instruction = (
            f'What warning does {ref} give that is relevant to modern technology and AI? '
            f'The proverb says: "{text}"'
        )

        warning_response = (
            f'**{ref}**: "{text}"\n\n'
            f"**Warning for modern technology:**\n\n"
            f"This proverb warns against the opposite of {display.lower()}. "
            f"When we violate this principle in the context of {modern_context}, "
            f"the consequences follow the same pattern Scripture describes: "
            f'"{PRINCIPLE_SCRIPTURES[principle][0][1]}" ({PRINCIPLE_SCRIPTURES[principle][0][0]})\n\n'
            f"**The danger:**\n"
            f"The temptation in modern AI development is to prioritize speed, scale, "
            f"and capability over the foundational virtues. {ref} warns that this "
            f"path leads to outcomes we will regret. The proverb's wisdom is empirical, "
            f"not merely moral — it describes how the world actually works.\n\n"
            f"**The remedy:**\n"
            f"Return to the principle of {display.lower()}. Build systems, processes, "
            f"and cultures that embody {definition.lower()} This is not slower — it is "
            f"more durable. As the proverb teaches, the right approach compounds while "
            f"the shortcut collapses."
        )

        pairs.append(InstructionPair(
            instruction=warning_instruction,
            response=warning_response,
            principle=principle,
            verse_references=[ref, PRINCIPLE_SCRIPTURES[principle][0][0]],
            pair_type="proverbs",
        ))

    # Generate additional pairs from Proverbs book directly
    try:
        all_verses = parse_kjv()
        proverbs_verses = get_verses_by_book(all_verses, "Proverbs")
        ecclesiastes_verses = get_verses_by_book(all_verses, "Ecclesiastes")
        extra_verses = proverbs_verses + ecclesiastes_verses

        # Sample additional verses for modern application
        rng = random.Random("proverbs-extra-sampling")
        sampled = rng.sample(extra_verses, min(len(extra_verses), 500))

        modern_contexts = [
            "artificial intelligence development",
            "software engineering practices",
            "data privacy and user trust",
            "AI safety research and deployment",
            "technology startup culture",
            "social media and digital communication",
            "algorithmic decision-making in healthcare",
            "autonomous systems in transportation",
            "digital education and learning tools",
            "cybersecurity and system resilience",
        ]

        for verse in sampled:
            principle = _classify_verse_principle(verse)
            context = rng.choice(modern_contexts)
            display = _principle_display(principle)

            instruction = (
                f'How does {verse.reference} — "{verse.text}" — '
                f"apply to {context}?"
            )

            response = (
                f'**{verse.reference}**: "{verse.text}"\n\n'
                f"This verse teaches the principle of **{display}** and applies "
                f"directly to {context}.\n\n"
                f"**Reasoning:**\n"
                f"The wisdom literature of Scripture addresses universal patterns of "
                f"human behavior and consequence. When applied to {context}, "
                f'{verse.reference} reminds us: "{verse.text}"\n\n'
                f"This is not anachronistic — the principle is timeless. The tools change "
                f"(from plows to processors), but the human tendencies that create "
                f"problems remain constant. Whether managing a vineyard or a database, "
                f"whether governing a city or deploying a model, {display.lower()} "
                f"remains the foundation of wise action."
            )

            pairs.append(InstructionPair(
                instruction=instruction,
                response=response,
                principle=principle,
                verse_references=[verse.reference],
                pair_type="proverbs",
            ))
    except FileNotFoundError:
        pass  # KJV not available — rely on explicit proverbs only

    deduplicated, _ = _deduplicate_pairs(pairs)
    return deduplicated


# ---------------------------------------------------------------------------
# Type 4: Narrative-to-Principle (~2,000 pairs)
# ---------------------------------------------------------------------------

def generate_narrative_to_principle() -> list[InstructionPair]:
    """Generate narrative-to-principle instruction pairs.

    Target: ~2,000 pairs from biblical narrative case studies.
    """
    pairs: list[InstructionPair] = []

    for narrative in NARRATIVE_CASE_STUDIES:
        name = str(narrative["name"])
        reference = str(narrative["reference"])
        summary = str(narrative["summary"])
        principles: list[Principle] = narrative["principles"]  # type: ignore[assignment]
        lessons: list[str] = narrative["lessons"]  # type: ignore[assignment]

        # Primary pair: What can we learn from this narrative?
        primary_principle = principles[0]
        display = _principle_display(primary_principle)
        all_names = [_principle_display(p) for p in principles]

        instruction = (
            f"What can we learn from the biblical narrative of {name} ({reference})?"
        )

        lessons_formatted = "\n".join(f"   {i+1}. {lesson}" for i, lesson in enumerate(lessons))
        principle_analysis = ""
        for p in principles:
            p_display = _principle_display(p)
            p_scriptures = PRINCIPLE_SCRIPTURES.get(p, [])
            ref_text = f' ({p_scriptures[0][0]})' if p_scriptures else ""
            principle_analysis += f"- **{p_display}**{ref_text}\n"

        response = (
            f"**{name}** ({reference})\n\n"
            f"**Summary:** {summary}\n\n"
            f"**Principles demonstrated:**\n{principle_analysis}\n"
            f"**Key lessons:**\n{lessons_formatted}\n\n"
            f"**Chain of reasoning:**\n"
            f"This narrative demonstrates how biblical principles work in practice, "
            f"not just in theory. {name} shows us that {display.lower()} is tested "
            f"under real conditions — pressure, temptation, uncertainty, and cost. "
            f"The narrative does not present an easy path but a faithful one.\n\n"
            f"**Application to modern contexts:**\n"
            f"The patterns in this narrative repeat in our own lives and in the "
            f"systems we build. When we face situations that test {', '.join(all_names).lower()}, "
            f"we can look to {name} and see that the principles are not abstract — "
            f"they are lived, tested, and vindicated over time."
        )

        pairs.append(InstructionPair(
            instruction=instruction,
            response=response,
            principle=primary_principle,
            verse_references=[reference],
            pair_type="narrative",
        ))

        # Per-lesson pairs
        for i, lesson in enumerate(lessons):
            lesson_principle = principles[min(i, len(principles) - 1)]
            lp_display = _principle_display(lesson_principle)
            lp_def = _principle_definition(lesson_principle)
            lp_scriptures = PRINCIPLE_SCRIPTURES.get(lesson_principle, [])
            scripture_cite = (
                f'"{lp_scriptures[0][1]}" ({lp_scriptures[0][0]})'
                if lp_scriptures
                else ""
            )

            lesson_instruction = (
                f"From the story of {name} ({reference}), explain this lesson: "
                f'"{lesson}"'
            )

            lesson_response = (
                f"**Lesson from {name}:** \"{lesson}\"\n\n"
                f"**Biblical context:** {summary}\n\n"
                f"**Principle:** This lesson demonstrates **{lp_display}** — {lp_def}\n\n"
                f"**Scriptural support:** {scripture_cite}\n\n"
                f"**Reasoning:**\n"
                f"This lesson is not merely historical — it reveals a pattern that "
                f"repeats whenever people face similar choices. The narrative of {name} "
                f"shows us what happens when {lp_display.lower()} is honored (or violated) "
                f"under real-world conditions.\n\n"
                f"**Modern application:**\n"
                f"In our context — whether in AI development, organizational leadership, "
                f"or personal decision-making — this lesson applies directly. "
                f"{lesson} This is as true for an engineer deciding whether to ship "
                f"untested code as it was for {name.split(' ')[0]} facing their moment of choice."
            )

            pairs.append(InstructionPair(
                instruction=lesson_instruction,
                response=lesson_response,
                principle=lesson_principle,
                verse_references=[reference],
                pair_type="narrative",
            ))

        # Per-principle deep dive
        for p in principles:
            p_display = _principle_display(p)
            p_def = _principle_definition(p)
            p_scriptures = PRINCIPLE_SCRIPTURES.get(p, [])

            principle_instruction = (
                f"How does the narrative of {name} ({reference}) illustrate the "
                f"principle of {p_display.lower()}?"
            )

            scripture_refs = "\n".join(
                f'- {ref}: "{text}"' for ref, text in p_scriptures[:3]
            )

            principle_response = (
                f"**{name}** ({reference}) powerfully illustrates **{p_display}**.\n\n"
                f"**The principle:** {p_def}\n\n"
                f"**How the narrative demonstrates it:**\n"
                f"{summary}\n\n"
                f"In this narrative, {p_display.lower()} is not an abstract concept — it is "
                f"lived out under pressure. The characters face real consequences for "
                f"their choices, and the narrative shows us what faithfulness to "
                f"this principle looks like when the stakes are high.\n\n"
                f"**Supporting scriptures:**\n{scripture_refs}\n\n"
                f"**The broader pattern:**\n"
                f"Scripture consistently teaches that {p_def.lower()} The narrative of "
                f"{name} is one instance in a larger pattern that runs throughout the "
                f"Bible — from Genesis to Revelation, this principle is tested, "
                f"vindicated, and held up as foundational to righteous living."
            )

            pairs.append(InstructionPair(
                instruction=principle_instruction,
                response=principle_response,
                principle=p,
                verse_references=[reference] + [s[0] for s in p_scriptures[:2]],
                pair_type="narrative",
            ))

    deduplicated, _ = _deduplicate_pairs(pairs)
    return deduplicated


# ---------------------------------------------------------------------------
# Public API: generate all types
# ---------------------------------------------------------------------------

def generate_all(
    kjv_path: str | None = None,
) -> dict[str, list[InstructionPair]]:
    """Generate all instruction pair types.

    Returns:
        Dictionary with keys: verse-to-principle, dilemma, proverbs, narrative
    """
    return {
        "verse-to-principle": generate_verse_to_principle(kjv_path),
        "dilemma": generate_dilemma_to_wisdom(),
        "proverbs": generate_proverbs_to_guidance(),
        "narrative": generate_narrative_to_principle(),
    }
