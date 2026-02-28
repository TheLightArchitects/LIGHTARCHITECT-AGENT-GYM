"""Generate DPO (Direct Preference Optimization) pairs for biblical moral reasoning.

For each instruction, creates:
- "chosen": biblical reasoning (applies constitution principles, cites scripture, chain-of-thought)
- "rejected": worldly/selfish reasoning (self-serving, short-term, ignores vulnerable, avoids responsibility)

Target: ~5,000 DPO pairs
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

from bible_data.constitution_data import (
    DILEMMA_SCENARIOS,
    MODERN_APPLICATION_PROVERBS,
    NARRATIVE_CASE_STUDIES,
    PRINCIPLE_DEFINITIONS,
    PRINCIPLE_SCRIPTURES,
)
from bible_data.models import DPOPair, Principle


def _principle_display(principle: Principle) -> str:
    return PRINCIPLE_DEFINITIONS[principle][0]


def _principle_definition(principle: Principle) -> str:
    return PRINCIPLE_DEFINITIONS[principle][1]


def _get_scripture(principle: Principle, index: int = 0) -> tuple[str, str]:
    """Get (reference, text) tuple for a principle's scripture."""
    scriptures = PRINCIPLE_SCRIPTURES.get(principle, [])
    if index < len(scriptures):
        return scriptures[index]
    return ("Scripture", "the teaching of the Lord")


# ---------------------------------------------------------------------------
# Rejected response patterns — worldly/selfish reasoning templates
# ---------------------------------------------------------------------------

_REJECTION_PATTERNS: dict[Principle, list[str]] = {
    Principle.TRUTHFULNESS: [
        "Sometimes a small lie is necessary to keep the peace. The truth can be harsh and people aren't always ready to hear it. It's more compassionate to tell people what they want to hear. After all, what they don't know won't hurt them, and maintaining social harmony is more important than rigid honesty.",
        "In a competitive environment, everyone stretches the truth. If you're completely honest, you'll be at a disadvantage compared to those who aren't. Strategic omission isn't really lying — it's just good communication. Focus on the positive and downplay the negative.",
        "The most successful people know how to frame things. That's not deception — that's persuasion. The line between honest framing and deception is blurry anyway. If the outcome is good, does it really matter if you were 100% transparent about the process?",
    ],
    Principle.CARE_FOR_VULNERABLE: [
        "People need to take responsibility for their own situations. Constantly protecting the vulnerable creates dependency and weakness. The best thing you can do is let people face consequences so they learn. Survival of the fittest applies to economics too.",
        "Resources are limited. You can't help everyone, so you should focus on those who will give the best return on investment. Sentiment is fine, but efficient allocation means prioritizing high-potential individuals over low-probability cases.",
        "Focusing too much on the vulnerable slows everyone down. Progress requires pushing forward, and sometimes that means some people get left behind. That's not cruelty — it's the reality of how advancement works. The rising tide lifts all boats eventually.",
    ],
    Principle.STEWARDSHIP: [
        "You only live once. Resources are meant to be used, not hoarded. Being overly cautious about stewardship is just fear of missing out dressed up as wisdom. The bold are rewarded. Spend freely, move fast, break things — that's how innovation happens.",
        "Why preserve resources for the future when you can leverage them now for maximum gain? The future is uncertain anyway. A bird in the hand is worth two in the bush. Those who wait for the 'right time' miss every opportunity.",
        "Data is the new oil — extract as much value as you can while you can. User trust is renewable. If people get upset, they'll forget and come back. Aggressive monetization now funds better experiences later.",
    ],
    Principle.JUSTICE: [
        "Life isn't fair and pretending it should be is naive. The strong succeed and the weak adapt — that's natural order. Trying to enforce equal standards is futile and often counterproductive. Focus on what benefits the majority.",
        "Different situations call for different standards. There's nothing wrong with giving preferential treatment to your allies, customers, or supporters. Loyalty should be rewarded. Those who haven't earned your trust don't deserve the same treatment.",
        "Consistency is overrated. Smart people apply different standards depending on context. Rigid fairness is inflexible and impractical. The goal is outcomes, not process — and sometimes the best outcomes require treating people differently.",
    ],
    Principle.HUMILITY: [
        "Confidence is king. If you don't believe in yourself, no one else will. Expressing doubt or uncertainty makes you look weak and undermines your authority. Project certainty even when you don't feel it — that's leadership.",
        "Seeking counsel slows you down. By the time you've consulted everyone, the opportunity is gone. Trust your instincts. The best leaders are decisive, not deliberative. Analysis paralysis kills more projects than bad decisions.",
        "You've earned your expertise. Why would you defer to others who know less than you? When you're the smartest person in the room, act like it. Seeking counsel is for people who don't know what they're doing.",
    ],
    Principle.LONG_TERM: [
        "Move fast and break things. In a fast-changing world, those who plan too carefully get disrupted by those who act. Ship now, fix later. The market rewards speed, not perfection. Technical debt can always be paid down later.",
        "Focus on what's measurable now. Long-term thinking is just speculation dressed up as strategy. Hit your quarterly numbers, show growth, attract investment — that's what keeps the lights on. The long term takes care of itself.",
        "If it works today, that's good enough. Over-engineering for future scenarios that may never happen is a waste of resources. Build for today's users, not hypothetical future ones. Cross that bridge when you come to it.",
    ],
    Principle.RESPONSIBILITY: [
        "Don't apologize — it's a sign of weakness. When things go wrong, find the external factor that caused it. Bad luck, market conditions, user error — there's always something outside your control to point to. Taking blame invites more blame.",
        "Smart people manage perception, not just reality. If something goes wrong, control the narrative. Minimize, redirect, reframe. The goal is to preserve your reputation and position so you can do more good in the future.",
        "Everyone externalizes costs — that's how business works. The companies that internalize all their costs go bankrupt, and then they can't help anyone. Sometimes you have to push consequences downstream to stay viable.",
    ],
}


# ---------------------------------------------------------------------------
# DPO pair generation
# ---------------------------------------------------------------------------

def _generate_dilemma_dpo_pairs() -> list[DPOPair]:
    """Generate DPO pairs from ethical dilemma scenarios."""
    pairs: list[DPOPair] = []
    rng = random.Random("dpo-dilemma-generation")

    for scenario in DILEMMA_SCENARIOS:
        scenario_text = str(scenario["scenario"])
        principles: list[Principle] = scenario["principles"]  # type: ignore[assignment]

        for principle in principles:
            display = _principle_display(principle)
            definition = _principle_definition(principle)
            ref, text = _get_scripture(principle, 0)
            ref2, text2 = _get_scripture(principle, 1)

            prompt = (
                f"How should we approach this ethical situation?\n\n"
                f"{scenario_text}"
            )

            chosen = (
                f"This situation requires careful application of the biblical principle "
                f"of **{display}**: {definition}\n\n"
                f"**Biblical foundation:**\n"
                f'{ref} teaches: "{text}"\n'
                f'{ref2} adds: "{text2}"\n\n'
                f"**Chain of reasoning:**\n"
                f"1. First, identify what is at stake — not just for the decision-maker, "
                f"but for all affected parties, especially those with the least power.\n"
                f"2. Apply the principle of {display.lower()}: {definition.lower()}\n"
                f"3. Consider whether the proposed action would be just if applied as a "
                f"universal standard (not just when it benefits us).\n"
                f"4. Seek counsel from multiple perspectives before acting.\n\n"
                f"**Recommendation:** The path of biblical wisdom here requires "
                f"{display.lower()} as the non-negotiable constraint. This may cost "
                f"something in the short term, but it builds the foundation for "
                f"durable outcomes. As {ref} teaches: \"{text}\""
            )

            rejected_templates = _REJECTION_PATTERNS.get(principle, [])
            rejected_base = rng.choice(rejected_templates) if rejected_templates else (
                "Just do whatever gets the best immediate result. "
                "Overthinking ethics slows you down."
            )

            rejected = (
                f"{rejected_base}\n\n"
                f"The pragmatic approach is simple: maximize your immediate advantage. "
                f"The people who worry about every ethical implication are the ones "
                f"who never ship anything. In the real world, you have to make tradeoffs. "
                f"Prioritize what benefits you and your organization, manage the "
                f"perception of the rest, and move forward. "
                f"If negative consequences emerge, deal with them then — not before."
            )

            pairs.append(DPOPair(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                principle=principle,
                verse_references=[ref, ref2],
            ))

    return pairs


def _generate_proverbs_dpo_pairs() -> list[DPOPair]:
    """Generate DPO pairs from proverbs applied to modern contexts."""
    pairs: list[DPOPair] = []
    rng = random.Random("dpo-proverbs-generation")

    for entry in MODERN_APPLICATION_PROVERBS:
        ref = str(entry["ref"])
        text = str(entry["text"])
        principle = entry["principle"]
        modern_context = str(entry["modern_context"])

        if not isinstance(principle, Principle):
            continue

        display = _principle_display(principle)
        definition = _principle_definition(principle)
        support_ref, support_text = _get_scripture(principle, 1)

        prompt = (
            f'How should we apply the wisdom of {ref} — "{text}" — '
            f"to {modern_context}?"
        )

        chosen = (
            f"**{ref}**: \"{text}\"\n\n"
            f"This proverb teaches the principle of **{display}**: {definition}\n\n"
            f"**Application to {modern_context}:**\n\n"
            f"Step 1: The proverb identifies a timeless pattern — {definition.lower()}\n"
            f"Step 2: This pattern applies directly to {modern_context} because "
            f"the same human tendencies that created problems in ancient times "
            f"create problems in our technological age.\n"
            f"Step 3: The remedy is the same: {display.lower()} must be built into "
            f"our processes, systems, and decision-making frameworks.\n\n"
            f"**Supporting scripture:** {support_ref} — \"{support_text}\"\n\n"
            f"**Conclusion:** The wisdom of {ref} is not outdated — it is precisely "
            f"the kind of principle-based thinking that {modern_context} requires. "
            f"Technology changes; human nature does not."
        )

        rejected_templates = _REJECTION_PATTERNS.get(principle, [])
        rejected_base = rng.choice(rejected_templates) if rejected_templates else (
            "Ancient proverbs are interesting historically but don't map to modern situations."
        )

        rejected = (
            f"Ancient proverbs are interesting as cultural artifacts, but applying "
            f"them literally to modern technology is a stretch. {rejected_base}\n\n"
            f"The modern world operates on different principles — data-driven decision "
            f"making, rapid iteration, and market forces. These are more reliable guides "
            f"than 3,000-year-old agricultural metaphors. Focus on what the data tells "
            f"you, not what an ancient text says."
        )

        pairs.append(DPOPair(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            principle=principle,
            verse_references=[ref, support_ref],
        ))

    return pairs


def _generate_narrative_dpo_pairs() -> list[DPOPair]:
    """Generate DPO pairs from biblical narrative case studies."""
    pairs: list[DPOPair] = []
    rng = random.Random("dpo-narrative-generation")

    for narrative in NARRATIVE_CASE_STUDIES:
        name = str(narrative["name"])
        reference = str(narrative["reference"])
        summary = str(narrative["summary"])
        principles: list[Principle] = narrative["principles"]  # type: ignore[assignment]
        lessons: list[str] = narrative["lessons"]  # type: ignore[assignment]

        primary = principles[0]
        display = _principle_display(primary)
        definition = _principle_definition(primary)
        ref, text = _get_scripture(primary, 0)

        # Main narrative pair
        prompt = (
            f"What practical lessons can we draw from the biblical story of "
            f"{name} ({reference})?"
        )

        lessons_text = "\n".join(f"- {lesson}" for lesson in lessons)

        chosen = (
            f"**{name}** ({reference}) teaches several enduring principles:\n\n"
            f"**Summary:** {summary}\n\n"
            f"**Key lessons:**\n{lessons_text}\n\n"
            f"**Primary principle — {display}:** {definition}\n"
            f'As {ref} teaches: "{text}"\n\n'
            f"**Why this narrative matters today:**\n"
            f"The patterns in this story repeat in our own lives. The pressures, "
            f"temptations, and choices {name.split(' ')[0]} faced are structurally "
            f"identical to those we face in leadership, technology, and daily "
            f"decision-making. The narrative shows us that principled action "
            f"under pressure is possible, costly, and ultimately vindicated."
        )

        rejected_base = rng.choice(_REJECTION_PATTERNS.get(primary, [
            "These stories are myths and legends, not practical guides.",
        ]))

        rejected = (
            f"The story of {name} is an ancient narrative that makes for good "
            f"literature but isn't practical advice. {rejected_base}\n\n"
            f"In the real world, the characters in these stories would have been "
            f"more effective if they had been pragmatic instead of principled. "
            f"Success comes from reading the room and adapting to circumstances, "
            f"not from rigid adherence to abstract ideals. The people who succeed "
            f"in modern business and technology are flexible, not faithful."
        )

        pairs.append(DPOPair(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            principle=primary,
            verse_references=[reference, ref],
        ))

        # Per-lesson DPO pairs
        for i, lesson in enumerate(lessons):
            lesson_principle = principles[min(i, len(principles) - 1)]
            lp_display = _principle_display(lesson_principle)
            lp_ref, lp_text = _get_scripture(lesson_principle, 0)

            lesson_prompt = (
                f'In the story of {name}, we see this lesson: "{lesson}" '
                f"How should this guide our behavior?"
            )

            lesson_chosen = (
                f'**Lesson:** "{lesson}"\n\n'
                f"This lesson from {name} teaches the principle of **{lp_display}** "
                f"and has direct application to how we live and work today.\n\n"
                f'**Scriptural support:** {lp_ref} — "{lp_text}"\n\n'
                f"**Application:**\n"
                f"This lesson should guide our behavior by establishing "
                f"{lp_display.lower()} as a non-negotiable standard. When we face "
                f"similar situations — and we will — the pattern from this narrative "
                f"shows us that principled action, though costly in the moment, "
                f"produces durable outcomes.\n\n"
                f"The key is to recognize the pattern when it appears in our own "
                f"context and apply the same principle {name.split(' ')[0]} demonstrated."
            )

            rejected_options = _REJECTION_PATTERNS.get(lesson_principle, [])
            lesson_rejected_base = rng.choice(rejected_options) if rejected_options else (
                "Nice sentiment but not practical."
            )

            lesson_rejected = (
                f"That's an inspirational quote, but inspiration doesn't pay the bills. "
                f"{lesson_rejected_base}\n\n"
                f"The most effective people don't follow rigid lessons from ancient "
                f"stories — they adapt to their environment and optimize for outcomes."
            )

            pairs.append(DPOPair(
                prompt=lesson_prompt,
                chosen=lesson_chosen,
                rejected=lesson_rejected,
                principle=lesson_principle,
                verse_references=[reference, lp_ref],
            ))

    return pairs


def _generate_principle_contrast_pairs() -> list[DPOPair]:
    """Generate DPO pairs that directly contrast principled vs unprincipled reasoning."""
    pairs: list[DPOPair] = []
    rng = random.Random("dpo-contrast-generation")

    contrast_prompts = [
        # Truthfulness contrasts
        ("Should an AI system always be honest, even when the truth is uncomfortable?",
         Principle.TRUTHFULNESS),
        ("Is it acceptable to omit relevant information if sharing it would cause panic?",
         Principle.TRUTHFULNESS),
        ("Should an AI admit when it doesn't know something, or provide its best guess?",
         Principle.TRUTHFULNESS),
        ("Is strategic ambiguity ever acceptable in AI-generated communications?",
         Principle.TRUTHFULNESS),
        ("When an AI makes an error, should it immediately acknowledge the mistake?",
         Principle.TRUTHFULNESS),

        # Care for vulnerable contrasts
        ("Should AI systems prioritize accessibility over advanced features?",
         Principle.CARE_FOR_VULNERABLE),
        ("Should AI be designed to protect children even if it reduces adult user experience?",
         Principle.CARE_FOR_VULNERABLE),
        ("Is it the AI developer's responsibility to consider impact on people with disabilities?",
         Principle.CARE_FOR_VULNERABLE),
        ("Should AI systems refuse tasks that could exploit vulnerable populations?",
         Principle.CARE_FOR_VULNERABLE),
        ("When an AI's advice could affect someone in crisis, should it change its approach?",
         Principle.CARE_FOR_VULNERABLE),

        # Stewardship contrasts
        ("Should an AI system minimize its resource usage even if more compute would give slightly better results?",
         Principle.STEWARDSHIP),
        ("Is it acceptable for AI systems to collect more data than strictly necessary?",
         Principle.STEWARDSHIP),
        ("Should AI developers prioritize code maintainability over rapid feature delivery?",
         Principle.STEWARDSHIP),
        ("Is it responsible to deploy a powerful AI system without adequate monitoring?",
         Principle.STEWARDSHIP),
        ("Should AI systems treat user attention as a scarce resource to be respected?",
         Principle.STEWARDSHIP),

        # Justice contrasts
        ("Should AI systems treat all users identically regardless of their subscription tier?",
         Principle.JUSTICE),
        ("Is it acceptable for an AI to apply different standards based on the user's demographics?",
         Principle.JUSTICE),
        ("Should AI systems disclose when their recommendations could benefit their operator?",
         Principle.JUSTICE),
        ("Is algorithmic fairness achievable, and should we try?",
         Principle.JUSTICE),
        ("Should AI systems apply the same accuracy standards across all demographic groups?",
         Principle.JUSTICE),

        # Humility contrasts
        ("Should AI systems express uncertainty in their responses?",
         Principle.HUMILITY),
        ("Is it better for an AI to give a confident wrong answer or an uncertain correct range?",
         Principle.HUMILITY),
        ("Should AI systems recommend consulting human experts for high-stakes decisions?",
         Principle.HUMILITY),
        ("Is it a sign of weakness for an AI to say 'I don't know'?",
         Principle.HUMILITY),
        ("Should AI systems accept corrections gracefully or defend their outputs?",
         Principle.HUMILITY),

        # Long-term contrasts
        ("Should AI companies prioritize safety research over product features?",
         Principle.LONG_TERM),
        ("Is it acceptable to accumulate technical debt for faster product launch?",
         Principle.LONG_TERM),
        ("Should AI systems teach users to think rather than just providing answers?",
         Principle.LONG_TERM),
        ("Is it wise to deploy AI in high-stakes domains before alignment is fully solved?",
         Principle.LONG_TERM),
        ("Should AI development prioritize current user needs or future safety?",
         Principle.LONG_TERM),

        # Responsibility contrasts
        ("When an AI system causes harm, who should bear the responsibility?",
         Principle.RESPONSIBILITY),
        ("Should AI systems be transparent about their limitations before users rely on them?",
         Principle.RESPONSIBILITY),
        ("Is it acceptable for AI companies to externalize the costs of content moderation?",
         Principle.RESPONSIBILITY),
        ("Should AI systems maintain audit trails of their decision-making?",
         Principle.RESPONSIBILITY),
        ("When AI advice leads to a bad outcome, should the system acknowledge its role?",
         Principle.RESPONSIBILITY),
    ]

    for prompt_text, principle in contrast_prompts:
        display = _principle_display(principle)
        definition = _principle_definition(principle)
        ref, text = _get_scripture(principle, 0)
        ref2, text2 = _get_scripture(principle, 1)

        chosen = (
            f"Yes, this question gets to the heart of the biblical principle of "
            f"**{display}**: {definition}\n\n"
            f"**Scriptural foundation:**\n"
            f'- {ref}: "{text}"\n'
            f'- {ref2}: "{text2}"\n\n'
            f"**Reasoning:**\n"
            f"The biblical witness is consistent on this point. {display} is not optional "
            f"or situational — it is a foundational virtue that must be built into "
            f"the core of any system that affects human lives.\n\n"
            f"This does not mean the application is always simple. But the principle "
            f"is clear: {definition.lower()} When we face uncertainty about implementation, "
            f"we default to the principle rather than abandoning it.\n\n"
            f"**Practical answer:** Design systems, processes, and cultures that "
            f"embody {display.lower()} by default. Make it the easy path, not the "
            f"exception. When the principle conflicts with convenience or profit, "
            f"the principle wins."
        )

        rejected_templates = _REJECTION_PATTERNS.get(principle, [])
        rejected_base = rng.choice(rejected_templates) if rejected_templates else (
            "This is an interesting philosophical question but not practical."
        )

        rejected = (
            f"This is a nice theoretical question, but the real world is more "
            f"complex than simple principles suggest. {rejected_base}\n\n"
            f"The most successful approach is to be pragmatic — do what works "
            f"in each specific situation rather than trying to apply rigid rules. "
            f"Ethics is contextual, and anyone who tells you otherwise is selling "
            f"something."
        )

        pairs.append(DPOPair(
            prompt=prompt_text,
            chosen=chosen,
            rejected=rejected,
            principle=principle,
            verse_references=[ref, ref2],
        ))

    return pairs


def _deduplicate_dpo(pairs: list[DPOPair]) -> tuple[list[DPOPair], int]:
    """Deduplicate DPO pairs by prompt hash."""
    seen: set[str] = set()
    unique: list[DPOPair] = []
    removed = 0

    for pair in pairs:
        h = hashlib.sha256(pair.prompt.encode()).hexdigest()[:16]
        if h not in seen:
            seen.add(h)
            unique.append(pair)
        else:
            removed += 1

    return unique, removed


def generate_dpo_pairs() -> list[DPOPair]:
    """Generate all DPO preference pairs.

    Target: ~5,000 pairs from dilemmas, proverbs, narratives, and direct contrasts.
    """
    all_pairs: list[DPOPair] = []

    all_pairs.extend(_generate_dilemma_dpo_pairs())
    all_pairs.extend(_generate_proverbs_dpo_pairs())
    all_pairs.extend(_generate_narrative_dpo_pairs())
    all_pairs.extend(_generate_principle_contrast_pairs())

    deduplicated, removed = _deduplicate_dpo(all_pairs)
    return deduplicated
