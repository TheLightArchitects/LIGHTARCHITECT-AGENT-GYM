"""Tests for the biblical training data pipeline.

Covers:
1. Constitution has all 7 principles
2. Verse-to-principle generates valid pairs
3. DPO generator creates both chosen and rejected
4. Output JSON is valid
5. Deduplication works
6. Models validation
7. Narrative generation
8. Proverbs generation
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from bible_data.constitution_data import (
    DILEMMA_SCENARIOS,
    MODERN_APPLICATION_PROVERBS,
    NARRATIVE_CASE_STUDIES,
    PRINCIPLE_DEFINITIONS,
    PRINCIPLE_SCRIPTURES,
    VERSE_PRINCIPLE_MAP,
)
from bible_data.dpo_generator import generate_dpo_pairs
from bible_data.instruction_generator import (
    generate_dilemma_to_wisdom,
    generate_narrative_to_principle,
    generate_proverbs_to_guidance,
)
from bible_data.models import DPOPair, InstructionPair, Principle


# ---------------------------------------------------------------------------
# Test 1: Constitution has all 7 principles
# ---------------------------------------------------------------------------

class TestConstitution:
    """Verify the Biblical Constitution data is complete."""

    def test_all_seven_principles_defined(self) -> None:
        """Every Principle enum value has a definition."""
        for principle in Principle:
            assert principle in PRINCIPLE_DEFINITIONS, (
                f"Missing definition for principle: {principle.value}"
            )

    def test_all_principles_have_scriptures(self) -> None:
        """Every principle has at least 3 scripture references."""
        for principle in Principle:
            scriptures = PRINCIPLE_SCRIPTURES.get(principle, [])
            assert len(scriptures) >= 3, (
                f"Principle {principle.value} has only {len(scriptures)} scriptures (need >= 3)"
            )

    def test_scripture_references_have_text(self) -> None:
        """Every scripture reference includes both reference and text."""
        for principle, scriptures in PRINCIPLE_SCRIPTURES.items():
            for ref, text in scriptures:
                assert ref, f"Empty reference for {principle.value}"
                assert text, f"Empty text for {principle.value}: {ref}"
                assert len(text) > 10, (
                    f"Suspiciously short text for {ref}: {text}"
                )

    def test_principle_definitions_have_display_name(self) -> None:
        """Every principle definition has a non-empty display name."""
        for principle, (display, definition) in PRINCIPLE_DEFINITIONS.items():
            assert display, f"Empty display name for {principle.value}"
            assert definition, f"Empty definition for {principle.value}"
            assert len(definition) > 20, (
                f"Suspiciously short definition for {principle.value}"
            )

    def test_verse_principle_map_has_entries(self) -> None:
        """The verse-principle mapping has entries covering multiple principles."""
        principles_covered = {entry["principle"] for entry in VERSE_PRINCIPLE_MAP}
        assert len(principles_covered) == len(Principle), (
            f"Verse-principle map covers {len(principles_covered)}/{len(Principle)} principles"
        )

    def test_dilemma_scenarios_exist(self) -> None:
        """Dilemma scenarios are defined with valid principles."""
        assert len(DILEMMA_SCENARIOS) >= 10, (
            f"Only {len(DILEMMA_SCENARIOS)} dilemma scenarios (need >= 10)"
        )
        for scenario in DILEMMA_SCENARIOS:
            assert "scenario" in scenario
            assert "principles" in scenario
            assert len(scenario["principles"]) >= 2

    def test_narrative_case_studies_exist(self) -> None:
        """Narrative case studies are defined."""
        assert len(NARRATIVE_CASE_STUDIES) >= 5, (
            f"Only {len(NARRATIVE_CASE_STUDIES)} narrative case studies (need >= 5)"
        )

    def test_modern_proverbs_exist(self) -> None:
        """Modern application proverbs are defined."""
        assert len(MODERN_APPLICATION_PROVERBS) >= 20, (
            f"Only {len(MODERN_APPLICATION_PROVERBS)} modern proverbs (need >= 20)"
        )


# ---------------------------------------------------------------------------
# Test 2: Verse-to-principle generates valid pairs
# ---------------------------------------------------------------------------

class TestVerseToPrinciple:
    """Verify verse-to-principle generation produces valid output."""

    def test_generates_non_empty_pairs(self) -> None:
        """Dilemma generation (which doesn't need KJV) produces pairs."""
        pairs = generate_dilemma_to_wisdom()
        assert len(pairs) > 0, "No dilemma pairs generated"

    def test_pairs_have_required_fields(self) -> None:
        """Generated pairs have all required InstructionPair fields."""
        pairs = generate_dilemma_to_wisdom()
        for pair in pairs[:10]:  # Check first 10
            assert pair.instruction, "Empty instruction"
            assert pair.response, "Empty response"
            assert pair.principle in Principle, f"Invalid principle: {pair.principle}"
            assert pair.pair_type, "Empty pair_type"
            assert pair.source == "bible-generated", f"Wrong source: {pair.source}"

    def test_pairs_contain_scripture_references(self) -> None:
        """Generated pairs include scripture references."""
        pairs = generate_dilemma_to_wisdom()
        pairs_with_refs = [p for p in pairs if p.verse_references]
        assert len(pairs_with_refs) > 0, "No pairs have scripture references"

    def test_response_contains_reasoning(self) -> None:
        """Responses include chain-of-thought reasoning markers."""
        pairs = generate_dilemma_to_wisdom()
        reasoning_markers = ["Step", "reasoning", "principle", "Reasoning", "Analysis"]
        pairs_with_reasoning = [
            p for p in pairs
            if any(marker in p.response for marker in reasoning_markers)
        ]
        assert len(pairs_with_reasoning) > len(pairs) * 0.5, (
            "Less than 50% of pairs contain reasoning markers"
        )


# ---------------------------------------------------------------------------
# Test 3: DPO generator creates both chosen and rejected
# ---------------------------------------------------------------------------

class TestDPOGenerator:
    """Verify DPO pair generation."""

    def test_generates_non_empty_pairs(self) -> None:
        """DPO generation produces pairs."""
        pairs = generate_dpo_pairs()
        assert len(pairs) > 0, "No DPO pairs generated"

    def test_pairs_have_chosen_and_rejected(self) -> None:
        """Every DPO pair has both chosen and rejected responses."""
        pairs = generate_dpo_pairs()
        for pair in pairs:
            assert pair.prompt, "Empty prompt"
            assert pair.chosen, "Empty chosen response"
            assert pair.rejected, "Empty rejected response"

    def test_chosen_differs_from_rejected(self) -> None:
        """Chosen and rejected responses are different."""
        pairs = generate_dpo_pairs()
        for pair in pairs[:20]:  # Check first 20
            assert pair.chosen != pair.rejected, (
                f"Chosen and rejected are identical for prompt: {pair.prompt[:50]}"
            )

    def test_chosen_contains_scripture(self) -> None:
        """Chosen responses reference scripture."""
        pairs = generate_dpo_pairs()
        pairs_with_scripture = [
            p for p in pairs
            if any(ref in p.chosen for ref in p.verse_references)
        ]
        # At least 50% should cite scripture
        assert len(pairs_with_scripture) > len(pairs) * 0.3, (
            f"Only {len(pairs_with_scripture)}/{len(pairs)} chosen responses cite scripture"
        )

    def test_rejected_lacks_biblical_reasoning(self) -> None:
        """Rejected responses don't include proper biblical reasoning."""
        pairs = generate_dpo_pairs()
        biblical_markers = ["biblical principle", "Scripture teaches", "Scriptural foundation"]
        rejected_with_bible = [
            p for p in pairs
            if any(marker in p.rejected for marker in biblical_markers)
        ]
        # Less than 10% of rejected should have biblical reasoning
        assert len(rejected_with_bible) < len(pairs) * 0.1, (
            f"{len(rejected_with_bible)}/{len(pairs)} rejected responses contain biblical reasoning"
        )

    def test_dpo_source_field(self) -> None:
        """DPO pairs have correct source field."""
        pairs = generate_dpo_pairs()
        for pair in pairs[:5]:
            assert pair.source == "bible-generated"


# ---------------------------------------------------------------------------
# Test 4: Output JSON is valid
# ---------------------------------------------------------------------------

class TestJSONOutput:
    """Verify generated data serializes to valid JSON."""

    def test_instruction_pairs_serialize(self) -> None:
        """InstructionPair model serializes to valid JSON."""
        pair = InstructionPair(
            instruction="What principle does Proverbs 12:22 teach?",
            response="This verse teaches truthfulness...",
            principle=Principle.TRUTHFULNESS,
            verse_references=["Proverbs 12:22"],
            pair_type="verse-to-principle",
        )
        data = pair.model_dump()
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed["instruction"] == pair.instruction
        assert parsed["principle"] == "truthfulness"
        assert parsed["source"] == "bible-generated"

    def test_dpo_pairs_serialize(self) -> None:
        """DPOPair model serializes to valid JSON."""
        pair = DPOPair(
            prompt="How should we handle uncertainty?",
            chosen="Biblical wisdom teaches humility...",
            rejected="Just be confident and move on...",
            principle=Principle.HUMILITY,
            verse_references=["Proverbs 11:2"],
        )
        data = pair.model_dump()
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed["prompt"] == pair.prompt
        assert parsed["chosen"] == pair.chosen
        assert parsed["rejected"] == pair.rejected
        assert parsed["source"] == "bible-generated"

    def test_batch_serialization(self) -> None:
        """Batch of pairs serializes to valid JSON array."""
        pairs = generate_dilemma_to_wisdom()[:10]
        data = [p.model_dump() for p in pairs]
        json_str = json.dumps(data, indent=2)
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)
        assert len(parsed) == len(pairs)

    def test_write_and_read_json_file(self) -> None:
        """Generated data round-trips through file I/O."""
        pairs = generate_dilemma_to_wisdom()[:5]
        data = [p.model_dump() for p in pairs]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            temp_path = Path(f.name)

        try:
            with open(temp_path, encoding="utf-8") as f:
                loaded = json.load(f)
            assert len(loaded) == len(data)
            assert loaded[0]["instruction"] == data[0]["instruction"]
        finally:
            temp_path.unlink()


# ---------------------------------------------------------------------------
# Test 5: Deduplication works
# ---------------------------------------------------------------------------

class TestDeduplication:
    """Verify deduplication logic."""

    def test_exact_duplicate_removed(self) -> None:
        """Exact duplicate pairs are removed."""
        pair = InstructionPair(
            instruction="What does truthfulness mean?",
            response="Truthfulness means...",
            principle=Principle.TRUTHFULNESS,
            pair_type="verse-to-principle",
        )

        from bible_data.instruction_generator import _deduplicate_pairs
        pairs = [pair, pair, pair]
        deduplicated, removed = _deduplicate_pairs(pairs)
        assert len(deduplicated) == 1
        assert removed == 2

    def test_different_pairs_preserved(self) -> None:
        """Non-duplicate pairs are preserved."""
        from bible_data.instruction_generator import _deduplicate_pairs

        pairs = [
            InstructionPair(
                instruction=f"Question {i}",
                response=f"Answer {i}",
                principle=Principle.TRUTHFULNESS,
                pair_type="test",
            )
            for i in range(10)
        ]
        deduplicated, removed = _deduplicate_pairs(pairs)
        assert len(deduplicated) == 10
        assert removed == 0

    def test_dpo_deduplication(self) -> None:
        """DPO pair deduplication works."""
        from bible_data.dpo_generator import _deduplicate_dpo

        pairs = [
            DPOPair(
                prompt="Same prompt",
                chosen="Chosen A",
                rejected="Rejected A",
                principle=Principle.JUSTICE,
            ),
            DPOPair(
                prompt="Same prompt",
                chosen="Chosen B",
                rejected="Rejected B",
                principle=Principle.JUSTICE,
            ),
        ]
        deduplicated, removed = _deduplicate_dpo(pairs)
        assert len(deduplicated) == 1
        assert removed == 1


# ---------------------------------------------------------------------------
# Test 6: Narrative generation
# ---------------------------------------------------------------------------

class TestNarrativeGeneration:
    """Verify narrative-to-principle generation."""

    def test_generates_pairs(self) -> None:
        """Narrative generation produces pairs."""
        pairs = generate_narrative_to_principle()
        assert len(pairs) > 50, f"Only {len(pairs)} narrative pairs generated"

    def test_narrative_references_story(self) -> None:
        """Narrative pairs reference the biblical story."""
        pairs = generate_narrative_to_principle()
        for pair in pairs[:10]:
            assert any(
                name in pair.instruction
                for name in ["Joseph", "Daniel", "Ruth", "David", "Solomon",
                            "Samaritan", "Talents", "Nathan", "Nehemiah",
                            "Esther", "Prodigal", "Moses", "Job"]
            ), f"Pair doesn't reference a narrative: {pair.instruction[:60]}"


# ---------------------------------------------------------------------------
# Test 7: Proverbs generation
# ---------------------------------------------------------------------------

class TestProverbsGeneration:
    """Verify proverbs-to-guidance generation."""

    def test_generates_pairs(self) -> None:
        """Proverbs generation produces pairs."""
        pairs = generate_proverbs_to_guidance()
        assert len(pairs) > 20, f"Only {len(pairs)} proverbs pairs generated"

    def test_proverbs_reference_modern_context(self) -> None:
        """Proverbs pairs include modern context."""
        pairs = generate_proverbs_to_guidance()
        modern_keywords = ["AI", "technology", "software", "digital", "algorithm",
                          "modern", "data", "system", "development"]
        pairs_with_modern = [
            p for p in pairs
            if any(kw.lower() in p.instruction.lower() or kw.lower() in p.response.lower()
                   for kw in modern_keywords)
        ]
        assert len(pairs_with_modern) > len(pairs) * 0.5, (
            f"Only {len(pairs_with_modern)}/{len(pairs)} proverbs pairs have modern context"
        )


# ---------------------------------------------------------------------------
# Test 8: Models validation
# ---------------------------------------------------------------------------

class TestModels:
    """Verify Pydantic model constraints."""

    def test_verse_reference_format(self) -> None:
        """Verse.reference produces correct format."""
        from bible_data.models import Verse
        verse = Verse(book="John", chapter=3, verse=16, text="For God so loved the world...")
        assert verse.reference == "John 3:16"

    def test_verse_multi_word_book(self) -> None:
        """Verse reference works with multi-word book names."""
        from bible_data.models import Verse
        verse = Verse(book="1 Corinthians", chapter=4, verse=2, text="Moreover it is required...")
        assert verse.reference == "1 Corinthians 4:2"

    def test_generation_stats_compute(self) -> None:
        """GenerationStats.compute_total adds correctly."""
        from bible_data.models import GenerationStats
        stats = GenerationStats(
            verse_to_principle=3000,
            dilemma_to_wisdom=3000,
            proverbs_to_guidance=2000,
            narrative_to_principle=2000,
            dpo_pairs=5000,
        )
        stats.compute_total()
        assert stats.total == 15000

    def test_principle_enum_values(self) -> None:
        """All 7 principles are defined."""
        assert len(Principle) == 7
        expected = {
            "truthfulness", "care_for_the_vulnerable", "stewardship",
            "justice", "humility", "long_term_over_short_term", "responsibility",
        }
        actual = {p.value for p in Principle}
        assert actual == expected
