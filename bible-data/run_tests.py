#!/usr/bin/env python3
"""Minimal test runner that doesn't depend on pytest/typer/rich.

Run: python3 run_tests.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


def test_all_seven_principles_defined() -> None:
    """Test 1: Constitution has all 7 principles."""
    from bible_data.constitution_data import PRINCIPLE_DEFINITIONS, PRINCIPLE_SCRIPTURES
    from bible_data.models import Principle

    for principle in Principle:
        assert principle in PRINCIPLE_DEFINITIONS, f"Missing definition: {principle.value}"
        assert principle in PRINCIPLE_SCRIPTURES, f"Missing scriptures: {principle.value}"
        scriptures = PRINCIPLE_SCRIPTURES[principle]
        assert len(scriptures) >= 3, f"{principle.value} has only {len(scriptures)} scriptures"


def test_verse_to_principle_generates_valid_pairs() -> None:
    """Test 2: Verse-to-principle generation produces valid pairs."""
    from bible_data.instruction_generator import generate_dilemma_to_wisdom
    from bible_data.models import Principle

    pairs = generate_dilemma_to_wisdom()
    assert len(pairs) > 0, "No dilemma pairs generated"

    for pair in pairs[:10]:
        assert pair.instruction, "Empty instruction"
        assert pair.response, "Empty response"
        assert pair.principle in Principle, f"Invalid principle: {pair.principle}"
        assert pair.source == "bible-generated", f"Wrong source: {pair.source}"


def test_dpo_generator_creates_chosen_and_rejected() -> None:
    """Test 3: DPO generator creates both chosen and rejected."""
    from bible_data.dpo_generator import generate_dpo_pairs

    pairs = generate_dpo_pairs()
    assert len(pairs) > 0, "No DPO pairs generated"

    for pair in pairs[:20]:
        assert pair.prompt, "Empty prompt"
        assert pair.chosen, "Empty chosen"
        assert pair.rejected, "Empty rejected"
        assert pair.chosen != pair.rejected, "Chosen equals rejected"


def test_output_json_is_valid() -> None:
    """Test 4: Output JSON is valid."""
    from bible_data.instruction_generator import generate_dilemma_to_wisdom
    from bible_data.models import InstructionPair, Principle

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

    # Batch serialization
    pairs = generate_dilemma_to_wisdom()[:5]
    batch_data = [p.model_dump() for p in pairs]
    batch_json = json.dumps(batch_data, indent=2)
    batch_parsed = json.loads(batch_json)
    assert isinstance(batch_parsed, list)
    assert len(batch_parsed) == 5


def test_deduplication_works() -> None:
    """Test 5: Deduplication removes exact duplicates."""
    from bible_data.instruction_generator import _deduplicate_pairs
    from bible_data.models import InstructionPair, Principle

    pair = InstructionPair(
        instruction="What does truthfulness mean?",
        response="Truthfulness means...",
        principle=Principle.TRUTHFULNESS,
        pair_type="verse-to-principle",
    )

    pairs = [pair, pair, pair]
    deduplicated, removed = _deduplicate_pairs(pairs)
    assert len(deduplicated) == 1, f"Expected 1, got {len(deduplicated)}"
    assert removed == 2, f"Expected 2 removed, got {removed}"

    # Different pairs preserved
    different = [
        InstructionPair(
            instruction=f"Question {i}",
            response=f"Answer {i}",
            principle=Principle.TRUTHFULNESS,
            pair_type="test",
        )
        for i in range(10)
    ]
    deduplicated2, removed2 = _deduplicate_pairs(different)
    assert len(deduplicated2) == 10, f"Expected 10, got {len(deduplicated2)}"
    assert removed2 == 0, f"Expected 0 removed, got {removed2}"


def test_narrative_generation() -> None:
    """Test 6: Narrative generation produces pairs."""
    from bible_data.instruction_generator import generate_narrative_to_principle

    pairs = generate_narrative_to_principle()
    assert len(pairs) > 50, f"Only {len(pairs)} narrative pairs"

    names = ["Joseph", "Daniel", "Ruth", "David", "Solomon",
             "Samaritan", "Talents", "Nathan", "Nehemiah",
             "Esther", "Prodigal", "Moses", "Job"]
    for pair in pairs[:10]:
        assert any(n in pair.instruction for n in names), (
            f"No narrative name in: {pair.instruction[:60]}"
        )


def test_proverbs_generation() -> None:
    """Test 7: Proverbs generation produces pairs."""
    from bible_data.instruction_generator import generate_proverbs_to_guidance

    pairs = generate_proverbs_to_guidance()
    assert len(pairs) > 20, f"Only {len(pairs)} proverbs pairs"


def test_models_validation() -> None:
    """Test 8: Pydantic models validate correctly."""
    from bible_data.models import GenerationStats, Principle, Verse

    verse = Verse(book="John", chapter=3, verse=16, text="For God so loved...")
    assert verse.reference == "John 3:16"

    verse2 = Verse(book="1 Corinthians", chapter=4, verse=2, text="Moreover...")
    assert verse2.reference == "1 Corinthians 4:2"

    stats = GenerationStats(
        verse_to_principle=3000,
        dilemma_to_wisdom=3000,
        proverbs_to_guidance=2000,
        narrative_to_principle=2000,
        dpo_pairs=5000,
    )
    stats.compute_total()
    assert stats.total == 15000

    assert len(Principle) == 7


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> int:
    tests = [
        test_all_seven_principles_defined,
        test_verse_to_principle_generates_valid_pairs,
        test_dpo_generator_creates_chosen_and_rejected,
        test_output_json_is_valid,
        test_deduplication_works,
        test_narrative_generation,
        test_proverbs_generation,
        test_models_validation,
    ]

    passed = 0
    failed = 0
    errors: list[str] = []

    for test_func in tests:
        name = test_func.__name__
        try:
            test_func()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as e:
            failed += 1
            errors.append(f"  FAIL  {name}: {e}")
            print(f"  FAIL  {name}: {e}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")

    if errors:
        print("\nFailures:")
        for err in errors:
            print(err)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
