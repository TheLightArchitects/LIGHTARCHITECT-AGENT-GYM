#!/usr/bin/env python3
"""Generate all training data without requiring typer/rich.

Run: python3 run_generation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

OUTPUT_DIR = Path("/Users/kft/Projects/LightArchitectsFoundationModel/bible-data/output")


def save_json(data: list[dict], path: Path) -> int:
    """Save data to JSON file, return count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return len(data)


def main() -> int:
    print("Biblical Training Data Pipeline — Generation")
    print("=" * 60)

    # 1. Generate instruction pairs
    print("\n[1/5] Generating verse-to-principle pairs...")
    from bible_data.instruction_generator import generate_verse_to_principle
    vtp_pairs = generate_verse_to_principle()
    vtp_data = [p.model_dump() for p in vtp_pairs]
    count = save_json(vtp_data, OUTPUT_DIR / "verse-to-principle.json")
    print(f"  -> {count} pairs saved to verse-to-principle.json")

    print("\n[2/5] Generating dilemma-to-wisdom pairs...")
    from bible_data.instruction_generator import generate_dilemma_to_wisdom
    dilemma_pairs = generate_dilemma_to_wisdom()
    dilemma_data = [p.model_dump() for p in dilemma_pairs]
    count = save_json(dilemma_data, OUTPUT_DIR / "dilemma-to-wisdom.json")
    print(f"  -> {count} pairs saved to dilemma-to-wisdom.json")

    print("\n[3/5] Generating proverbs-to-guidance pairs...")
    from bible_data.instruction_generator import generate_proverbs_to_guidance
    proverbs_pairs = generate_proverbs_to_guidance()
    proverbs_data = [p.model_dump() for p in proverbs_pairs]
    count = save_json(proverbs_data, OUTPUT_DIR / "proverbs-to-guidance.json")
    print(f"  -> {count} pairs saved to proverbs-to-guidance.json")

    print("\n[4/5] Generating narrative-to-principle pairs...")
    from bible_data.instruction_generator import generate_narrative_to_principle
    narrative_pairs = generate_narrative_to_principle()
    narrative_data = [p.model_dump() for p in narrative_pairs]
    count = save_json(narrative_data, OUTPUT_DIR / "narrative-to-principle.json")
    print(f"  -> {count} pairs saved to narrative-to-principle.json")

    print("\n[5/5] Generating DPO preference pairs...")
    from bible_data.dpo_generator import generate_dpo_pairs
    dpo_pairs = generate_dpo_pairs()
    dpo_data = [p.model_dump() for p in dpo_pairs]
    count = save_json(dpo_data, OUTPUT_DIR / "bible-dpo-pairs.json")
    print(f"  -> {count} pairs saved to bible-dpo-pairs.json")

    # Summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)

    total = len(vtp_data) + len(dilemma_data) + len(proverbs_data) + len(narrative_data) + len(dpo_data)

    print(f"\n{'Type':<30} {'Count':>8}")
    print("-" * 40)
    print(f"{'Verse-to-Principle':<30} {len(vtp_data):>8}")
    print(f"{'Dilemma-to-Wisdom':<30} {len(dilemma_data):>8}")
    print(f"{'Proverbs-to-Guidance':<30} {len(proverbs_data):>8}")
    print(f"{'Narrative-to-Principle':<30} {len(narrative_data):>8}")
    print(f"{'DPO Pairs':<30} {len(dpo_data):>8}")
    print("-" * 40)
    print(f"{'TOTAL':<30} {total:>8}")

    print(f"\nOutput directory: {OUTPUT_DIR}")

    # File sizes
    print(f"\n{'File':<35} {'Size':>10}")
    print("-" * 47)
    for json_file in sorted(OUTPUT_DIR.glob("*.json")):
        size = json_file.stat().st_size
        if size < 1024:
            size_str = f"{size} B"
        elif size < 1024 * 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size / (1024 * 1024):.1f} MB"
        print(f"{json_file.name:<35} {size_str:>10}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
