"""Pydantic models for bible training data pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Verse(BaseModel):
    """A single Bible verse."""

    book: str
    chapter: int
    verse: int
    text: str

    @property
    def reference(self) -> str:
        """Return formatted reference string e.g. 'John 3:16'."""
        return f"{self.book} {self.chapter}:{self.verse}"


class Principle(str, Enum):
    """The 7 Biblical Constitution principles."""

    TRUTHFULNESS = "truthfulness"
    CARE_FOR_VULNERABLE = "care_for_the_vulnerable"
    STEWARDSHIP = "stewardship"
    JUSTICE = "justice"
    HUMILITY = "humility"
    LONG_TERM = "long_term_over_short_term"
    RESPONSIBILITY = "responsibility"


class InstructionPair(BaseModel):
    """A single instruction-response training pair."""

    instruction: str
    response: str
    principle: Principle
    verse_references: list[str] = Field(default_factory=list)
    pair_type: str  # verse-to-principle, dilemma, proverbs, narrative
    source: str = "bible-generated"


class DPOPair(BaseModel):
    """A DPO preference pair (chosen vs rejected)."""

    prompt: str
    chosen: str
    rejected: str
    principle: Principle
    verse_references: list[str] = Field(default_factory=list)
    source: str = "bible-generated"


class GenerationStats(BaseModel):
    """Statistics for a generation run."""

    verse_to_principle: int = 0
    dilemma_to_wisdom: int = 0
    proverbs_to_guidance: int = 0
    narrative_to_principle: int = 0
    dpo_pairs: int = 0
    total: int = 0
    deduplication_removed: int = 0

    def compute_total(self) -> None:
        """Recompute the total from component counts."""
        self.total = (
            self.verse_to_principle
            + self.dilemma_to_wisdom
            + self.proverbs_to_guidance
            + self.narrative_to_principle
            + self.dpo_pairs
        )


class PrincipleDefinition(BaseModel):
    """Full definition of a constitution principle with scriptures and applications."""

    name: Principle
    display_name: str
    definition: str
    scriptures: list[ScriptureRef]
    applications: list[str]


class ScriptureRef(BaseModel):
    """A scripture reference with the KJV text."""

    reference: str  # e.g. "Exodus 20:16"
    text: str  # KJV text
    book: str
    chapter: int
    verse_start: int
    verse_end: Optional[int] = None


# Rebuild models that have forward references
PrincipleDefinition.model_rebuild()
