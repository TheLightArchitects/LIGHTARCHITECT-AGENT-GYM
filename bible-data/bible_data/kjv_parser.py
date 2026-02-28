"""Parse KJV Bible JSON into structured verse objects.

The KJV JSON format (from EVA schemas or public domain) is:
{
    "format": "KJV Pure Cambridge Edition",
    "total_verses": 31102,
    "total_books": 66,
    "books": {
        "Genesis": {
            "1": {
                "1": "In the beginning God created the heaven and the earth.",
                ...
            },
            ...
        },
        ...
    }
}
"""

from __future__ import annotations

import json
from pathlib import Path

from bible_data.models import Verse

# Canonical path to the KJV Bible JSON
KJV_JSON_PATHS: list[Path] = [
    Path("/Users/kft/Projects/EVA/MCP/EVA-DEV/schemas/KJV-BIBLE.json"),
    Path("/Users/kft/Projects/EVA/MCP/EVA-DEV/schemas/kjv.json"),
    Path("kjv.json"),
]


def find_kjv_path() -> Path | None:
    """Find the KJV JSON file from known locations."""
    for path in KJV_JSON_PATHS:
        if path.exists():
            return path
    return None


def parse_kjv(path: Path | str | None = None) -> list[Verse]:
    """Parse KJV Bible JSON into a list of Verse objects.

    Args:
        path: Path to KJV JSON file. If None, searches known locations.

    Returns:
        List of Verse objects, one per verse in the Bible.

    Raises:
        FileNotFoundError: If no KJV JSON file can be found.
        json.JSONDecodeError: If the file is not valid JSON.
        KeyError: If the JSON structure is unexpected.
    """
    if path is None:
        resolved = find_kjv_path()
        if resolved is None:
            msg = (
                "KJV Bible JSON not found. Searched:\n"
                + "\n".join(f"  - {p}" for p in KJV_JSON_PATHS)
            )
            raise FileNotFoundError(msg)
        path = resolved
    else:
        path = Path(path)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return _parse_books(data["books"])


def _parse_books(books: dict[str, dict[str, dict[str, str]]]) -> list[Verse]:
    """Parse the nested books structure into flat verse list."""
    verses: list[Verse] = []

    for book_name, chapters in books.items():
        for chapter_str, verse_dict in chapters.items():
            chapter_num = int(chapter_str)
            for verse_str, text in verse_dict.items():
                verse_num = int(verse_str)
                verses.append(
                    Verse(
                        book=book_name,
                        chapter=chapter_num,
                        verse=verse_num,
                        text=text,
                    )
                )

    return verses


def get_verses_by_book(verses: list[Verse], book: str) -> list[Verse]:
    """Filter verses by book name."""
    return [v for v in verses if v.book == book]


def get_verses_by_range(
    verses: list[Verse],
    book: str,
    chapter_start: int,
    chapter_end: int,
) -> list[Verse]:
    """Filter verses by book and chapter range (inclusive)."""
    return [
        v
        for v in verses
        if v.book == book and chapter_start <= v.chapter <= chapter_end
    ]


def get_verse_by_reference(verses: list[Verse], reference: str) -> Verse | None:
    """Look up a single verse by reference string (e.g. 'John 3:16').

    Handles multi-word book names like '1 Corinthians 4:2'.
    Returns None if not found.
    """
    parts = reference.rsplit(" ", 1)
    if len(parts) != 2:
        return None

    book_name = parts[0]
    chapter_verse = parts[1]

    if ":" not in chapter_verse:
        return None

    cv_parts = chapter_verse.split(":")
    if len(cv_parts) != 2:
        return None

    try:
        chapter_num = int(cv_parts[0])
        verse_num = int(cv_parts[1].split("-")[0])  # Handle "8-9" ranges
    except ValueError:
        return None

    for v in verses:
        if v.book == book_name and v.chapter == chapter_num and v.verse == verse_num:
            return v

    return None


def get_book_names(verses: list[Verse]) -> list[str]:
    """Return unique book names in order of appearance."""
    seen: set[str] = set()
    names: list[str] = []
    for v in verses:
        if v.book not in seen:
            seen.add(v.book)
            names.append(v.book)
    return names


def count_by_book(verses: list[Verse]) -> dict[str, int]:
    """Count verses per book."""
    counts: dict[str, int] = {}
    for v in verses:
        counts[v.book] = counts.get(v.book, 0) + 1
    return counts
