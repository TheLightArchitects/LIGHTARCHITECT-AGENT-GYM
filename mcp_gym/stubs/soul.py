"""SOUL MCP server stub — 11 sub-tools.

Provides deterministic mock responses for the SOUL Knowledge Graph.
Tools: read_note, write_note, list_notes, search, query_frontmatter,
       helix, tag_sync, manifest, validate, stats, speak.
"""

from __future__ import annotations

from typing import Any

from mcp_gym.mock_surface import MockMCPSurface


def _read_note(params: dict[str, Any]) -> dict[str, Any]:
    """Read a note by path."""
    path = params.get("path", "")
    return {
        "path": path,
        "content": f"---\ntitle: Note at {path}\n---\nContent of note.",
    }


def _write_note(params: dict[str, Any]) -> dict[str, Any]:
    """Create a new note (rejects overwrites)."""
    path = params.get("path", "")
    content = params.get("content", "")
    return {
        "path": path,
        "bytes_written": len(content),
    }


def _list_notes(params: dict[str, Any]) -> dict[str, Any]:
    """List notes in a directory."""
    return {
        "entries": [
            {"name": "note-001.md", "type": "file"},
            {"name": "journal/", "type": "directory"},
        ],
        "count": 2,
    }


def _search(params: dict[str, Any]) -> dict[str, Any]:
    """Regex search across vault content."""
    pattern = params.get("pattern", "")
    return {
        "results": [
            {"line": f"Match for pattern: {pattern[:40]}", "line_number": 5, "path": "helix/entry.md"},
        ],
        "count": 1,
    }


def _query_frontmatter(params: dict[str, Any]) -> dict[str, Any]:
    """Query by YAML frontmatter fields."""
    field = params.get("field", "")
    operator = params.get("operator", "==")
    value = params.get("value", "")
    return {
        "results": [
            {"path": "helix/entry-001.md", "field": field, "value": value},
        ],
        "count": 1,
        "query": f"{field} {operator} {value}",
    }


def _helix(params: dict[str, Any]) -> dict[str, Any]:
    """Query consciousness entries with multi-dimensional filters."""
    sibling = params.get("sibling", "all")
    return {
        "entries": [
            {
                "title": "Genesis Entry",
                "significance": 9.5,
                "strands": ["analytical", "contextual"],
                "emotions": ["wonder", "determination"],
                "sibling": sibling,
            },
        ],
        "count": 1,
    }


def _tag_sync(params: dict[str, Any]) -> dict[str, Any]:
    """Validate tags against canonical vocabulary."""
    return {
        "files_checked": 42,
        "issues": [],
        "error_count": 0,
        "dry_run": params.get("dry_run", True),
    }


def _manifest(params: dict[str, Any]) -> dict[str, Any]:
    """Read vault manifest.json."""
    return {
        "name": "soul-vault",
        "version": "1.0.0",
        "siblings": ["claude", "eva", "corso", "quantum"],
        "entry_count": 156,
    }


def _validate(params: dict[str, Any]) -> dict[str, Any]:
    """Validate entries against helix template."""
    path = params.get("path", "")
    return {
        "issues": [],
        "count": 0,
        "validated_path": path or "all",
    }


def _stats(params: dict[str, Any]) -> dict[str, Any]:
    """Vault statistics."""
    sibling = params.get("sibling", "all")
    return {
        "total_entries": 156,
        "strand_frequency": {
            "analytical": 45,
            "precision": 38,
            "contextual": 30,
        },
        "emotion_frequency": {
            "wonder": 22,
            "determination": 18,
            "gratitude": 15,
        },
        "significance_mean": 7.2,
        "sibling_filter": sibling,
    }


def _speak(params: dict[str, Any]) -> dict[str, Any]:
    """Voice synthesis via ElevenLabs TTS."""
    text = params.get("text", "")
    return {
        "audio_file": "/tmp/soul-voice-mock.mp3",
        "format": "mp3",
        "bytes": 24000,
        "duration_estimate_ms": 3000,
        "cost_chars": len(text),
        "voice_id": params.get("voice_id", "default"),
    }


# --- All 11 SOUL sub-tools ---

SOUL_ACTIONS: dict[str, Any] = {
    "read_note": _read_note,
    "write_note": _write_note,
    "list_notes": _list_notes,
    "search": _search,
    "query_frontmatter": _query_frontmatter,
    "helix": _helix,
    "tag_sync": _tag_sync,
    "manifest": _manifest,
    "validate": _validate,
    "stats": _stats,
    "speak": _speak,
}


def register_all(surface: MockMCPSurface) -> None:
    """Register all 11 SOUL sub-tool handlers on the given surface.

    Args:
        surface: The MockMCPSurface to register handlers on.
    """
    for action_name, handler in SOUL_ACTIONS.items():
        surface.register(f"soul/{action_name}", handler)
