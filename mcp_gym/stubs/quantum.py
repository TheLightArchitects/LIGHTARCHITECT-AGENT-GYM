"""QUANTUM MCP server stub — 13 actions.

Provides deterministic mock responses for the QUANTUM Investigation Toolkit.
Actions: scan, sweep, trace, probe, theorize, verify, close, quick,
         research, helix, discover, list, workflow.
"""

from __future__ import annotations

from typing import Any

from mcp_gym.mock_surface import MockMCPSurface


def _scan(params: dict[str, Any]) -> dict[str, Any]:
    """Triage scan — initial evidence collection."""
    return {
        "phase": "scan",
        "findings": [
            {"type": "anomaly", "description": "Unusual pattern detected", "severity": "medium"},
        ],
        "triage_score": 6.5,
        "recommended_next": "sweep",
    }


def _sweep(params: dict[str, Any]) -> dict[str, Any]:
    """Evidence collection sweep."""
    return {
        "phase": "sweep",
        "evidence": [
            {"id": "EV-001", "type": "log_entry", "source": "system.log", "timestamp": "2026-02-27T10:00:00Z"},
            {"id": "EV-002", "type": "config_change", "source": "settings.json", "timestamp": "2026-02-27T09:30:00Z"},
        ],
        "evidence_count": 2,
    }


def _trace(params: dict[str, Any]) -> dict[str, Any]:
    """Pattern forensics — trace execution chains."""
    return {
        "phase": "trace",
        "patterns": [
            {"name": "call_chain", "nodes": 5, "depth": 3, "suspicious": False},
        ],
        "correlations": [],
    }


def _probe(params: dict[str, Any]) -> dict[str, Any]:
    """Multi-source research probe."""
    query = params.get("query", "")
    return {
        "phase": "probe",
        "query": query,
        "sources_checked": ["helix", "documentation", "web"],
        "findings": [
            {"source": "helix", "result": f"Related entry found for: {query[:40]}"},
        ],
    }


def _theorize(params: dict[str, Any]) -> dict[str, Any]:
    """Hypothesis generation."""
    return {
        "phase": "theorize",
        "hypotheses": [
            {
                "id": "H-001",
                "statement": "Configuration drift caused the anomaly",
                "confidence": 0.75,
                "evidence_refs": ["EV-001", "EV-002"],
            },
        ],
    }


def _verify(params: dict[str, Any]) -> dict[str, Any]:
    """Solution validation."""
    return {
        "phase": "verify",
        "hypothesis_tested": "H-001",
        "result": "confirmed",
        "confidence": 0.92,
        "tests_passed": 3,
        "tests_total": 3,
    }


def _close(params: dict[str, Any]) -> dict[str, Any]:
    """Deliverable generation — close investigation."""
    return {
        "phase": "close",
        "deliverable": {
            "title": "Investigation Report",
            "root_cause": "Configuration drift",
            "resolution": "Applied corrective config",
            "evidence_chain": ["EV-001", "EV-002"],
            "confidence": 0.92,
        },
    }


def _quick(params: dict[str, Any]) -> dict[str, Any]:
    """Abbreviated investigation (scan + theorize + verify in one pass)."""
    return {
        "phase": "quick",
        "summary": "Quick analysis complete",
        "root_cause": "Minor configuration issue",
        "confidence": 0.80,
        "recommendation": "Review and update config",
    }


def _research(params: dict[str, Any]) -> dict[str, Any]:
    """Web search + Helix + synthesis."""
    query = params.get("query", "")
    return {
        "phase": "research",
        "query": query,
        "synthesis": f"Research findings for: {query[:60]}",
        "sources": ["web", "helix", "documentation"],
        "confidence": 0.85,
    }


def _helix(params: dict[str, Any]) -> dict[str, Any]:
    """Investigation-aware knowledge queries."""
    return {
        "phase": "helix",
        "entries": [
            {
                "title": "Prior investigation",
                "significance": 8.0,
                "relevance": 0.88,
            },
        ],
        "count": 1,
    }


def _discover(params: dict[str, Any]) -> dict[str, Any]:
    """Find tools by query."""
    query = params.get("query", "")
    return {
        "phase": "discover",
        "query": query,
        "tools": [
            {"name": "scan", "description": "Triage scan", "relevance": 0.9},
            {"name": "sweep", "description": "Evidence collection", "relevance": 0.85},
        ],
    }


def _list(params: dict[str, Any]) -> dict[str, Any]:
    """Show all available actions."""
    return {
        "phase": "list",
        "actions": [
            "scan", "sweep", "trace", "probe", "theorize",
            "verify", "close", "quick", "research", "helix",
            "discover", "list", "workflow",
        ],
        "count": 13,
    }


def _workflow(params: dict[str, Any]) -> dict[str, Any]:
    """Run workflow template."""
    template = params.get("template", "full-investigation")
    return {
        "phase": "workflow",
        "template": template,
        "status": "complete",
        "phases_executed": ["scan", "sweep", "trace", "theorize", "verify", "close"],
        "duration_ms": 1500,
    }


# --- All 13 QUANTUM actions ---

QUANTUM_ACTIONS: dict[str, Any] = {
    "scan": _scan,
    "sweep": _sweep,
    "trace": _trace,
    "probe": _probe,
    "theorize": _theorize,
    "verify": _verify,
    "close": _close,
    "quick": _quick,
    "research": _research,
    "helix": _helix,
    "discover": _discover,
    "list": _list,
    "workflow": _workflow,
}


def register_all(surface: MockMCPSurface) -> None:
    """Register all 13 QUANTUM action handlers on the given surface.

    Args:
        surface: The MockMCPSurface to register handlers on.
    """
    for action_name, handler in QUANTUM_ACTIONS.items():
        surface.register(f"quantum/{action_name}", handler)
