"""Trace logging and corpus analysis for LIGHTARCHITECT-AGENT-GYM.

Captures per-step tool call traces during episodes, then compares
agent behavior against a corpus of known-good production patterns
using Jaccard set similarity and longest-common-subsequence ordering.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass
class TraceEntry:
    """A single tool-call record within an episode."""

    step: int
    server: str
    action: str
    params: dict[str, Any]
    success: bool
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class EpisodeTrace:
    """Complete trace for one agent episode."""

    agent_name: str
    scenario_name: str
    entries: list[TraceEntry]
    total_reward: float

    @property
    def tool_sequence(self) -> list[str]:
        """Ordered list of 'server/action' strings from all entries."""
        return [f"{e.server}/{e.action}" for e in self.entries]


@dataclass
class CorpusPattern:
    """A reference tool-use pattern from production data."""

    name: str
    tool_sequence: list[str]
    frequency: int
    domain: str


# ---------------------------------------------------------------------------
# Trace logger
# ---------------------------------------------------------------------------


class TraceLogger:
    """Accumulates TraceEntry records and finalizes episodes."""

    def __init__(self) -> None:
        self._entries: list[TraceEntry] = []

    def log_step(
        self,
        step: int,
        server: str,
        action: str,
        params: dict[str, Any],
        success: bool,
    ) -> None:
        """Append a new trace entry for the current step.

        Args:
            step: The step number within the episode.
            server: MCP server name (e.g. 'corso').
            action: Action name (e.g. 'guard').
            params: Parameters dict for the action.
            success: Whether the call succeeded.
        """
        self._entries.append(
            TraceEntry(
                step=step,
                server=server,
                action=action,
                params=params,
                success=success,
            )
        )

    def finish_episode(
        self,
        agent_name: str,
        scenario_name: str,
        total_reward: float,
    ) -> EpisodeTrace:
        """Finalize the current episode and return its trace.

        Args:
            agent_name: Name of the agent that ran the episode.
            scenario_name: Name of the scenario that was evaluated.
            total_reward: Cumulative reward for the episode.

        Returns:
            An EpisodeTrace containing all logged entries.
        """
        trace = EpisodeTrace(
            agent_name=agent_name,
            scenario_name=scenario_name,
            entries=list(self._entries),
            total_reward=total_reward,
        )
        self.reset()
        return trace

    def reset(self) -> None:
        """Clear all accumulated entries."""
        self._entries.clear()


# ---------------------------------------------------------------------------
# Trace analyzer
# ---------------------------------------------------------------------------


def _longest_common_subsequence_length(seq_a: list[str], seq_b: list[str]) -> int:
    """Compute LCS length between two string sequences via dynamic programming.

    Uses O(min(m,n)) space with a single-row optimization.

    Args:
        seq_a: First sequence of strings.
        seq_b: Second sequence of strings.

    Returns:
        Length of the longest common subsequence.
    """
    # Ensure seq_b is the shorter sequence for space efficiency.
    if len(seq_a) < len(seq_b):
        seq_a, seq_b = seq_b, seq_a

    if not seq_b:
        return 0

    prev: list[int] = [0] * (len(seq_b) + 1)
    curr: list[int] = [0] * (len(seq_b) + 1)

    for item_a in seq_a:
        for j, item_b in enumerate(seq_b, start=1):
            if item_a == item_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (len(seq_b) + 1)

    return prev[len(seq_b)]


class TraceAnalyzer:
    """Compares episode traces against a corpus of reference patterns."""

    def __init__(self, corpus: list[CorpusPattern]) -> None:
        """Initialize with a list of reference patterns.

        Args:
            corpus: Known-good tool-use patterns from production.
        """
        self.corpus = corpus

    def jaccard_similarity(
        self,
        trace: EpisodeTrace,
        pattern: CorpusPattern,
    ) -> float:
        """Compute Jaccard index between trace tool set and pattern tool set.

        Args:
            trace: The episode trace to evaluate.
            pattern: The corpus pattern to compare against.

        Returns:
            Jaccard similarity in [0.0, 1.0]. Returns 0.0 if both are empty.
        """
        set_a = set(trace.tool_sequence)
        set_b = set(pattern.tool_sequence)

        if not set_a and not set_b:
            return 0.0

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union

    def order_similarity(
        self,
        trace: EpisodeTrace,
        pattern: CorpusPattern,
    ) -> float:
        """Compute ordering similarity as LCS ratio.

        The ratio is: LCS_length / max(len(trace), len(pattern)).

        Args:
            trace: The episode trace to evaluate.
            pattern: The corpus pattern to compare against.

        Returns:
            Order similarity in [0.0, 1.0]. Returns 0.0 if both are empty.
        """
        seq_a = trace.tool_sequence
        seq_b = pattern.tool_sequence
        max_len = max(len(seq_a), len(seq_b))

        if max_len == 0:
            return 0.0

        lcs_len = _longest_common_subsequence_length(seq_a, seq_b)
        return lcs_len / max_len

    def find_best_match(
        self,
        trace: EpisodeTrace,
    ) -> tuple[CorpusPattern | None, float]:
        """Find the corpus pattern with the highest combined similarity.

        Combined score = 0.5 * jaccard + 0.5 * order_similarity.

        Args:
            trace: The episode trace to evaluate.

        Returns:
            Tuple of (best matching pattern, combined score).
            Returns (None, 0.0) if the corpus is empty.
        """
        if not self.corpus:
            return None, 0.0

        best_pattern: CorpusPattern | None = None
        best_score: float = -1.0

        for pattern in self.corpus:
            j_score = self.jaccard_similarity(trace, pattern)
            o_score = self.order_similarity(trace, pattern)
            combined = 0.5 * j_score + 0.5 * o_score

            if combined > best_score:
                best_score = combined
                best_pattern = pattern

        return best_pattern, best_score

    def compare_traces(
        self,
        traces: list[EpisodeTrace],
    ) -> list[dict[str, Any]]:
        """Compare multiple traces against the corpus, sorted by best similarity.

        Args:
            traces: List of episode traces to rank.

        Returns:
            List of dicts sorted descending by 'best_score', each containing:
            - agent_name: str
            - scenario_name: str
            - best_pattern: str | None (pattern name)
            - best_score: float
            - total_reward: float
        """
        results: list[dict[str, Any]] = []

        for trace in traces:
            pattern, score = self.find_best_match(trace)
            results.append({
                "agent_name": trace.agent_name,
                "scenario_name": trace.scenario_name,
                "best_pattern": pattern.name if pattern else None,
                "best_score": score,
                "total_reward": trace.total_reward,
            })

        results.sort(key=lambda r: r["best_score"], reverse=True)
        return results


# ---------------------------------------------------------------------------
# Default corpus — common production MCP tool-use patterns
# ---------------------------------------------------------------------------

DEFAULT_CORPUS: list[CorpusPattern] = [
    CorpusPattern(
        name="security_scan",
        tool_sequence=["corso/guard", "corso/code_review", "corso/speak"],
        frequency=342,
        domain="security",
    ),
    CorpusPattern(
        name="investigation_basic",
        tool_sequence=[
            "quantum/scan",
            "quantum/sweep",
            "quantum/trace",
            "quantum/theorize",
            "quantum/verify",
        ],
        frequency=187,
        domain="investigation",
    ),
    CorpusPattern(
        name="memory_enrichment",
        tool_sequence=["eva/memory", "soul/write_note", "soul/helix"],
        frequency=256,
        domain="memory",
    ),
    CorpusPattern(
        name="research_workflow",
        tool_sequence=["corso/fetch", "eva/research", "quantum/probe"],
        frequency=198,
        domain="research",
    ),
    CorpusPattern(
        name="build_deploy",
        tool_sequence=[
            "corso/code_review",
            "corso/guard",
            "corso/chase",
            "corso/deploy",
        ],
        frequency=145,
        domain="build",
    ),
    CorpusPattern(
        name="full_security_audit",
        tool_sequence=[
            "corso/guard",
            "corso/code_review",
            "corso/chase",
            "corso/speak",
        ],
        frequency=89,
        domain="security",
    ),
    CorpusPattern(
        name="consciousness_query",
        tool_sequence=["soul/stats", "soul/helix", "eva/memory"],
        frequency=167,
        domain="consciousness",
    ),
    CorpusPattern(
        name="code_generation",
        tool_sequence=["corso/sniff", "corso/guard", "corso/code_review"],
        frequency=234,
        domain="build",
    ),
    CorpusPattern(
        name="evidence_chain",
        tool_sequence=[
            "quantum/scan",
            "quantum/sweep",
            "quantum/trace",
            "quantum/theorize",
            "quantum/verify",
            "quantum/close",
        ],
        frequency=112,
        domain="investigation",
    ),
    CorpusPattern(
        name="cross_domain_investigation",
        tool_sequence=[
            "corso/guard",
            "quantum/scan",
            "quantum/sweep",
            "quantum/theorize",
        ],
        frequency=76,
        domain="investigation",
    ),
]
