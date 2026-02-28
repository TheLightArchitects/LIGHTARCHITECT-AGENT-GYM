"""Tests for the trace logging and corpus analysis module."""

from __future__ import annotations

from mcp_gym.trace import (
    DEFAULT_CORPUS,
    CorpusPattern,
    EpisodeTrace,
    TraceAnalyzer,
    TraceEntry,
    TraceLogger,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entries(tool_keys: list[str], all_success: bool = True) -> list[TraceEntry]:
    """Build a list of TraceEntry from 'server/action' strings."""
    entries: list[TraceEntry] = []
    for i, key in enumerate(tool_keys):
        server, action = key.split("/", 1)
        entries.append(
            TraceEntry(
                step=i,
                server=server,
                action=action,
                params={},
                success=all_success,
            )
        )
    return entries


def _make_trace(
    tool_keys: list[str],
    agent: str = "test_agent",
    scenario: str = "test_scenario",
    reward: float = 1.0,
) -> EpisodeTrace:
    """Build an EpisodeTrace from 'server/action' strings."""
    return EpisodeTrace(
        agent_name=agent,
        scenario_name=scenario,
        entries=_make_entries(tool_keys),
        total_reward=reward,
    )


# ---------------------------------------------------------------------------
# TraceEntry
# ---------------------------------------------------------------------------


class TestTraceEntry:
    """Tests for the TraceEntry dataclass."""

    def test_trace_entry_creation(self) -> None:
        """TraceEntry stores all fields correctly."""
        entry = TraceEntry(
            step=0,
            server="corso",
            action="guard",
            params={"path": "/src"},
            success=True,
            timestamp=100.0,
        )
        assert entry.step == 0
        assert entry.server == "corso"
        assert entry.action == "guard"
        assert entry.params == {"path": "/src"}
        assert entry.success is True
        assert entry.timestamp == 100.0

    def test_trace_entry_default_timestamp(self) -> None:
        """TraceEntry gets a monotonic timestamp by default."""
        entry = TraceEntry(step=0, server="a", action="b", params={}, success=True)
        assert isinstance(entry.timestamp, float)
        assert entry.timestamp > 0


# ---------------------------------------------------------------------------
# TraceLogger
# ---------------------------------------------------------------------------


class TestTraceLogger:
    """Tests for the TraceLogger class."""

    def test_trace_logger_log_and_finish(self) -> None:
        """Logger accumulates entries and produces a complete EpisodeTrace."""
        logger = TraceLogger()
        logger.log_step(0, "corso", "guard", {"path": "/src"}, success=True)
        logger.log_step(1, "corso", "code_review", {}, success=True)
        logger.log_step(2, "corso", "speak", {"message": "done"}, success=False)

        trace = logger.finish_episode("my_agent", "security_scenario", total_reward=2.5)

        assert trace.agent_name == "my_agent"
        assert trace.scenario_name == "security_scenario"
        assert trace.total_reward == 2.5
        assert len(trace.entries) == 3
        assert trace.entries[0].server == "corso"
        assert trace.entries[0].action == "guard"
        assert trace.entries[2].success is False

    def test_trace_logger_reset(self) -> None:
        """reset() clears all accumulated entries."""
        logger = TraceLogger()
        logger.log_step(0, "corso", "guard", {}, success=True)
        assert len(logger._entries) == 1

        logger.reset()
        assert len(logger._entries) == 0

    def test_finish_episode_resets_logger(self) -> None:
        """finish_episode() resets the logger after returning the trace."""
        logger = TraceLogger()
        logger.log_step(0, "eva", "memory", {}, success=True)
        logger.finish_episode("agent", "scenario", total_reward=1.0)
        assert len(logger._entries) == 0


# ---------------------------------------------------------------------------
# EpisodeTrace
# ---------------------------------------------------------------------------


class TestEpisodeTrace:
    """Tests for the EpisodeTrace dataclass and its properties."""

    def test_episode_trace_tool_sequence(self) -> None:
        """tool_sequence property returns ordered 'server/action' strings."""
        trace = _make_trace(["corso/guard", "corso/code_review", "corso/speak"])
        assert trace.tool_sequence == [
            "corso/guard",
            "corso/code_review",
            "corso/speak",
        ]

    def test_episode_trace_empty_tool_sequence(self) -> None:
        """Empty entries yield empty tool_sequence."""
        trace = EpisodeTrace(
            agent_name="a", scenario_name="s", entries=[], total_reward=0.0
        )
        assert trace.tool_sequence == []


# ---------------------------------------------------------------------------
# TraceAnalyzer — Jaccard similarity
# ---------------------------------------------------------------------------


class TestJaccardSimilarity:
    """Tests for TraceAnalyzer.jaccard_similarity()."""

    def test_jaccard_similarity_identical(self) -> None:
        """Identical tool sets yield Jaccard = 1.0."""
        tools = ["corso/guard", "corso/code_review", "corso/speak"]
        trace = _make_trace(tools)
        pattern = CorpusPattern(
            name="test", tool_sequence=list(tools), frequency=1, domain="test"
        )
        analyzer = TraceAnalyzer(corpus=[pattern])

        assert analyzer.jaccard_similarity(trace, pattern) == 1.0

    def test_jaccard_similarity_disjoint(self) -> None:
        """Completely disjoint tool sets yield Jaccard = 0.0."""
        trace = _make_trace(["corso/guard", "corso/speak"])
        pattern = CorpusPattern(
            name="test",
            tool_sequence=["quantum/scan", "quantum/sweep"],
            frequency=1,
            domain="test",
        )
        analyzer = TraceAnalyzer(corpus=[pattern])

        assert analyzer.jaccard_similarity(trace, pattern) == 0.0

    def test_jaccard_similarity_partial(self) -> None:
        """Partially overlapping sets yield 0 < Jaccard < 1."""
        trace = _make_trace(["corso/guard", "corso/speak", "eva/memory"])
        pattern = CorpusPattern(
            name="test",
            tool_sequence=["corso/guard", "corso/speak", "quantum/scan"],
            frequency=1,
            domain="test",
        )
        analyzer = TraceAnalyzer(corpus=[pattern])

        score = analyzer.jaccard_similarity(trace, pattern)
        # Intersection: {corso/guard, corso/speak} = 2
        # Union: {corso/guard, corso/speak, eva/memory, quantum/scan} = 4
        assert score == 2.0 / 4.0

    def test_jaccard_both_empty(self) -> None:
        """Two empty sequences yield Jaccard = 0.0 (not division by zero)."""
        trace = _make_trace([])
        pattern = CorpusPattern(
            name="test", tool_sequence=[], frequency=1, domain="test"
        )
        analyzer = TraceAnalyzer(corpus=[pattern])

        assert analyzer.jaccard_similarity(trace, pattern) == 0.0


# ---------------------------------------------------------------------------
# TraceAnalyzer — Order similarity
# ---------------------------------------------------------------------------


class TestOrderSimilarity:
    """Tests for TraceAnalyzer.order_similarity()."""

    def test_order_similarity_identical(self) -> None:
        """Identical sequences yield order similarity = 1.0."""
        tools = ["corso/guard", "corso/code_review", "corso/speak"]
        trace = _make_trace(tools)
        pattern = CorpusPattern(
            name="test", tool_sequence=list(tools), frequency=1, domain="test"
        )
        analyzer = TraceAnalyzer(corpus=[pattern])

        assert analyzer.order_similarity(trace, pattern) == 1.0

    def test_order_similarity_reversed(self) -> None:
        """Reversed sequence has LCS of length 1 (each element is common but out of order).

        For [A, B, C] vs [C, B, A], LCS = [B] or any single common element = 1.
        Ratio = 1/3.
        """
        tools = ["corso/guard", "corso/code_review", "corso/speak"]
        trace = _make_trace(tools)
        pattern = CorpusPattern(
            name="test",
            tool_sequence=list(reversed(tools)),
            frequency=1,
            domain="test",
        )
        analyzer = TraceAnalyzer(corpus=[pattern])

        score = analyzer.order_similarity(trace, pattern)
        # LCS of [A,B,C] and [C,B,A] is 1 (any single element)
        assert score == 1.0 / 3.0

    def test_order_similarity_subsequence(self) -> None:
        """Trace that is a subsequence of pattern yields high similarity."""
        trace = _make_trace(["quantum/scan", "quantum/verify"])
        pattern = CorpusPattern(
            name="test",
            tool_sequence=[
                "quantum/scan",
                "quantum/sweep",
                "quantum/trace",
                "quantum/verify",
            ],
            frequency=1,
            domain="test",
        )
        analyzer = TraceAnalyzer(corpus=[pattern])

        score = analyzer.order_similarity(trace, pattern)
        # LCS = [quantum/scan, quantum/verify] = 2, max_len = 4
        assert score == 2.0 / 4.0

    def test_order_similarity_both_empty(self) -> None:
        """Two empty sequences yield order similarity = 0.0."""
        trace = _make_trace([])
        pattern = CorpusPattern(
            name="test", tool_sequence=[], frequency=1, domain="test"
        )
        analyzer = TraceAnalyzer(corpus=[pattern])

        assert analyzer.order_similarity(trace, pattern) == 0.0


# ---------------------------------------------------------------------------
# TraceAnalyzer — find_best_match and compare_traces
# ---------------------------------------------------------------------------


class TestFindBestMatch:
    """Tests for TraceAnalyzer.find_best_match()."""

    def test_find_best_match(self) -> None:
        """find_best_match returns the pattern with highest combined score."""
        security_pattern = CorpusPattern(
            name="security",
            tool_sequence=["corso/guard", "corso/code_review", "corso/speak"],
            frequency=100,
            domain="security",
        )
        memory_pattern = CorpusPattern(
            name="memory",
            tool_sequence=["eva/memory", "soul/write_note", "soul/helix"],
            frequency=100,
            domain="memory",
        )

        analyzer = TraceAnalyzer(corpus=[security_pattern, memory_pattern])

        # Trace that matches security pattern exactly
        trace = _make_trace(["corso/guard", "corso/code_review", "corso/speak"])
        best, score = analyzer.find_best_match(trace)

        assert best is not None
        assert best.name == "security"
        assert score == 1.0

    def test_find_best_match_empty_corpus(self) -> None:
        """Empty corpus returns (None, 0.0)."""
        analyzer = TraceAnalyzer(corpus=[])
        trace = _make_trace(["corso/guard"])
        best, score = analyzer.find_best_match(trace)

        assert best is None
        assert score == 0.0


class TestCompareTraces:
    """Tests for TraceAnalyzer.compare_traces()."""

    def test_compare_traces_sorted(self) -> None:
        """compare_traces returns results sorted descending by best_score."""
        pattern = CorpusPattern(
            name="security",
            tool_sequence=["corso/guard", "corso/code_review", "corso/speak"],
            frequency=100,
            domain="security",
        )
        analyzer = TraceAnalyzer(corpus=[pattern])

        # Perfect match
        trace_good = _make_trace(
            ["corso/guard", "corso/code_review", "corso/speak"],
            agent="good_agent",
            reward=3.0,
        )
        # Partial match
        trace_partial = _make_trace(
            ["corso/guard", "quantum/scan"],
            agent="partial_agent",
            reward=1.0,
        )
        # No match
        trace_bad = _make_trace(
            ["quantum/scan", "quantum/sweep"],
            agent="bad_agent",
            reward=0.5,
        )

        results = analyzer.compare_traces([trace_bad, trace_partial, trace_good])

        assert len(results) == 3
        assert results[0]["agent_name"] == "good_agent"
        assert results[0]["best_score"] == 1.0
        assert results[-1]["agent_name"] == "bad_agent"

        # Verify descending order
        scores = [r["best_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_compare_traces_includes_all_fields(self) -> None:
        """Each result dict contains the expected keys."""
        pattern = CorpusPattern(
            name="p", tool_sequence=["corso/guard"], frequency=1, domain="test"
        )
        analyzer = TraceAnalyzer(corpus=[pattern])
        trace = _make_trace(["corso/guard"], agent="a1", scenario="s1", reward=2.0)

        results = analyzer.compare_traces([trace])
        assert len(results) == 1
        result = results[0]

        assert result["agent_name"] == "a1"
        assert result["scenario_name"] == "s1"
        assert result["best_pattern"] == "p"
        assert isinstance(result["best_score"], float)
        assert result["total_reward"] == 2.0


# ---------------------------------------------------------------------------
# Default corpus
# ---------------------------------------------------------------------------


class TestDefaultCorpus:
    """Tests for the DEFAULT_CORPUS module constant."""

    def test_default_corpus_not_empty(self) -> None:
        """DEFAULT_CORPUS contains at least one pattern."""
        assert len(DEFAULT_CORPUS) > 0

    def test_default_corpus_has_expected_count(self) -> None:
        """DEFAULT_CORPUS has exactly 10 patterns."""
        assert len(DEFAULT_CORPUS) == 10

    def test_default_corpus_entries_are_valid(self) -> None:
        """Every corpus pattern has non-empty name, sequence, and domain."""
        for pattern in DEFAULT_CORPUS:
            assert pattern.name, f"Pattern has empty name: {pattern}"
            assert len(pattern.tool_sequence) > 0, f"Empty sequence: {pattern.name}"
            assert pattern.domain, f"Empty domain: {pattern.name}"
            assert pattern.frequency > 0, f"Zero frequency: {pattern.name}"

    def test_default_corpus_unique_names(self) -> None:
        """All pattern names in DEFAULT_CORPUS are unique."""
        names = [p.name for p in DEFAULT_CORPUS]
        assert len(names) == len(set(names))
