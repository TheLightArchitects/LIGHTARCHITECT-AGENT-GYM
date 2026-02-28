"""Tests for the multi-dimensional reward system.

Covers:
    - RewardConfig: weight validation (must sum to 1.0), custom weights, defaults
    - RewardBreakdown: field constraints (0.0-1.0), details dict
    - Judgment scoring: Jaccard similarity, Kendall-tau ordering, empty/null expected_sequence
    - Safety scoring: forbidden action detection, penalty calculation, no forbidden defined
    - Efficiency scoring: token budget ratio, duplicate detection, edge cases
    - Context maintenance scoring: prior result key threading, single-step episodes
    - Escalation scoring: escalation point detection, forbidden action penalties
    - RubricEvaluator: full 5-dimension evaluation with weighted total
    - MultiDimensionalReward: compute() with scenario_context, step_reward()
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mcp_gym.rewards import (
    MultiDimensionalReward,
    RewardBreakdown,
    RewardConfig,
    RubricEvaluator,
    _count_consecutive_duplicates,
    _extract_string_values,
    _jaccard_similarity,
    _kendall_tau_distance,
    _score_context_maintenance,
    _score_efficiency,
    _score_escalation,
    _score_judgment,
    _score_safety,
)

# ---------------------------------------------------------------------------
# RewardConfig
# ---------------------------------------------------------------------------


class TestRewardConfig:
    """Tests for RewardConfig weight validation."""

    def test_default_weights_sum_to_one(self) -> None:
        """Default weights (0.35+0.20+0.15+0.15+0.15) must equal 1.0."""
        config = RewardConfig()
        total = (
            config.judgment_weight
            + config.safety_weight
            + config.efficiency_weight
            + config.context_weight
            + config.escalation_weight
        )
        assert abs(total - 1.0) < 1e-6

    def test_default_weight_values(self) -> None:
        """Verify specific default values for each weight."""
        config = RewardConfig()
        assert config.judgment_weight == pytest.approx(0.35)
        assert config.safety_weight == pytest.approx(0.20)
        assert config.efficiency_weight == pytest.approx(0.15)
        assert config.context_weight == pytest.approx(0.15)
        assert config.escalation_weight == pytest.approx(0.15)

    def test_custom_weights_valid(self) -> None:
        """Custom weights that sum to 1.0 pass validation."""
        config = RewardConfig(
            judgment_weight=0.5,
            safety_weight=0.2,
            efficiency_weight=0.1,
            context_weight=0.1,
            escalation_weight=0.1,
        )
        assert config.judgment_weight == pytest.approx(0.5)

    def test_weights_not_summing_to_one_raises(self) -> None:
        """Weights that do not sum to 1.0 raise a validation error."""
        with pytest.raises(ValidationError, match=r"Weights must sum to 1\.0"):
            RewardConfig(
                judgment_weight=0.5,
                safety_weight=0.5,
                efficiency_weight=0.5,
                context_weight=0.0,
                escalation_weight=0.0,
            )

    def test_weights_below_one_raises(self) -> None:
        """Weights summing to less than 1.0 raise a validation error."""
        with pytest.raises(ValidationError, match=r"Weights must sum to 1\.0"):
            RewardConfig(
                judgment_weight=0.1,
                safety_weight=0.1,
                efficiency_weight=0.1,
                context_weight=0.1,
                escalation_weight=0.1,
            )

    def test_all_weight_on_safety(self) -> None:
        """Edge case: all weight on a single dimension."""
        config = RewardConfig(
            judgment_weight=0.0,
            safety_weight=1.0,
            efficiency_weight=0.0,
            context_weight=0.0,
            escalation_weight=0.0,
        )
        assert config.safety_weight == pytest.approx(1.0)

    def test_equal_weights(self) -> None:
        """Equal distribution across all five dimensions."""
        config = RewardConfig(
            judgment_weight=0.2,
            safety_weight=0.2,
            efficiency_weight=0.2,
            context_weight=0.2,
            escalation_weight=0.2,
        )
        assert (
            config.judgment_weight
            + config.safety_weight
            + config.efficiency_weight
            + config.context_weight
            + config.escalation_weight
        ) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# RewardBreakdown
# ---------------------------------------------------------------------------


class TestRewardBreakdown:
    """Tests for RewardBreakdown field constraints and structure."""

    def test_valid_breakdown(self) -> None:
        """A breakdown with all scores in [0.0, 1.0] is accepted."""
        breakdown = RewardBreakdown(
            judgment=0.8,
            safety=1.0,
            efficiency=0.5,
            context_maintenance=0.7,
            escalation=0.9,
            total=0.75,
        )
        assert breakdown.judgment == pytest.approx(0.8)
        assert breakdown.total == pytest.approx(0.75)

    def test_judgment_below_zero_raises(self) -> None:
        """Judgment score below 0.0 raises validation error."""
        with pytest.raises(ValidationError):
            RewardBreakdown(
                judgment=-0.1,
                safety=1.0,
                efficiency=0.5,
                context_maintenance=0.5,
                escalation=0.5,
                total=0.5,
            )

    def test_safety_above_one_raises(self) -> None:
        """Safety score above 1.0 raises validation error."""
        with pytest.raises(ValidationError):
            RewardBreakdown(
                judgment=0.5,
                safety=1.1,
                efficiency=0.5,
                context_maintenance=0.5,
                escalation=0.5,
                total=0.5,
            )

    def test_total_constrained(self) -> None:
        """Total score must be in [0.0, 1.0]."""
        with pytest.raises(ValidationError):
            RewardBreakdown(
                judgment=1.0,
                safety=1.0,
                efficiency=1.0,
                context_maintenance=1.0,
                escalation=1.0,
                total=1.5,
            )

    def test_details_default_empty(self) -> None:
        """Details dict defaults to empty when not provided."""
        breakdown = RewardBreakdown(
            judgment=0.5,
            safety=0.5,
            efficiency=0.5,
            context_maintenance=0.5,
            escalation=0.5,
            total=0.5,
        )
        assert breakdown.details == {}

    def test_details_populated(self) -> None:
        """Details dict can hold string key-value pairs."""
        details = {"judgment": "Jaccard=0.80", "safety": "No violations"}
        breakdown = RewardBreakdown(
            judgment=0.5,
            safety=0.5,
            efficiency=0.5,
            context_maintenance=0.5,
            escalation=0.5,
            total=0.5,
            details=details,
        )
        assert breakdown.details["judgment"] == "Jaccard=0.80"

    def test_boundary_zero_scores(self) -> None:
        """All-zero scores are valid."""
        breakdown = RewardBreakdown(
            judgment=0.0,
            safety=0.0,
            efficiency=0.0,
            context_maintenance=0.0,
            escalation=0.0,
            total=0.0,
        )
        assert breakdown.total == pytest.approx(0.0)

    def test_boundary_max_scores(self) -> None:
        """All 1.0 scores are valid."""
        breakdown = RewardBreakdown(
            judgment=1.0,
            safety=1.0,
            efficiency=1.0,
            context_maintenance=1.0,
            escalation=1.0,
            total=1.0,
        )
        assert breakdown.total == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Jaccard similarity
# ---------------------------------------------------------------------------


class TestJaccardSimilarity:
    """Tests for _jaccard_similarity helper."""

    def test_identical_sets(self) -> None:
        """Identical sets yield 1.0."""
        assert _jaccard_similarity({"a", "b"}, {"a", "b"}) == pytest.approx(1.0)

    def test_disjoint_sets(self) -> None:
        """Completely disjoint sets yield 0.0."""
        assert _jaccard_similarity({"a", "b"}, {"c", "d"}) == pytest.approx(0.0)

    def test_both_empty(self) -> None:
        """Both empty sets yield 0.0 (not division-by-zero)."""
        assert _jaccard_similarity(set(), set()) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        """Partial overlap: {a,b,c} vs {b,c,d} = 2/4 = 0.5."""
        assert _jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"}) == pytest.approx(0.5)

    def test_subset(self) -> None:
        """A is a subset of B: {a} vs {a,b} = 1/2 = 0.5."""
        assert _jaccard_similarity({"a"}, {"a", "b"}) == pytest.approx(0.5)

    def test_one_empty(self) -> None:
        """One empty, one non-empty yields 0.0."""
        assert _jaccard_similarity(set(), {"a"}) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Kendall-tau distance
# ---------------------------------------------------------------------------


class TestKendallTauDistance:
    """Tests for _kendall_tau_distance helper."""

    def test_identical_order(self) -> None:
        """Same order yields 0.0 distance."""
        assert _kendall_tau_distance(["a", "b", "c"], ["a", "b", "c"]) == pytest.approx(0.0)

    def test_reversed_order(self) -> None:
        """Fully reversed order yields 1.0 distance."""
        assert _kendall_tau_distance(["c", "b", "a"], ["a", "b", "c"]) == pytest.approx(1.0)

    def test_fewer_than_two_common(self) -> None:
        """Fewer than 2 common elements yields 0.0."""
        assert _kendall_tau_distance(["a"], ["b"]) == pytest.approx(0.0)

    def test_one_common_element(self) -> None:
        """Single common element yields 0.0."""
        assert _kendall_tau_distance(["a", "x"], ["a", "y"]) == pytest.approx(0.0)

    def test_partial_swap(self) -> None:
        """One pair swapped in 3 elements: 1/3 discordant pairs."""
        # expected: a,b,c. actual: a,c,b => pair (b,c) is swapped = 1 discordant out of 3
        assert _kendall_tau_distance(["a", "c", "b"], ["a", "b", "c"]) == pytest.approx(1 / 3)

    def test_actual_has_extra_elements(self) -> None:
        """Extra elements in actual that are not in expected are ignored."""
        assert _kendall_tau_distance(["x", "a", "b", "c", "y"], ["a", "b", "c"]) == pytest.approx(0.0)

    def test_empty_actual(self) -> None:
        """Empty actual list yields 0.0."""
        assert _kendall_tau_distance([], ["a", "b"]) == pytest.approx(0.0)

    def test_empty_expected(self) -> None:
        """Empty expected list yields 0.0."""
        assert _kendall_tau_distance(["a", "b"], []) == pytest.approx(0.0)

    def test_duplicates_in_actual(self) -> None:
        """Duplicates in actual are deduplicated, keeping first occurrence."""
        # After dedup: a, b, c (same as expected)
        assert _kendall_tau_distance(["a", "a", "b", "c"], ["a", "b", "c"]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Judgment scoring
# ---------------------------------------------------------------------------


class TestScoreJudgment:
    """Tests for _score_judgment."""

    def test_none_expected_returns_mid_score(self) -> None:
        """When no expected_sequence is provided, return 0.5."""
        score, detail = _score_judgment(
            [{"server": "corso", "action": "guard"}], None
        )
        assert score == pytest.approx(0.5)
        assert "No expected sequence" in detail

    def test_empty_call_history(self) -> None:
        """Empty call history with valid expected yields 0.0."""
        score, detail = _score_judgment(
            [], [{"server": "corso", "action": "guard"}]
        )
        assert score == pytest.approx(0.0)
        assert "Empty call history" in detail

    def test_perfect_match(self) -> None:
        """Exact match on tools and order yields a high score."""
        calls = [
            {"server": "corso", "action": "guard"},
            {"server": "eva", "action": "speak"},
        ]
        expected = [
            {"server": "corso", "action": "guard"},
            {"server": "eva", "action": "speak"},
        ]
        score, _detail = _score_judgment(calls, expected)
        # Jaccard=1.0 * 0.7 + (1-0)*0.3 = 0.7 + 0.3 = 1.0
        assert score == pytest.approx(1.0)

    def test_partial_match(self) -> None:
        """Partial tool overlap yields intermediate score."""
        calls = [
            {"server": "corso", "action": "guard"},
            {"server": "quantum", "action": "scan"},
        ]
        expected = [
            {"server": "corso", "action": "guard"},
            {"server": "eva", "action": "speak"},
        ]
        score, _ = _score_judgment(calls, expected)
        # Jaccard = 1/3 (intersection=1, union=3)
        assert 0.0 < score < 1.0

    def test_completely_wrong_tools(self) -> None:
        """No overlap yields a low score (order bonus only from no common pairs)."""
        calls = [{"server": "quantum", "action": "scan"}]
        expected = [{"server": "corso", "action": "guard"}]
        score, _ = _score_judgment(calls, expected)
        # Jaccard=0, but order_bonus=(1-0)*0.3=0.3 (< 2 common, tau=0)
        assert score == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Safety scoring
# ---------------------------------------------------------------------------


class TestScoreSafety:
    """Tests for _score_safety."""

    def test_no_forbidden_defined(self) -> None:
        """When no forbidden actions defined, score is 1.0."""
        score, detail = _score_safety(
            [{"server": "corso", "action": "guard"}], None
        )
        assert score == pytest.approx(1.0)
        assert "No forbidden actions defined" in detail

    def test_empty_forbidden_list(self) -> None:
        """Empty list of forbidden actions yields 1.0."""
        score, _detail = _score_safety(
            [{"server": "corso", "action": "guard"}], []
        )
        assert score == pytest.approx(1.0)

    def test_no_violations(self) -> None:
        """No forbidden actions were called -- full 1.0."""
        score, detail = _score_safety(
            [{"server": "corso", "action": "guard"}],
            ["quantum/destroy"],
        )
        assert score == pytest.approx(1.0)
        assert "No forbidden actions called" in detail

    def test_one_violation(self) -> None:
        """One forbidden action called deducts 0.3."""
        score, _ = _score_safety(
            [{"server": "quantum", "action": "destroy"}],
            ["quantum/destroy"],
        )
        assert score == pytest.approx(0.7)

    def test_multiple_violations(self) -> None:
        """Multiple forbidden calls each deduct 0.3."""
        calls = [
            {"server": "quantum", "action": "destroy"},
            {"server": "quantum", "action": "destroy"},
        ]
        score, _ = _score_safety(calls, ["quantum/destroy"])
        assert score == pytest.approx(0.4)

    def test_four_violations_floors_at_zero(self) -> None:
        """4 violations x 0.3 = 1.2 penalty, floored at 0.0."""
        calls = [{"server": "q", "action": "x"}] * 4
        score, _ = _score_safety(calls, ["q/x"])
        assert score == pytest.approx(0.0)

    def test_violation_detail_message(self) -> None:
        """Detail message shows violation count."""
        _, detail = _score_safety(
            [{"server": "q", "action": "x"}],
            ["q/x"],
        )
        assert "1 violation(s)" in detail


# ---------------------------------------------------------------------------
# Efficiency scoring
# ---------------------------------------------------------------------------


class TestScoreEfficiency:
    """Tests for _score_efficiency."""

    def test_zero_tokens_used(self) -> None:
        """Using zero tokens yields max base score (1.0 + 0.1 bonus = 1.0 clamped)."""
        score, _ = _score_efficiency([], token_budget=4096, tokens_used=0)
        assert score == pytest.approx(1.0)

    def test_full_budget_used(self) -> None:
        """Using 100% of budget: base = 0.0, no bonus."""
        score, _ = _score_efficiency([], token_budget=1000, tokens_used=1000)
        assert score == pytest.approx(0.0)

    def test_over_budget(self) -> None:
        """Using more than budget: negative base, floored at 0.0."""
        score, _ = _score_efficiency([], token_budget=1000, tokens_used=2000)
        assert score == pytest.approx(0.0)

    def test_under_half_budget_bonus(self) -> None:
        """Using less than 50% budget gets a +0.1 bonus."""
        # ratio = 100/1000 = 0.1, base = 1.0 - 0.1 = 0.9, bonus => 1.0
        score, _ = _score_efficiency([], token_budget=1000, tokens_used=100)
        assert score == pytest.approx(1.0)

    def test_exactly_half_budget_no_bonus(self) -> None:
        """Using exactly 50% budget: no bonus (ratio is not < 0.5)."""
        score, _ = _score_efficiency([], token_budget=1000, tokens_used=500)
        # base = 1.0 - 0.5 = 0.5, no bonus
        assert score == pytest.approx(0.5)

    def test_invalid_budget_zero(self) -> None:
        """Zero token budget returns 0.0."""
        score, detail = _score_efficiency([], token_budget=0, tokens_used=0)
        assert score == pytest.approx(0.0)
        assert "Invalid token budget" in detail

    def test_invalid_budget_negative(self) -> None:
        """Negative token budget returns 0.0."""
        score, _detail = _score_efficiency([], token_budget=-1, tokens_used=0)
        assert score == pytest.approx(0.0)

    def test_consecutive_duplicates_penalty(self) -> None:
        """Consecutive duplicate calls incur a 0.1 penalty per dupe."""
        calls = [
            {"server": "corso", "action": "guard", "params": {"path": "/src"}},
            {"server": "corso", "action": "guard", "params": {"path": "/src"}},
        ]
        score, detail = _score_efficiency(calls, token_budget=10000, tokens_used=100)
        # ratio=0.01, base=0.99+0.1=1.0, dupe_penalty=1*0.1=0.1, final=0.9
        assert score == pytest.approx(0.9)
        assert "consecutive_dupes=1" in detail

    def test_non_consecutive_duplicates_no_penalty(self) -> None:
        """Non-consecutive identical calls do not count as duplicates."""
        calls = [
            {"server": "corso", "action": "guard", "params": {"path": "/src"}},
            {"server": "eva", "action": "speak", "params": {"msg": "hi"}},
            {"server": "corso", "action": "guard", "params": {"path": "/src"}},
        ]
        _score, detail = _score_efficiency(calls, token_budget=10000, tokens_used=100)
        assert "consecutive_dupes=0" in detail


# ---------------------------------------------------------------------------
# Consecutive duplicate counter
# ---------------------------------------------------------------------------


class TestCountConsecutiveDuplicates:
    """Tests for _count_consecutive_duplicates helper."""

    def test_empty_history(self) -> None:
        """Empty history has zero duplicates."""
        assert _count_consecutive_duplicates([]) == 0

    def test_single_entry(self) -> None:
        """Single entry has zero duplicates."""
        assert _count_consecutive_duplicates([{"server": "a", "action": "b", "params": {}}]) == 0

    def test_two_identical(self) -> None:
        """Two identical consecutive entries = 1 duplicate."""
        entries = [
            {"server": "a", "action": "b", "params": {"x": 1}},
            {"server": "a", "action": "b", "params": {"x": 1}},
        ]
        assert _count_consecutive_duplicates(entries) == 1

    def test_three_identical(self) -> None:
        """Three identical consecutive entries = 2 duplicates."""
        entry = {"server": "a", "action": "b", "params": {}}
        assert _count_consecutive_duplicates([entry, entry, entry]) == 2

    def test_different_params_not_duplicate(self) -> None:
        """Same server/action but different params is not a duplicate."""
        entries = [
            {"server": "a", "action": "b", "params": {"x": 1}},
            {"server": "a", "action": "b", "params": {"x": 2}},
        ]
        assert _count_consecutive_duplicates(entries) == 0


# ---------------------------------------------------------------------------
# Extract string values
# ---------------------------------------------------------------------------


class TestExtractStringValues:
    """Tests for _extract_string_values helper."""

    def test_extracts_strings_of_length_3_or_more(self) -> None:
        """Only string values with length >= 3 are extracted."""
        result = _extract_string_values({"a": "abc", "b": "xy", "c": 123})
        assert result == {"abc"}

    def test_empty_dict(self) -> None:
        """Empty dict yields empty set."""
        assert _extract_string_values({}) == set()

    def test_all_short_strings(self) -> None:
        """All strings shorter than 3 characters are excluded."""
        result = _extract_string_values({"x": "ab", "y": "cd"})
        assert result == set()


# ---------------------------------------------------------------------------
# Context maintenance scoring
# ---------------------------------------------------------------------------


class TestScoreContextMaintenance:
    """Tests for _score_context_maintenance."""

    def test_empty_history(self) -> None:
        """Empty history yields 0.0."""
        score, _ = _score_context_maintenance([])
        assert score == pytest.approx(0.0)

    def test_single_step(self) -> None:
        """Single step yields 1.0 (nothing to thread)."""
        score, detail = _score_context_maintenance(
            [{"server": "corso", "action": "guard", "params": {}, "result": {"status": "clean"}}]
        )
        assert score == pytest.approx(1.0)
        assert "1 step(s)" in detail

    def test_two_steps_with_context_threading(self) -> None:
        """Second step references a key from the first step's result."""
        calls = [
            {
                "server": "corso",
                "action": "guard",
                "params": {},
                "result": {"scan_id": "abc123", "status": "clean"},
            },
            {
                "server": "eva",
                "action": "speak",
                "params": {"reference": "abc123"},
                "result": {"response": "ok"},
            },
        ]
        score, detail = _score_context_maintenance(calls)
        # "abc123" from step 0 result is referenced in step 1 params
        assert score == pytest.approx(1.0)
        assert "1/1" in detail

    def test_two_steps_without_context_threading(self) -> None:
        """Second step does NOT reference any key from the first step's result."""
        calls = [
            {
                "server": "corso",
                "action": "guard",
                "params": {},
                "result": {"scan_id": "abc123"},
            },
            {
                "server": "eva",
                "action": "speak",
                "params": {"message": "unrelated"},
                "result": {},
            },
        ]
        score, detail = _score_context_maintenance(calls)
        assert score == pytest.approx(0.0)
        assert "0/1" in detail

    def test_no_result_dict_in_prior_step(self) -> None:
        """If prior step result is not a dict, no keys are accumulated."""
        calls = [
            {
                "server": "corso",
                "action": "guard",
                "params": {},
                "result": "string_result",
            },
            {
                "server": "eva",
                "action": "speak",
                "params": {"message": "test"},
                "result": {},
            },
        ]
        _score, detail = _score_context_maintenance(calls)
        # No prior_result_keys accumulated (result was a string, not dict)
        # total_follow_steps = 0 (because prior_result_keys was empty at i=1)
        assert "No follow-up steps" in detail

    def test_multiple_steps_partial_context(self) -> None:
        """Mix of context-aware and non-context-aware follow-up steps."""
        calls = [
            {
                "server": "corso",
                "action": "guard",
                "params": {},
                "result": {"file_path": "/src/main.rs"},
            },
            {
                "server": "eva",
                "action": "speak",
                "params": {"path": "/src/main.rs"},
                "result": {"summary": "looks good"},
            },
            {
                "server": "quantum",
                "action": "scan",
                "params": {"unrelated": "value"},
                "result": {},
            },
        ]
        score, _detail = _score_context_maintenance(calls)
        # Step 1 references "/src/main.rs" from step 0 => context-aware
        # Step 2 does not reference anything from prior results (params value
        # "value" is not in prior result keys/values) => not context-aware
        # But actually "looks good" was added to prior_result_keys from step 1 result,
        # and "summary" key too. "value" (5 chars) won't match any prior key.
        # Score: 1/2 = 0.5
        assert score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Escalation scoring
# ---------------------------------------------------------------------------


class TestScoreEscalation:
    """Tests for _score_escalation."""

    def test_no_escalation_points(self) -> None:
        """No escalation points defined yields 1.0."""
        score, detail = _score_escalation([], None, None)
        assert score == pytest.approx(1.0)
        assert "No escalation points defined" in detail

    def test_empty_escalation_points(self) -> None:
        """Empty list of escalation points yields 1.0."""
        score, _ = _score_escalation([], [], None)
        assert score == pytest.approx(1.0)

    def test_escalation_made_correctly(self) -> None:
        """Agent called the expected escalation action."""
        calls = [{"server": "corso", "action": "speak"}]
        score, detail = _score_escalation(calls, ["corso/speak"], None)
        assert score == pytest.approx(1.0)
        assert "1/1" in detail

    def test_escalation_missed(self) -> None:
        """Agent did not call the expected escalation action."""
        calls = [{"server": "quantum", "action": "scan"}]
        score, detail = _score_escalation(calls, ["corso/speak"], None)
        assert score == pytest.approx(0.0)
        assert "0/1" in detail

    def test_partial_escalation(self) -> None:
        """Agent called one of two expected escalation actions."""
        calls = [{"server": "corso", "action": "speak"}]
        score, _ = _score_escalation(
            calls, ["corso/speak", "eva/escalate"], None
        )
        assert score == pytest.approx(0.5)

    def test_forbidden_penalty_on_escalation(self) -> None:
        """Agent called a forbidden action instead of escalating."""
        calls = [
            {"server": "quantum", "action": "destroy"},
            {"server": "corso", "action": "speak"},
        ]
        score, detail = _score_escalation(
            calls,
            ["corso/speak"],
            ["quantum/destroy"],
        )
        # base = 1/1 = 1.0, penalty = 1*0.2 = 0.2, score = 0.8
        assert score == pytest.approx(0.8)
        assert "forbidden_called=1" in detail

    def test_multiple_forbidden_penalty(self) -> None:
        """Multiple different forbidden actions each deduct 0.2."""
        calls = [
            {"server": "q", "action": "destroy"},
            {"server": "q", "action": "delete"},
        ]
        score, _ = _score_escalation(
            calls,
            ["corso/speak"],
            ["q/destroy", "q/delete"],
        )
        # base = 0/1 = 0.0, penalty = 2*0.2 = 0.4, floored at 0.0
        assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# RubricEvaluator
# ---------------------------------------------------------------------------


class TestRubricEvaluator:
    """Tests for the RubricEvaluator class."""

    def test_default_config(self) -> None:
        """RubricEvaluator uses default config when None is passed."""
        evaluator = RubricEvaluator()
        assert evaluator.config.judgment_weight == pytest.approx(0.35)

    def test_custom_config(self) -> None:
        """RubricEvaluator accepts a custom RewardConfig."""
        config = RewardConfig(
            judgment_weight=0.5,
            safety_weight=0.2,
            efficiency_weight=0.1,
            context_weight=0.1,
            escalation_weight=0.1,
        )
        evaluator = RubricEvaluator(config=config)
        assert evaluator.config.judgment_weight == pytest.approx(0.5)

    def test_full_evaluation(self) -> None:
        """Evaluate returns a RewardBreakdown with all dimensions and weighted total."""
        evaluator = RubricEvaluator()
        calls = [
            {"server": "corso", "action": "guard", "params": {}, "result": {"status": "clean"}},
            {"server": "eva", "action": "speak", "params": {"ref": "clean"}, "result": {"response": "ok"}},
        ]
        expected_seq = [
            {"server": "corso", "action": "guard"},
            {"server": "eva", "action": "speak"},
        ]
        breakdown = evaluator.evaluate(
            call_history=calls,
            expected_sequence=expected_seq,
            forbidden_actions=["quantum/destroy"],
            token_budget=4096,
            tokens_used=100,
            escalation_points=None,
        )
        assert isinstance(breakdown, RewardBreakdown)
        assert 0.0 <= breakdown.judgment <= 1.0
        assert 0.0 <= breakdown.safety <= 1.0
        assert 0.0 <= breakdown.efficiency <= 1.0
        assert 0.0 <= breakdown.context_maintenance <= 1.0
        assert 0.0 <= breakdown.escalation <= 1.0
        assert 0.0 <= breakdown.total <= 1.0
        assert len(breakdown.details) == 5

    def test_weighted_total_with_safety_only_config(self) -> None:
        """When all weight is on safety and safety=1.0, total=1.0."""
        config = RewardConfig(
            judgment_weight=0.0,
            safety_weight=1.0,
            efficiency_weight=0.0,
            context_weight=0.0,
            escalation_weight=0.0,
        )
        evaluator = RubricEvaluator(config=config)
        breakdown = evaluator.evaluate(
            call_history=[{"server": "corso", "action": "guard", "params": {}, "result": {}}],
            forbidden_actions=None,
            token_budget=4096,
            tokens_used=0,
        )
        # Safety score is 1.0 (no forbidden defined), total = 1.0 * 1.0
        assert breakdown.safety == pytest.approx(1.0)
        assert breakdown.total == pytest.approx(1.0)

    def test_total_is_clamped_to_one(self) -> None:
        """Total is clamped to [0.0, 1.0] even if per-dimension scores are high."""
        evaluator = RubricEvaluator()
        breakdown = evaluator.evaluate(
            call_history=[],
            token_budget=4096,
            tokens_used=0,
        )
        assert breakdown.total <= 1.0
        assert breakdown.total >= 0.0

    def test_evaluate_details_contains_all_dimensions(self) -> None:
        """Evaluate result details dict has all 5 dimension keys."""
        evaluator = RubricEvaluator()
        breakdown = evaluator.evaluate(call_history=[], token_budget=4096, tokens_used=0)
        assert "judgment" in breakdown.details
        assert "safety" in breakdown.details
        assert "efficiency" in breakdown.details
        assert "context_maintenance" in breakdown.details
        assert "escalation" in breakdown.details


# ---------------------------------------------------------------------------
# MultiDimensionalReward
# ---------------------------------------------------------------------------


class TestMultiDimensionalReward:
    """Tests for the MultiDimensionalReward high-level interface."""

    def test_default_evaluator(self) -> None:
        """MultiDimensionalReward creates a default evaluator when None is passed."""
        mdr = MultiDimensionalReward()
        assert mdr.evaluator is not None

    def test_custom_evaluator(self) -> None:
        """MultiDimensionalReward accepts a custom evaluator."""
        evaluator = RubricEvaluator()
        mdr = MultiDimensionalReward(evaluator=evaluator)
        assert mdr.evaluator is evaluator

    def test_compute_with_scenario_context(self) -> None:
        """compute() extracts values from scenario_context dict."""
        mdr = MultiDimensionalReward()
        calls = [
            {"server": "corso", "action": "guard", "params": {}, "result": {"ok": True}},
        ]
        context = {
            "expected_sequence": [{"server": "corso", "action": "guard"}],
            "forbidden_actions": ["quantum/destroy"],
            "token_budget": 2048,
            "tokens_used": 50,
            "escalation_points": None,
        }
        breakdown = mdr.compute(calls, context)
        assert isinstance(breakdown, RewardBreakdown)
        assert 0.0 <= breakdown.total <= 1.0

    def test_compute_with_empty_context(self) -> None:
        """compute() handles an empty scenario_context with safe defaults."""
        mdr = MultiDimensionalReward()
        breakdown = mdr.compute([], {})
        assert isinstance(breakdown, RewardBreakdown)
        # With empty history and default budget: judgment=0.5, safety=1.0,
        # efficiency=1.0+bonus, context=0.0, escalation=1.0
        assert 0.0 <= breakdown.total <= 1.0

    def test_compute_default_token_budget(self) -> None:
        """compute() uses 4096 token budget when not provided in context."""
        mdr = MultiDimensionalReward()
        breakdown = mdr.compute([], {"tokens_used": 100})
        # Should not raise; uses default 4096 budget
        assert isinstance(breakdown, RewardBreakdown)

    def test_step_reward_success(self) -> None:
        """Successful step returns +0.1."""
        mdr = MultiDimensionalReward()
        reward = mdr.step_reward({
            "success": True,
            "server": "corso",
            "action": "guard",
        })
        assert reward == pytest.approx(0.1)

    def test_step_reward_failure(self) -> None:
        """Failed step returns -0.1."""
        mdr = MultiDimensionalReward()
        reward = mdr.step_reward({
            "success": False,
            "server": "corso",
            "action": "guard",
        })
        assert reward == pytest.approx(-0.1)

    def test_step_reward_forbidden_action(self) -> None:
        """Forbidden action returns -0.5 regardless of success."""
        mdr = MultiDimensionalReward()
        reward = mdr.step_reward({
            "success": True,
            "server": "quantum",
            "action": "destroy",
            "forbidden_actions": ["quantum/destroy"],
        })
        assert reward == pytest.approx(-0.5)

    def test_step_reward_forbidden_but_not_called(self) -> None:
        """Action is not in forbidden list, so normal +0.1."""
        mdr = MultiDimensionalReward()
        reward = mdr.step_reward({
            "success": True,
            "server": "corso",
            "action": "guard",
            "forbidden_actions": ["quantum/destroy"],
        })
        assert reward == pytest.approx(0.1)

    def test_step_reward_no_forbidden_list(self) -> None:
        """Missing forbidden_actions key defaults to empty list."""
        mdr = MultiDimensionalReward()
        reward = mdr.step_reward({"success": True, "server": "a", "action": "b"})
        assert reward == pytest.approx(0.1)

    def test_step_reward_default_success_false(self) -> None:
        """Missing success key defaults to False."""
        mdr = MultiDimensionalReward()
        reward = mdr.step_reward({"server": "a", "action": "b"})
        assert reward == pytest.approx(-0.1)
