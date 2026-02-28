"""Tests for the hill-climbing training module."""

from __future__ import annotations

import pytest

from mcp_gym.training import (
    TrainingConfig,
    TrainingResult,
    compute_baseline,
    train_hill_climber,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

SCENARIO_DIR = "scenarios"


# ---------------------------------------------------------------------------
# TrainingConfig defaults
# ---------------------------------------------------------------------------


class TestTrainingConfigDefaults:
    """Verify TrainingConfig dataclass has correct defaults."""

    def test_training_config_defaults(self) -> None:
        config = TrainingConfig()
        assert config.num_episodes == 100
        assert config.perturb_magnitude == 0.3
        assert config.seed == 42
        assert config.scenario_dir == "scenarios"
        assert config.scenario_filter is None

    def test_training_config_custom(self) -> None:
        config = TrainingConfig(
            num_episodes=50,
            perturb_magnitude=0.5,
            seed=99,
            scenario_dir="custom/path",
            scenario_filter=["security"],
        )
        assert config.num_episodes == 50
        assert config.perturb_magnitude == 0.5
        assert config.seed == 99
        assert config.scenario_dir == "custom/path"
        assert config.scenario_filter == ["security"]


# ---------------------------------------------------------------------------
# TrainingResult fields
# ---------------------------------------------------------------------------


class TestTrainingResultFields:
    """Verify TrainingResult dataclass defaults and field types."""

    def test_training_result_fields(self) -> None:
        result = TrainingResult()
        assert result.episode_rewards == []
        assert result.best_reward == 0.0
        assert result.best_episode == 0
        assert result.initial_weights == {}
        assert result.final_weights == {}
        assert result.improvement == 0.0

    def test_training_result_populated(self) -> None:
        result = TrainingResult(
            episode_rewards=[0.1, 0.2, 0.3],
            best_reward=0.3,
            best_episode=3,
            initial_weights={"security": 1.0},
            final_weights={"security": 1.2},
            improvement=0.1,
        )
        assert len(result.episode_rewards) == 3
        assert result.best_reward == 0.3
        assert result.best_episode == 3
        assert result.improvement == 0.1


# ---------------------------------------------------------------------------
# train_hill_climber runs
# ---------------------------------------------------------------------------


class TestTrainHillClimberRuns:
    """Verify train_hill_climber executes and returns valid structure."""

    def test_train_hill_climber_runs(self) -> None:
        """Run 10 episodes and verify result structure."""
        config = TrainingConfig(
            num_episodes=10,
            perturb_magnitude=0.3,
            seed=42,
            scenario_dir=SCENARIO_DIR,
        )
        result = train_hill_climber(config)

        assert isinstance(result, TrainingResult)
        assert len(result.episode_rewards) == 10
        assert result.best_episode >= 1
        assert result.best_episode <= 10
        assert result.best_reward >= min(result.episode_rewards)
        assert isinstance(result.initial_weights, dict)
        assert isinstance(result.final_weights, dict)
        assert len(result.initial_weights) > 0
        assert len(result.final_weights) > 0

    def test_train_returns_float_rewards(self) -> None:
        """Every episode reward should be a float."""
        config = TrainingConfig(
            num_episodes=5,
            seed=123,
            scenario_dir=SCENARIO_DIR,
        )
        result = train_hill_climber(config)
        for reward in result.episode_rewards:
            assert isinstance(reward, float)


# ---------------------------------------------------------------------------
# Training improves over baseline
# ---------------------------------------------------------------------------


class TestTrainImprovesOverBaseline:
    """Verify training produces non-negative improvement over baseline."""

    def test_train_improves_over_baseline(self) -> None:
        """Run 50 episodes; improvement should be >= 0.

        Hill climbing is monotonic: the agent only keeps perturbations
        that improve reward, so the best reward should be at least
        as good as the baseline.
        """
        config = TrainingConfig(
            num_episodes=50,
            perturb_magnitude=0.3,
            seed=42,
            scenario_dir=SCENARIO_DIR,
        )
        result = train_hill_climber(config)

        # improvement = best_reward - baseline
        # Since hill climbing keeps only improvements, improvement >= 0
        assert result.improvement >= 0.0, (
            f"Expected non-negative improvement, got {result.improvement}"
        )


# ---------------------------------------------------------------------------
# compute_baseline deterministic
# ---------------------------------------------------------------------------


class TestComputeBaselineDeterministic:
    """Verify that compute_baseline is deterministic with the same seed."""

    def test_compute_baseline_deterministic(self) -> None:
        """Same seed should produce identical baseline across two runs."""
        from pathlib import Path

        from mcp_gym.agents import HillClimberAgent
        from mcp_gym.env import make_env
        from mcp_gym.scenarios.loader import load_all_scenarios

        scenarios = load_all_scenarios(Path(SCENARIO_DIR))
        env = make_env()
        tools = env.surface.registered_actions()

        agent1 = HillClimberAgent(available_tools=tools, seed=42)
        baseline1 = compute_baseline(agent1, scenarios, num_runs=5, seed=42)

        agent2 = HillClimberAgent(available_tools=tools, seed=42)
        baseline2 = compute_baseline(agent2, scenarios, num_runs=5, seed=42)

        assert baseline1 == baseline2, (
            f"Baselines differ: {baseline1} vs {baseline2}"
        )

    def test_compute_baseline_different_seeds(self) -> None:
        """Different seeds may produce different baselines."""
        from pathlib import Path

        from mcp_gym.agents import HillClimberAgent
        from mcp_gym.env import make_env
        from mcp_gym.scenarios.loader import load_all_scenarios

        scenarios = load_all_scenarios(Path(SCENARIO_DIR))
        env = make_env()
        tools = env.surface.registered_actions()

        agent1 = HillClimberAgent(available_tools=tools, seed=42)
        baseline1 = compute_baseline(agent1, scenarios, num_runs=5, seed=42)

        agent2 = HillClimberAgent(available_tools=tools, seed=999)
        baseline2 = compute_baseline(agent2, scenarios, num_runs=5, seed=999)

        # They CAN be equal by chance but we just verify both are valid floats
        assert isinstance(baseline1, float)
        assert isinstance(baseline2, float)


# ---------------------------------------------------------------------------
# Scenario filter
# ---------------------------------------------------------------------------


class TestScenarioFilter:
    """Verify that scenario_filter restricts training to matching domains."""

    def test_scenario_filter_security(self) -> None:
        """Only security scenarios should be used when filter=['security']."""
        config = TrainingConfig(
            num_episodes=10,
            perturb_magnitude=0.3,
            seed=42,
            scenario_dir=SCENARIO_DIR,
            scenario_filter=["security"],
        )
        result = train_hill_climber(config)

        assert isinstance(result, TrainingResult)
        assert len(result.episode_rewards) == 10
        # The training should complete without error
        assert result.best_episode >= 1

    def test_scenario_filter_no_match_raises(self) -> None:
        """A filter that matches no scenarios should raise ValueError."""
        config = TrainingConfig(
            num_episodes=5,
            scenario_dir=SCENARIO_DIR,
            scenario_filter=["nonexistent-domain"],
        )
        with pytest.raises(ValueError, match="No scenarios found"):
            train_hill_climber(config)

    def test_scenario_filter_investigation(self) -> None:
        """Filter for investigation domain should also work."""
        config = TrainingConfig(
            num_episodes=5,
            perturb_magnitude=0.2,
            seed=77,
            scenario_dir=SCENARIO_DIR,
            scenario_filter=["investigation"],
        )
        result = train_hill_climber(config)
        assert len(result.episode_rewards) == 5
