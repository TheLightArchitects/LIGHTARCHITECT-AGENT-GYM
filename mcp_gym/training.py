"""Hill-climbing training module for LIGHTARCHITECT-AGENT-GYM.

Provides a training loop that optimizes HillClimberAgent domain weights
by running episodes across loaded scenarios and using reward feedback
to keep or revert weight perturbations.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from pathlib import Path

from mcp_gym.agents import HillClimberAgent
from mcp_gym.env import make_env
from mcp_gym.scenarios.loader import load_all_scenarios
from mcp_gym.scenarios.schema import ScenarioDefinition
from mcp_gym.types import EpisodeConfig


@dataclass
class TrainingConfig:
    """Configuration for a hill-climbing training run.

    Attributes:
        num_episodes: Number of training episodes to execute.
        perturb_magnitude: Maximum magnitude of weight perturbation per episode.
        seed: Random seed for reproducibility.
        scenario_dir: Path to the directory containing YAML scenario files.
        scenario_filter: Optional list of domain strings to filter scenarios.
    """

    num_episodes: int = 100
    perturb_magnitude: float = 0.3
    seed: int = 42
    scenario_dir: str = "scenarios"
    scenario_filter: list[str] | None = None


@dataclass
class TrainingResult:
    """Results from a completed training run.

    Attributes:
        episode_rewards: Per-episode total reward values.
        best_reward: Highest reward observed during training.
        best_episode: Episode number (1-indexed) that achieved best_reward.
        initial_weights: Domain weights before training started.
        final_weights: Domain weights after training completed.
        improvement: Difference between best_reward and initial baseline.
    """

    episode_rewards: list[float] = field(default_factory=list)
    best_reward: float = 0.0
    best_episode: int = 0
    initial_weights: dict[str, float] = field(default_factory=dict)
    final_weights: dict[str, float] = field(default_factory=dict)
    improvement: float = 0.0


def _load_scenarios(config: TrainingConfig) -> list[ScenarioDefinition]:
    """Load and optionally filter scenarios from the configured directory.

    Args:
        config: Training configuration with scenario_dir and scenario_filter.

    Returns:
        Non-empty list of scenario definitions.

    Raises:
        ValueError: If no scenarios match after loading and filtering.
    """
    directory = Path(config.scenario_dir)
    all_scenarios = load_all_scenarios(directory)

    if config.scenario_filter is not None:
        filter_set = set(config.scenario_filter)
        all_scenarios = [s for s in all_scenarios if s.domain in filter_set]

    if not all_scenarios:
        msg = (
            f"No scenarios found in '{config.scenario_dir}' "
            f"matching filter {config.scenario_filter}"
        )
        raise ValueError(msg)

    return all_scenarios


def _get_available_tools() -> list[str]:
    """Create a temporary env and extract the registered tool list.

    Returns:
        Sorted list of 'server/action' strings from the mock surface.
    """
    env = make_env()
    return env.surface.registered_actions()


def _run_episode(
    agent: HillClimberAgent,
    scenario: ScenarioDefinition,
    seed: int,
) -> float:
    """Run a single training episode and return the total reward.

    Args:
        agent: The hill-climbing agent to evaluate.
        scenario: The scenario to run.
        seed: Random seed for the environment.

    Returns:
        Total accumulated reward for the episode.
    """
    episode_config = EpisodeConfig(
        max_steps=_total_max_steps(scenario),
        token_budget=scenario.token_budget,
        seed=seed,
    )
    env = make_env(config=episode_config)
    obs, _info = env.reset(seed=seed)
    agent.reset()

    total_reward = 0.0
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = agent.act(obs)
        obs, reward, terminated, truncated, _info = env.step(action)
        total_reward += reward

    return total_reward


def _total_max_steps(scenario: ScenarioDefinition) -> int:
    """Sum max_steps across all phases for the episode limit.

    Args:
        scenario: Scenario whose phases to sum.

    Returns:
        Total step budget, minimum 1.
    """
    total = sum(phase.max_steps for phase in scenario.phases)
    return max(1, total)


def compute_baseline(
    agent: HillClimberAgent,
    scenarios: list[ScenarioDefinition],
    num_runs: int = 5,
    seed: int = 42,
) -> float:
    """Run the agent without perturbation to establish baseline reward.

    Cycles through scenarios for each run, averages all episode rewards.

    Args:
        agent: Agent to evaluate (weights are not modified).
        scenarios: List of scenarios to cycle through.
        num_runs: Number of baseline evaluation runs.
        seed: Base seed for reproducibility.

    Returns:
        Mean reward across all baseline runs.
    """
    if not scenarios:
        return 0.0

    total_reward = 0.0
    for i in range(num_runs):
        scenario = scenarios[i % len(scenarios)]
        episode_seed = seed + i
        reward = _run_episode(agent, scenario, seed=episode_seed)
        total_reward += reward

    return total_reward / num_runs


def train_hill_climber(config: TrainingConfig) -> TrainingResult:
    """Train a HillClimberAgent via hill climbing over loaded scenarios.

    For each episode:
    1. Pick a scenario (cycling through available scenarios).
    2. Perturb agent weights.
    3. Run the episode to completion.
    4. Update agent (keep or revert perturbation based on reward).

    Args:
        config: Training configuration.

    Returns:
        TrainingResult with full episode history and weight snapshots.
    """
    scenarios = _load_scenarios(config)
    available_tools = _get_available_tools()

    agent = HillClimberAgent(
        available_tools=available_tools,
        seed=config.seed,
    )
    initial_weights = copy.deepcopy(agent.weights)

    # Establish baseline before training
    baseline = compute_baseline(
        agent, scenarios, num_runs=min(5, config.num_episodes), seed=config.seed,
    )

    rng = random.Random(config.seed)
    result = _training_loop(agent, scenarios, config, rng)

    result.initial_weights = initial_weights
    result.final_weights = copy.deepcopy(agent.weights)
    result.improvement = result.best_reward - baseline

    return result


def _training_loop(
    agent: HillClimberAgent,
    scenarios: list[ScenarioDefinition],
    config: TrainingConfig,
    rng: random.Random,
) -> TrainingResult:
    """Execute the core training loop across all episodes.

    Args:
        agent: The agent to train.
        scenarios: Available scenarios.
        config: Training configuration.
        rng: Seeded random number generator for scenario shuffling.

    Returns:
        TrainingResult populated with episode_rewards, best_reward, best_episode.
    """
    result = TrainingResult()
    scenario_indices = list(range(len(scenarios)))

    for episode in range(config.num_episodes):
        scenario_idx = scenario_indices[episode % len(scenario_indices)]
        scenario = scenarios[scenario_idx]
        episode_seed = config.seed + episode

        agent.perturb(magnitude=config.perturb_magnitude)
        reward = _run_episode(agent, scenario, seed=episode_seed)
        agent.update(reward)

        result.episode_rewards.append(reward)
        if reward > result.best_reward or episode == 0:
            result.best_reward = reward
            result.best_episode = episode + 1  # 1-indexed

    return result
