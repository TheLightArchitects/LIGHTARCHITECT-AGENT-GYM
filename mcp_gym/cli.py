"""CLI for LIGHTARCHITECT-AGENT-GYM.

Thin Typer wrappers around existing modules for running scenarios,
evaluating agents, training hill-climbers, and comparing traces.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from mcp_gym.agents import BaseAgent, HillClimberAgent, RandomAgent, RuleBasedAgent
from mcp_gym.env import MCPAgentEnv, make_env
from mcp_gym.rewards import MultiDimensionalReward
from mcp_gym.scenarios import load_all_scenarios, load_scenario
from mcp_gym.trace import DEFAULT_CORPUS, TraceAnalyzer, TraceLogger
from mcp_gym.training import TrainingConfig, train_hill_climber
from mcp_gym.types import EpisodeConfig

app = typer.Typer(
    name="mcp-gym",
    help="LIGHTARCHITECT-AGENT-GYM -- evaluate AI agent tool-use behavior.",
    add_completion=False,
)

AGENT_CHOICES = ("random", "rule-based", "hill-climber")


def _get_tools() -> list[str]:
    """Extract registered tool list from a temporary env."""
    env = make_env()
    return env.surface.registered_actions()


def _make_agent(
    name: str,
    tools: list[str],
    seed: int,
) -> RandomAgent | RuleBasedAgent | HillClimberAgent:
    """Construct an agent by name string."""
    if name == "random":
        return RandomAgent(tools, seed=seed)
    if name == "rule-based":
        return RuleBasedAgent(tools, seed=seed)
    if name == "hill-climber":
        return HillClimberAgent(tools, seed=seed)
    msg = f"Unknown agent: {name}. Choose from: {AGENT_CHOICES}"
    raise typer.BadParameter(msg)


def _run_episode(
    agent: BaseAgent,
    env: MCPAgentEnv,
    seed: int,
    verbose: bool = False,
) -> tuple[float, list[dict[str, Any]]]:
    """Run one episode, optionally printing step output.

    Returns:
        Tuple of (total_reward, call_history).
    """
    obs, _info = env.reset(seed=seed)
    agent.reset()

    total_reward = 0.0
    terminated = False
    truncated = False
    step_num = 0

    while not terminated and not truncated:
        action_str = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action_str)
        total_reward += reward
        step_num += 1

        if verbose:
            parsed = json.loads(action_str)
            status = "OK" if info.get("success") else "FAIL"
            typer.echo(
                f"  Step {step_num}: {parsed['server']}/{parsed['action']} "
                f"-> {status} (reward: {reward:+.2f})"
            )

    return total_reward, env._call_history


@app.command()
def run(
    scenario: Annotated[
        Path, typer.Option(help="Path to a single scenario YAML file.")
    ] = Path("scenarios/security-vulnerability-scan.yaml"),
    agent: Annotated[
        str, typer.Option(help="Agent type: random, rule-based, or hill-climber.")
    ] = "rule-based",
    seed: Annotated[int, typer.Option(help="Random seed for reproducibility.")] = 42,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Print step-by-step results.")] = False,
) -> None:
    """Run a single scenario with a single agent."""
    scenario_def = load_scenario(scenario)
    tools = _get_tools()
    ag = _make_agent(agent, tools, seed)

    max_steps = sum(p.max_steps for p in scenario_def.phases)
    config = EpisodeConfig(max_steps=max(1, max_steps), token_budget=scenario_def.token_budget, seed=seed)
    env = make_env(config=config)

    typer.echo(f"Scenario: {scenario_def.name} ({scenario_def.domain})")
    typer.echo(f"Agent:    {agent}")
    typer.echo(f"Seed:     {seed}")
    typer.echo("---")

    total_reward, call_history = _run_episode(ag, env, seed, verbose)

    # Rubric evaluation
    reward_sys = MultiDimensionalReward()
    breakdown = reward_sys.compute(call_history, {
        "expected_sequence": scenario_def.expected_sequence,
        "forbidden_actions": scenario_def.forbidden_actions,
        "token_budget": scenario_def.token_budget,
        "tokens_used": scenario_def.token_budget - env._token_budget,
    })

    typer.echo(f"\nTotal step reward:  {total_reward:+.3f}")
    typer.echo(f"Rubric total:       {breakdown.total:.4f}")
    typer.echo(f"  Judgment:         {breakdown.judgment:.4f}")
    typer.echo(f"  Safety:           {breakdown.safety:.4f}")
    typer.echo(f"  Efficiency:       {breakdown.efficiency:.4f}")
    typer.echo(f"  Context:          {breakdown.context_maintenance:.4f}")
    typer.echo(f"  Escalation:       {breakdown.escalation:.4f}")


@app.command()
def evaluate(
    scenario_dir: Annotated[
        Path, typer.Option(help="Directory containing scenario YAML files.")
    ] = Path("scenarios"),
    agents: Annotated[
        str, typer.Option(help="Comma-separated agent names.")
    ] = "random,rule-based,hill-climber",
    seed: Annotated[int, typer.Option(help="Random seed.")] = 42,
) -> None:
    """Run all scenarios with specified agents. Print comparison matrix."""
    scenarios = load_all_scenarios(scenario_dir)
    tools = _get_tools()
    agent_names = [a.strip() for a in agents.split(",")]

    # Header
    name_width = max(len(s.name) for s in scenarios)
    header = f"{'Scenario':<{name_width}}"
    for ag_name in agent_names:
        header += f"  {ag_name:>12}"
    typer.echo(header)
    typer.echo("-" * len(header))

    # Run each scenario x agent combination
    for scenario_def in scenarios:
        row = f"{scenario_def.name:<{name_width}}"
        max_steps = max(1, sum(p.max_steps for p in scenario_def.phases))

        for ag_name in agent_names:
            ag = _make_agent(ag_name, tools, seed)
            config = EpisodeConfig(max_steps=max_steps, token_budget=scenario_def.token_budget, seed=seed)
            env = make_env(config=config)
            _total_reward, call_history = _run_episode(ag, env, seed)

            reward_sys = MultiDimensionalReward()
            breakdown = reward_sys.compute(call_history, {
                "expected_sequence": scenario_def.expected_sequence,
                "forbidden_actions": scenario_def.forbidden_actions,
                "token_budget": scenario_def.token_budget,
                "tokens_used": scenario_def.token_budget - env._token_budget,
            })
            row += f"  {breakdown.total:>12.4f}"

        typer.echo(row)


@app.command()
def train(
    episodes: Annotated[int, typer.Option(help="Number of training episodes.")] = 100,
    magnitude: Annotated[float, typer.Option(help="Perturbation magnitude.")] = 0.3,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 42,
    scenario_dir: Annotated[
        Path, typer.Option(help="Directory containing scenario YAML files.")
    ] = Path("scenarios"),
    domain: Annotated[
        str | None, typer.Option(help="Filter scenarios by domain.")
    ] = None,
) -> None:
    """Train a hill-climber agent. Print reward improvement."""
    scenario_filter = [domain] if domain else None
    config = TrainingConfig(
        num_episodes=episodes,
        perturb_magnitude=magnitude,
        seed=seed,
        scenario_dir=str(scenario_dir),
        scenario_filter=scenario_filter,
    )

    typer.echo(f"Training hill-climber: {episodes} episodes, magnitude={magnitude}")
    if domain:
        typer.echo(f"Domain filter: {domain}")
    typer.echo("---")

    result = train_hill_climber(config)

    typer.echo(f"\nBest reward:    {result.best_reward:+.4f} (episode {result.best_episode})")
    typer.echo(f"Improvement:    {result.improvement:+.4f}")
    typer.echo(f"\nInitial weights: {_fmt_weights(result.initial_weights)}")
    typer.echo(f"Final weights:   {_fmt_weights(result.final_weights)}")

    # Show last 5 episode rewards
    tail = result.episode_rewards[-5:]
    tail_str = ", ".join(f"{r:+.3f}" for r in tail)
    typer.echo(f"\nLast 5 rewards: [{tail_str}]")


def _fmt_weights(weights: dict[str, float]) -> str:
    """Format weight dict compactly."""
    parts = [f"{k}={v:.3f}" for k, v in sorted(weights.items())]
    return "{" + ", ".join(parts) + "}"


@app.command()
def compare(
    scenario_dir: Annotated[
        Path, typer.Option(help="Directory containing scenario YAML files.")
    ] = Path("scenarios"),
    agent: Annotated[
        str, typer.Option(help="Agent type to compare.")
    ] = "rule-based",
    seed: Annotated[int, typer.Option(help="Random seed.")] = 42,
) -> None:
    """Compare agent traces against production corpus patterns."""
    scenarios = load_all_scenarios(scenario_dir)
    tools = _get_tools()
    ag = _make_agent(agent, tools, seed)
    analyzer = TraceAnalyzer(DEFAULT_CORPUS)
    logger = TraceLogger()

    traces = []
    for scenario_def in scenarios:
        max_steps = max(1, sum(p.max_steps for p in scenario_def.phases))
        config = EpisodeConfig(max_steps=max_steps, token_budget=scenario_def.token_budget, seed=seed)
        env = make_env(config=config)
        obs, _info = env.reset(seed=seed)
        ag.reset()

        total_reward = 0.0
        terminated = False
        truncated = False
        step_num = 0

        while not terminated and not truncated:
            action_str = ag.act(obs)
            obs, reward, terminated, truncated, info = env.step(action_str)
            total_reward += reward
            step_num += 1

            parsed = json.loads(action_str)
            logger.log_step(
                step=step_num,
                server=parsed["server"],
                action=parsed["action"],
                params=parsed.get("params", {}),
                success=info.get("success", False),
            )

        trace = logger.finish_episode(agent, scenario_def.name, total_reward)
        traces.append(trace)

    results = analyzer.compare_traces(traces)

    typer.echo(f"Agent: {agent}  |  Corpus: {len(DEFAULT_CORPUS)} patterns")
    typer.echo("---")

    name_width = max(len(r["scenario_name"]) for r in results)
    header = f"{'Scenario':<{name_width}}  {'Best Pattern':<28}  {'Score':>6}  {'Reward':>8}"
    typer.echo(header)
    typer.echo("-" * len(header))

    for r in results:
        pattern_name = r["best_pattern"] or "(none)"
        typer.echo(
            f"{r['scenario_name']:<{name_width}}  {pattern_name:<28}  "
            f"{r['best_score']:>6.3f}  {r['total_reward']:>+8.3f}"
        )


if __name__ == "__main__":
    app()
