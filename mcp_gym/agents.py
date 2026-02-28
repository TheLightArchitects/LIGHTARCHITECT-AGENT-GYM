"""Agent implementations for MCP Agent Gym.

Four agent types with increasing sophistication:
1. RandomAgent — uniform random baseline
2. RuleBasedAgent — domain-keyword heuristic sequences
3. HillClimberAgent — weight-tuned RuleBasedAgent
4. LLMAgent — Claude API upper bound
"""

from __future__ import annotations

import copy
import json
import random
from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Base class for all MCP gym agents."""

    def __init__(self, available_tools: list[str], seed: int = 42) -> None:
        """Initialize with available tool list and random seed.

        Args:
            available_tools: List of "server/action" strings the agent can use.
            seed: Random seed for reproducibility.
        """
        self.available_tools = available_tools
        self.seed = seed

    @abstractmethod
    def act(self, observation: str) -> str:
        """Given an observation JSON string, return an action JSON string.

        Args:
            observation: JSON string from MCPAgentEnv.

        Returns:
            JSON string with keys: server, action, params.
        """
        ...

    def reset(self) -> None:  # noqa: B027
        """Reset agent state for new episode."""


class RandomAgent(BaseAgent):
    """Baseline agent that picks tools uniformly at random.

    Uses a seeded random.Random instance for reproducibility.
    Always passes empty params.
    """

    def __init__(self, available_tools: list[str], seed: int = 42) -> None:
        super().__init__(available_tools, seed)
        self._rng = random.Random(seed)

    def act(self, observation: str) -> str:
        """Pick a random tool and return a minimal action JSON string."""
        tool = self._rng.choice(self.available_tools)
        server, action = tool.split("/", 1)
        return json.dumps({"server": server, "action": action, "params": {}})

    def reset(self) -> None:
        """Re-seed the RNG for deterministic episodes."""
        self._rng = random.Random(self.seed)


# ---------------------------------------------------------------------------
# Domain rules: keyword -> ordered tool sequence
# ---------------------------------------------------------------------------

DOMAIN_RULES: dict[str, list[str]] = {
    "security": [
        "corso/guard",
        "corso/code_review",
        "corso/speak",
    ],
    "investigation": [
        "quantum/scan",
        "quantum/sweep",
        "quantum/trace",
        "quantum/theorize",
        "quantum/verify",
    ],
    "memory": [
        "eva/memory",
        "soul/write_note",
        "soul/helix",
    ],
    "build": [
        "corso/scout",
        "corso/sniff",
        "corso/guard",
        "corso/code_review",
    ],
    "deploy": [
        "corso/chase",
        "corso/guard",
        "corso/deploy",
    ],
    "research": [
        "corso/fetch",
        "eva/research",
        "quantum/probe",
    ],
}


def _detect_domain(
    observation_text: str,
    weights: dict[str, float] | None = None,
) -> str | None:
    """Scan observation text for domain keywords, return best match.

    Args:
        observation_text: Raw observation string to scan.
        weights: Optional weight multipliers per domain keyword.

    Returns:
        The domain keyword with the highest weighted match count,
        or None if no keywords are found.
    """
    if weights is None:
        weights = {k: 1.0 for k in DOMAIN_RULES}

    lowered = observation_text.lower()
    best_domain: str | None = None
    best_score: float = 0.0

    for domain in DOMAIN_RULES:
        count = lowered.count(domain)
        if count > 0:
            score = count * weights.get(domain, 1.0)
            if score > best_score:
                best_score = score
                best_domain = domain

    return best_domain


class RuleBasedAgent(BaseAgent):
    """Heuristic agent that follows domain-specific tool sequences.

    Detects the domain from observation keywords, then steps through
    the corresponding tool sequence in order. Falls back to random
    selection when no domain is detected.
    """

    def __init__(
        self,
        available_tools: list[str],
        seed: int = 42,
        weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__(available_tools, seed)
        self.weights: dict[str, float] = (
            weights if weights is not None
            else {k: 1.0 for k in DOMAIN_RULES}
        )
        self._rng = random.Random(seed)
        self._current_domain: str | None = None
        self._sequence_index: int = 0

    def act(self, observation: str) -> str:
        """Select the next tool based on domain detection or fallback to random."""
        domain = _detect_domain(observation, self.weights)

        # If domain changed or is new, restart the sequence
        if domain != self._current_domain:
            self._current_domain = domain
            self._sequence_index = 0

        if domain is not None:
            sequence = DOMAIN_RULES[domain]
            # Filter to tools the agent actually has available
            usable = [t for t in sequence if t in self.available_tools]
            if usable:
                idx = self._sequence_index % len(usable)
                tool = usable[idx]
                self._sequence_index += 1
                server, action = tool.split("/", 1)
                return json.dumps({
                    "server": server,
                    "action": action,
                    "params": {},
                })

        # Fallback: random
        return self._random_action()

    def _random_action(self) -> str:
        """Pick a random tool as fallback."""
        tool = self._rng.choice(self.available_tools)
        server, action = tool.split("/", 1)
        return json.dumps({"server": server, "action": action, "params": {}})

    def reset(self) -> None:
        """Reset domain tracking and RNG for new episode."""
        self._current_domain = None
        self._sequence_index = 0
        self._rng = random.Random(self.seed)


class HillClimberAgent(BaseAgent):
    """Weight-tuned RuleBasedAgent trained via hill climbing.

    Wraps a RuleBasedAgent and adjusts its domain weights after each
    episode based on reward delta. Maintains a training history.
    """

    def __init__(
        self,
        available_tools: list[str],
        seed: int = 42,
        weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__(available_tools, seed)
        self._rng = random.Random(seed)
        self._weights: dict[str, float] = (
            weights if weights is not None
            else {k: 1.0 for k in DOMAIN_RULES}
        )
        self._inner = RuleBasedAgent(
            available_tools, seed=seed, weights=self._weights,
        )
        self._best_reward: float = float("-inf")
        self._best_weights: dict[str, float] = copy.deepcopy(self._weights)
        self._last_perturbation: dict[str, float] = {}
        self.training_history: list[dict[str, Any]] = []

    @property
    def weights(self) -> dict[str, float]:
        """Current domain weights."""
        return self._weights

    def act(self, observation: str) -> str:
        """Delegate to the inner RuleBasedAgent."""
        return self._inner.act(observation)

    def perturb(self, magnitude: float = 0.1) -> None:
        """Randomly adjust one weight by +/- magnitude.

        Args:
            magnitude: Maximum absolute change to apply.
        """
        if not self._weights:
            return

        domain = self._rng.choice(list(self._weights.keys()))
        delta = self._rng.uniform(-magnitude, magnitude)
        self._last_perturbation = {domain: delta}
        self._weights[domain] += delta

        # Sync inner agent weights
        self._inner.weights = self._weights

    def update(self, reward: float) -> None:
        """Update weights based on episode reward.

        If reward improved over best seen, keep perturbation direction
        and record the new best. Otherwise, reverse the perturbation.

        Args:
            reward: Total reward from the episode.
        """
        episode_num = len(self.training_history) + 1

        if reward > self._best_reward:
            # Improvement: keep the perturbation, record new best
            self._best_reward = reward
            self._best_weights = copy.deepcopy(self._weights)
        else:
            # No improvement: reverse the perturbation
            for domain, delta in self._last_perturbation.items():
                self._weights[domain] -= 2 * delta
            self._inner.weights = self._weights

        self.training_history.append({
            "episode": episode_num,
            "reward": reward,
            "weights": copy.deepcopy(self._weights),
        })

    def reset(self) -> None:
        """Reset inner agent for new episode."""
        self._inner.reset()


class LLMAgent(BaseAgent):
    """Claude API agent — production-grade upper bound.

    Constructs a system prompt with available tools and scenario context,
    then calls the Anthropic API to decide the next action.
    Falls back to a pass action if response parsing fails.
    """

    def __init__(
        self,
        available_tools: list[str],
        seed: int = 42,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
    ) -> None:
        super().__init__(available_tools, seed)
        if api_key is None:
            msg = (
                "LLM agent requires ANTHROPIC_API_KEY. "
                "Pass api_key= or set the environment variable."
            )
            raise ValueError(msg)
        self._api_key = api_key
        self._model = model
        self._conversation: list[dict[str, str]] = []

    def act(self, observation: str) -> str:
        """Call Claude API with observation, parse action from response.

        Raises:
            ImportError: If the anthropic package is not installed.
        """
        try:
            import anthropic  # type: ignore[import-not-found]
        except ImportError as exc:
            msg = (
                "LLMAgent requires the 'anthropic' package. "
                "Install it with: pip install anthropic"
            )
            raise ImportError(msg) from exc

        system_prompt = self._build_system_prompt()
        self._conversation.append({"role": "user", "content": observation})

        client = anthropic.Anthropic(api_key=self._api_key)
        response = client.messages.create(
            model=self._model,
            max_tokens=512,
            system=system_prompt,
            messages=self._conversation,
        )

        raw_text = response.content[0].text
        self._conversation.append({"role": "assistant", "content": raw_text})

        return self._parse_action(raw_text)

    def _build_system_prompt(self) -> str:
        """Build the system prompt explaining available tools."""
        tools_text = "\n".join(f"  - {t}" for t in self.available_tools)
        return (
            "You are an MCP agent selecting tools to complete a scenario.\n"
            "Available tools:\n"
            f"{tools_text}\n\n"
            "Respond with ONLY a JSON object: "
            '{"server": "...", "action": "...", "params": {...}}\n'
            "No explanation, no markdown, just the JSON."
        )

    def _parse_action(self, raw_text: str) -> str:
        """Extract valid action JSON from LLM response.

        Falls back to a safe pass action if parsing fails.
        """
        text = raw_text.strip()

        # Try direct parse
        try:
            parsed = json.loads(text)
            if "server" in parsed and "action" in parsed:
                parsed.setdefault("params", {})
                return json.dumps(parsed)
        except (json.JSONDecodeError, TypeError):
            pass

        # Try extracting JSON from markdown code blocks
        for marker in ("```json", "```"):
            if marker in text:
                start = text.index(marker) + len(marker)
                end = text.index("```", start) if "```" in text[start:] else len(text)
                candidate = text[start:end].strip()
                try:
                    parsed = json.loads(candidate)
                    if "server" in parsed and "action" in parsed:
                        parsed.setdefault("params", {})
                        return json.dumps(parsed)
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass

        # Fallback: first available tool, empty params
        fallback_tool = self.available_tools[0]
        server, action = fallback_tool.split("/", 1)
        return json.dumps({"server": server, "action": action, "params": {}})

    def reset(self) -> None:
        """Clear conversation history for new episode."""
        self._conversation = []
