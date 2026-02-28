"""MCPAgentEnv — Gymnasium environment for MCP agent evaluation."""

from __future__ import annotations

import json
import string
from typing import Any, ClassVar

import gymnasium as gym
from gymnasium import spaces

from mcp_gym.mock_surface import MockMCPSurface
from mcp_gym.types import EpisodeConfig, ToolNotAvailableError


class MCPAgentEnv(gym.Env):  # type: ignore[type-arg]
    """Gymnasium environment for MCP agent evaluation.

    Uses MockMCPSurface for deterministic evaluation.
    Compatible with gymnasium.utils.env_checker.check_env().

    Actions are JSON strings describing MCP tool calls.
    Observations are JSON strings describing environment state.
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": ["human", "json"]}  # type: ignore[misc]

    def __init__(
        self,
        surface: MockMCPSurface | None = None,
        config: EpisodeConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.surface = surface or MockMCPSurface()
        self.config = config or EpisodeConfig()
        self.render_mode = render_mode

        # Character set: printable ASCII + space (required for JSON)
        _charset = string.printable.strip() + " "

        # Action space: text-based (agent provides JSON string)
        self.action_space = spaces.Text(
            min_length=2,
            max_length=2048,
            charset=_charset,
        )

        # Observation space: text-based (environment returns JSON string)
        self.observation_space = spaces.Text(
            min_length=0,
            max_length=8192,
            charset=_charset,
        )

        self._step_count: int = 0
        self._token_budget: int = self.config.token_budget
        self._call_history: list[dict[str, Any]] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Reset environment for new episode.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options (unused currently).

        Returns:
            Tuple of (observation, info).
        """
        super().reset(seed=seed)
        self.surface.reset()
        self._step_count = 0
        self._token_budget = self.config.token_budget
        self._call_history = []

        observation = self._make_observation(None)
        info: dict[str, Any] = {"step": 0, "token_budget": self._token_budget}
        return observation, info

    def step(
        self, action: str
    ) -> tuple[str, float, bool, bool, dict[str, Any]]:
        """Execute one step: parse action JSON, call mock surface, return observation.

        Args:
            action: JSON string with keys: server, action, params.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        self._step_count += 1

        # Parse action JSON
        try:
            action_dict = json.loads(action)
            server = action_dict.get("server", "")
            action_name = action_dict.get("action", "")
            params = action_dict.get("params", {})
        except (json.JSONDecodeError, AttributeError):
            obs = self._make_observation({"error": "Invalid action JSON"})
            return obs, -0.5, False, False, {"parse_error": True}

        # Call mock surface
        result: dict[str, Any]
        success: bool
        try:
            result = self.surface.call(server, action_name, params)
            success = True
        except ToolNotAvailableError as exc:
            result = {"error": str(exc)}
            success = False
        except Exception as exc:
            result = {"error": f"Unexpected error: {exc}"}
            success = False

        self._call_history.append({
            "step": self._step_count,
            "server": server,
            "action": action_name,
            "params": params,
            "result": result,
            "success": success,
        })

        # Deduct from token budget (approximate: 1 token per 4 chars)
        tokens_used = len(action) // 4
        self._token_budget -= tokens_used

        # Check termination conditions
        terminated = self._step_count >= self.config.max_steps
        truncated = self._token_budget <= 0

        # Basic reward (placeholder -- MultiDimensionalReward in future phase)
        reward = 0.1 if success else -0.1

        obs = self._make_observation(result)
        info: dict[str, Any] = {
            "step": self._step_count,
            "token_budget": self._token_budget,
            "success": success,
        }

        return obs, reward, terminated, truncated, info

    def _make_observation(self, last_result: Any) -> str:
        """Create JSON observation string from current state.

        Args:
            last_result: The result from the most recent action (or None).

        Returns:
            JSON string containing the observation.
        """
        obs = {
            "last_result": last_result,
            "step_count": self._step_count,
            "token_budget_remaining": self._token_budget,
            "call_history_length": len(self._call_history),
        }
        return json.dumps(obs)

    def render(self) -> str | None:  # type: ignore[override]
        """Render the environment state.

        Returns:
            JSON string if render_mode is 'json', None otherwise.
        """
        if self.render_mode == "human":
            print(
                f"Step {self._step_count} | "
                f"Budget: {self._token_budget} | "
                f"Calls: {len(self._call_history)}"
            )
            return None
        elif self.render_mode == "json":
            return json.dumps(self._call_history, indent=2)
        return None


def make_env(
    config: EpisodeConfig | None = None,
    render_mode: str | None = None,
) -> MCPAgentEnv:
    """Factory function to create a fully configured MCPAgentEnv with all 4 server stubs.

    Registers all mock handlers for CORSO, EVA, SOUL, and QUANTUM servers.

    Args:
        config: Episode configuration. Defaults to EpisodeConfig().
        render_mode: One of 'human', 'json', or None.

    Returns:
        A configured MCPAgentEnv instance.
    """
    from mcp_gym.stubs.corso import register_all as register_corso
    from mcp_gym.stubs.eva import register_all as register_eva
    from mcp_gym.stubs.quantum import register_all as register_quantum
    from mcp_gym.stubs.soul import register_all as register_soul

    surface = MockMCPSurface()
    register_corso(surface)
    register_eva(surface)
    register_soul(surface)
    register_quantum(surface)

    return MCPAgentEnv(
        surface=surface,
        config=config,
        render_mode=render_mode,
    )
