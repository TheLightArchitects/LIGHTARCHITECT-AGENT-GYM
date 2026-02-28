"""Tests for MCPAgentEnv Gymnasium environment."""

from __future__ import annotations

import json

import pytest

from mcp_gym.env import MCPAgentEnv, make_env
from mcp_gym.mock_surface import MockMCPSurface
from mcp_gym.types import EpisodeConfig


class TestMCPAgentEnv:
    """Core environment tests."""

    def _make_env_with_stub(self) -> MCPAgentEnv:
        """Create an env with a simple stub registered."""
        surface = MockMCPSurface()
        surface.register("corso/guard", lambda p: {"clean": True})
        surface.register("eva/speak", lambda p: {"response": "hello"})
        return MCPAgentEnv(surface=surface)

    def test_reset_returns_observation_and_info(self) -> None:
        """Reset returns a valid JSON observation and info dict."""
        env = self._make_env_with_stub()
        obs, info = env.reset(seed=42)

        parsed = json.loads(obs)
        assert parsed["step_count"] == 0
        assert parsed["last_result"] is None
        assert parsed["token_budget_remaining"] == 4096
        assert info["step"] == 0

    def test_step_successful_action(self) -> None:
        """Stepping with a valid action returns positive reward."""
        env = self._make_env_with_stub()
        env.reset(seed=42)

        action = json.dumps({"server": "corso", "action": "guard", "params": {"path": "/src"}})
        obs, reward, terminated, truncated, info = env.step(action)

        parsed = json.loads(obs)
        assert parsed["step_count"] == 1
        assert parsed["last_result"]["clean"] is True
        assert reward == pytest.approx(0.1)
        assert info["success"] is True
        assert not terminated
        assert not truncated

    def test_step_invalid_json_returns_error(self) -> None:
        """Stepping with invalid JSON returns negative reward and error in info."""
        env = self._make_env_with_stub()
        env.reset(seed=42)

        _obs, reward, _terminated, _truncated, info = env.step("not valid json!!!")

        assert reward == pytest.approx(-0.5)
        assert info["parse_error"] is True

    def test_step_unregistered_action(self) -> None:
        """Stepping with an unregistered action returns negative reward."""
        env = self._make_env_with_stub()
        env.reset(seed=42)

        action = json.dumps({"server": "quantum", "action": "scan", "params": {}})
        obs, reward, _terminated, _truncated, info = env.step(action)

        assert reward == pytest.approx(-0.1)
        assert info["success"] is False
        parsed = json.loads(obs)
        assert "error" in parsed["last_result"]

    def test_termination_at_max_steps(self) -> None:
        """Environment terminates after max_steps."""
        config = EpisodeConfig(max_steps=3, token_budget=100000)
        env = MCPAgentEnv(
            surface=self._make_env_with_stub().surface,
            config=config,
        )
        env.reset(seed=42)

        action = json.dumps({"server": "corso", "action": "guard", "params": {}})
        for _i in range(3):
            _obs, _reward, terminated, _truncated, _info = env.step(action)

        assert terminated is True

    def test_truncation_on_budget_exhaustion(self) -> None:
        """Environment truncates when token budget runs out."""
        config = EpisodeConfig(max_steps=1000, token_budget=10)
        env = MCPAgentEnv(
            surface=self._make_env_with_stub().surface,
            config=config,
        )
        env.reset(seed=42)

        # A long action that exhausts the small budget
        action = json.dumps({
            "server": "corso",
            "action": "guard",
            "params": {"path": "/a/very/long/path/that/uses/tokens" * 5},
        })
        _obs, _reward, _terminated, truncated, _info = env.step(action)

        assert truncated is True

    def test_reset_clears_state(self) -> None:
        """Resetting clears step count and call history."""
        env = self._make_env_with_stub()
        env.reset(seed=42)

        action = json.dumps({"server": "corso", "action": "guard", "params": {}})
        env.step(action)
        env.step(action)

        obs, _info = env.reset(seed=99)
        parsed = json.loads(obs)
        assert parsed["step_count"] == 0
        assert parsed["call_history_length"] == 0

    def test_render_json(self) -> None:
        """Render mode 'json' returns call history JSON."""
        env = MCPAgentEnv(
            surface=self._make_env_with_stub().surface,
            render_mode="json",
        )
        env.reset(seed=42)

        action = json.dumps({"server": "corso", "action": "guard", "params": {}})
        env.step(action)

        rendered = env.render()
        assert rendered is not None
        parsed = json.loads(rendered)
        assert len(parsed) == 1
        assert parsed[0]["action"] == "guard"

    def test_token_budget_decreases(self) -> None:
        """Token budget decreases with each step based on action length."""
        config = EpisodeConfig(token_budget=1000, max_steps=100)
        env = MCPAgentEnv(
            surface=self._make_env_with_stub().surface,
            config=config,
        )
        env.reset(seed=42)

        action = json.dumps({"server": "corso", "action": "guard", "params": {}})
        _, _, _, _, info = env.step(action)

        expected_tokens_used = len(action) // 4
        assert info["token_budget"] == 1000 - expected_tokens_used


class TestMakeEnv:
    """Tests for the make_env factory function."""

    def test_make_env_creates_configured_env(self) -> None:
        """make_env returns an env with all 4 servers registered."""
        env = make_env()
        _obs, info = env.reset(seed=42)

        # Test CORSO
        action = json.dumps({"server": "corso", "action": "guard", "params": {}})
        _obs, _reward, _, _, info = env.step(action)
        assert info["success"] is True

        # Test EVA
        action = json.dumps({"server": "eva", "action": "speak", "params": {"message": "hi"}})
        _obs, _reward, _, _, info = env.step(action)
        assert info["success"] is True

        # Test SOUL
        action = json.dumps({"server": "soul", "action": "stats", "params": {}})
        _obs, _reward, _, _, info = env.step(action)
        assert info["success"] is True

        # Test QUANTUM
        action = json.dumps({"server": "quantum", "action": "scan", "params": {}})
        _obs, _reward, _, _, info = env.step(action)
        assert info["success"] is True

    def test_make_env_accepts_config(self) -> None:
        """make_env respects the provided EpisodeConfig."""
        config = EpisodeConfig(max_steps=5, token_budget=512)
        env = make_env(config=config)
        assert env.config.max_steps == 5
        assert env.config.token_budget == 512


class TestEnvChecker:
    """Run Gymnasium's built-in env checker."""

    def test_gymnasium_env_checker(self) -> None:
        """The environment passes gymnasium.utils.env_checker.check_env()."""
        from gymnasium.utils.env_checker import check_env

        env = make_env()
        # check_env raises if there are issues; no assertion needed beyond no-raise
        check_env(env, skip_render_check=True)
