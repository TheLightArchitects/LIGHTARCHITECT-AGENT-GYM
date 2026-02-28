"""Tests for MCP Agent Gym agent implementations."""

from __future__ import annotations

import importlib
import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from mcp_gym.agents import (
    BaseAgent,
    HillClimberAgent,
    LLMAgent,
    RandomAgent,
    RuleBasedAgent,
)


def _has_anthropic() -> bool:
    """Check if the anthropic package is available."""
    try:
        importlib.import_module("anthropic")
    except ImportError:
        return False
    return True

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TOOLS: list[str] = [
    "corso/guard",
    "corso/code_review",
    "corso/speak",
    "corso/scout",
    "corso/sniff",
    "corso/fetch",
    "corso/chase",
    "corso/deploy",
    "eva/speak",
    "eva/memory",
    "eva/research",
    "soul/write_note",
    "soul/helix",
    "soul/stats",
    "quantum/scan",
    "quantum/sweep",
    "quantum/trace",
    "quantum/theorize",
    "quantum/verify",
    "quantum/probe",
]


def _parse_action(action_str: str) -> dict:
    """Parse and validate an action JSON string."""
    parsed = json.loads(action_str)
    assert "server" in parsed, "Action missing 'server'"
    assert "action" in parsed, "Action missing 'action'"
    assert "params" in parsed, "Action missing 'params'"
    return parsed


def _action_tool_key(action_str: str) -> str:
    """Extract 'server/action' from an action JSON string."""
    parsed = json.loads(action_str)
    return f"{parsed['server']}/{parsed['action']}"


# ---------------------------------------------------------------------------
# BaseAgent interface
# ---------------------------------------------------------------------------

class TestBaseAgent:
    """BaseAgent is abstract and cannot be instantiated directly."""

    def test_cannot_instantiate_abc(self) -> None:
        """BaseAgent raises TypeError if instantiated directly."""
        with pytest.raises(TypeError):
            BaseAgent(available_tools=TOOLS)  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# RandomAgent
# ---------------------------------------------------------------------------

class TestRandomAgent:
    """Tests for the RandomAgent baseline."""

    def test_produces_valid_json_action(self) -> None:
        """act() returns a parseable JSON action with required keys."""
        agent = RandomAgent(TOOLS, seed=42)
        obs = json.dumps({"last_result": None, "step_count": 0,
                          "token_budget_remaining": 4096,
                          "call_history_length": 0})
        action = agent.act(obs)
        parsed = _parse_action(action)
        assert isinstance(parsed["params"], dict)

    def test_action_has_empty_params(self) -> None:
        """RandomAgent always passes empty params."""
        agent = RandomAgent(TOOLS, seed=1)
        obs = json.dumps({"last_result": None, "step_count": 0,
                          "token_budget_remaining": 4096,
                          "call_history_length": 0})
        action = agent.act(obs)
        parsed = json.loads(action)
        assert parsed["params"] == {}

    def test_action_uses_available_tool(self) -> None:
        """Every action references a tool from available_tools."""
        agent = RandomAgent(TOOLS, seed=7)
        obs = json.dumps({"last_result": None, "step_count": 0,
                          "token_budget_remaining": 4096,
                          "call_history_length": 0})
        for _ in range(20):
            tool_key = _action_tool_key(agent.act(obs))
            assert tool_key in TOOLS, f"{tool_key} not in available_tools"

    def test_reproducible_with_same_seed(self) -> None:
        """Two RandomAgents with the same seed produce identical sequences."""
        obs = json.dumps({"last_result": None, "step_count": 0,
                          "token_budget_remaining": 4096,
                          "call_history_length": 0})
        agent_a = RandomAgent(TOOLS, seed=99)
        agent_b = RandomAgent(TOOLS, seed=99)

        for _ in range(10):
            assert agent_a.act(obs) == agent_b.act(obs)

    def test_different_seeds_differ(self) -> None:
        """Two RandomAgents with different seeds diverge."""
        obs = json.dumps({"last_result": None, "step_count": 0,
                          "token_budget_remaining": 4096,
                          "call_history_length": 0})
        agent_a = RandomAgent(TOOLS, seed=1)
        agent_b = RandomAgent(TOOLS, seed=2)

        actions_a = [agent_a.act(obs) for _ in range(10)]
        actions_b = [agent_b.act(obs) for _ in range(10)]
        # Not all 10 should match (probabilistically near-certain)
        assert actions_a != actions_b

    def test_reset_restores_determinism(self) -> None:
        """After reset(), the agent replays the same sequence."""
        obs = json.dumps({"last_result": None, "step_count": 0,
                          "token_budget_remaining": 4096,
                          "call_history_length": 0})
        agent = RandomAgent(TOOLS, seed=42)
        first_run = [agent.act(obs) for _ in range(5)]

        agent.reset()
        second_run = [agent.act(obs) for _ in range(5)]

        assert first_run == second_run


# ---------------------------------------------------------------------------
# RuleBasedAgent
# ---------------------------------------------------------------------------

class TestRuleBasedAgent:
    """Tests for the domain-heuristic RuleBasedAgent."""

    def _obs_with_keyword(self, keyword: str) -> str:
        """Build an observation whose last_result contains a keyword."""
        return json.dumps({
            "last_result": {"message": f"Detected {keyword} issue"},
            "step_count": 1,
            "token_budget_remaining": 3000,
            "call_history_length": 1,
        })

    def test_follows_security_sequence(self) -> None:
        """Security-domain observation triggers the security tool sequence."""
        agent = RuleBasedAgent(TOOLS, seed=42)
        obs = self._obs_with_keyword("security")

        actions = [_action_tool_key(agent.act(obs)) for _ in range(3)]
        assert actions == [
            "corso/guard",
            "corso/code_review",
            "corso/speak",
        ]

    def test_follows_investigation_sequence(self) -> None:
        """Investigation-domain observation triggers the investigation sequence."""
        agent = RuleBasedAgent(TOOLS, seed=42)
        obs = self._obs_with_keyword("investigation")

        actions = [_action_tool_key(agent.act(obs)) for _ in range(5)]
        assert actions == [
            "quantum/scan",
            "quantum/sweep",
            "quantum/trace",
            "quantum/theorize",
            "quantum/verify",
        ]

    def test_follows_memory_sequence(self) -> None:
        """Memory-domain observation triggers the memory tool sequence."""
        agent = RuleBasedAgent(TOOLS, seed=42)
        obs = self._obs_with_keyword("memory")

        actions = [_action_tool_key(agent.act(obs)) for _ in range(3)]
        assert actions == [
            "eva/memory",
            "soul/write_note",
            "soul/helix",
        ]

    def test_falls_back_to_random_for_unknown_domain(self) -> None:
        """Observation with no domain keywords triggers random fallback."""
        agent = RuleBasedAgent(TOOLS, seed=42)
        obs = json.dumps({
            "last_result": {"message": "completely unrelated topic xyz"},
            "step_count": 1,
            "token_budget_remaining": 3000,
            "call_history_length": 1,
        })

        # Should still produce a valid action from available_tools
        tool_key = _action_tool_key(agent.act(obs))
        assert tool_key in TOOLS

    def test_action_within_available_tools(self) -> None:
        """All RuleBasedAgent actions are from available_tools."""
        agent = RuleBasedAgent(TOOLS, seed=42)
        obs = self._obs_with_keyword("security")
        for _ in range(10):
            tool_key = _action_tool_key(agent.act(obs))
            assert tool_key in TOOLS

    def test_sequence_wraps_around(self) -> None:
        """Sequence cycles back after exhausting all tools in domain."""
        agent = RuleBasedAgent(TOOLS, seed=42)
        obs = self._obs_with_keyword("security")

        # Security has 3 tools; 4th should wrap to first
        actions = [_action_tool_key(agent.act(obs)) for _ in range(4)]
        assert actions[3] == actions[0]


# ---------------------------------------------------------------------------
# HillClimberAgent
# ---------------------------------------------------------------------------

class TestHillClimberAgent:
    """Tests for the weight-tuning HillClimberAgent."""

    def test_perturb_changes_weights(self) -> None:
        """perturb() modifies at least one weight."""
        agent = HillClimberAgent(TOOLS, seed=42)
        original = dict(agent.weights)
        agent.perturb(magnitude=0.5)
        assert agent.weights != original

    def test_update_records_history(self) -> None:
        """update() appends to training_history."""
        agent = HillClimberAgent(TOOLS, seed=42)
        assert len(agent.training_history) == 0

        agent.perturb()
        agent.update(reward=1.5)
        assert len(agent.training_history) == 1
        assert agent.training_history[0]["episode"] == 1
        assert agent.training_history[0]["reward"] == 1.5
        assert "weights" in agent.training_history[0]

    def test_update_keeps_improvement(self) -> None:
        """When reward improves, weights are kept (not reversed)."""
        agent = HillClimberAgent(TOOLS, seed=42)
        agent.perturb(magnitude=0.5)
        weights_after_perturb = dict(agent.weights)

        # First update with a positive reward (better than -inf default)
        agent.update(reward=1.0)
        assert agent.weights == weights_after_perturb

    def test_update_reverses_on_decline(self) -> None:
        """When reward declines, perturbation is reversed."""
        agent = HillClimberAgent(TOOLS, seed=42)

        # First episode: establish a baseline
        agent.perturb(magnitude=0.5)
        agent.update(reward=5.0)

        # Second episode: worse reward
        dict(agent.weights)
        agent.perturb(magnitude=0.5)
        weights_perturbed = dict(agent.weights)
        agent.update(reward=1.0)  # Worse than 5.0

        # At least one weight should have moved away from perturbed value
        assert agent.weights != weights_perturbed

    def test_training_history_grows(self) -> None:
        """Multiple episodes build up training history."""
        agent = HillClimberAgent(TOOLS, seed=42)
        for i in range(5):
            agent.perturb()
            agent.update(reward=float(i))

        assert len(agent.training_history) == 5
        assert agent.training_history[4]["episode"] == 5

    def test_action_within_available_tools(self) -> None:
        """HillClimberAgent actions are from available_tools."""
        agent = HillClimberAgent(TOOLS, seed=42)
        obs = json.dumps({
            "last_result": {"message": "security scan needed"},
            "step_count": 0,
            "token_budget_remaining": 4096,
            "call_history_length": 0,
        })
        for _ in range(10):
            tool_key = _action_tool_key(agent.act(obs))
            assert tool_key in TOOLS


# ---------------------------------------------------------------------------
# LLMAgent
# ---------------------------------------------------------------------------

class TestLLMAgent:
    """Tests for the Claude API LLMAgent."""

    def test_raises_without_api_key(self) -> None:
        """LLMAgent raises ValueError when no API key is provided."""
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            LLMAgent(TOOLS, api_key=None)

    def test_raises_import_error_without_anthropic(self) -> None:
        """act() raises ImportError when anthropic package is missing."""
        # Create the agent with a dummy key
        agent = LLMAgent(TOOLS, api_key="sk-test-dummy-key-for-unit-test")

        # Temporarily hide the anthropic module
        original = sys.modules.get("anthropic")
        sys.modules["anthropic"] = None  # type: ignore[assignment]

        try:
            obs = json.dumps({"last_result": None, "step_count": 0,
                              "token_budget_remaining": 4096,
                              "call_history_length": 0})
            with pytest.raises(ImportError, match="anthropic"):
                agent.act(obs)
        finally:
            # Restore
            if original is not None:
                sys.modules["anthropic"] = original
            else:
                sys.modules.pop("anthropic", None)

    @pytest.mark.skipif(
        not _has_anthropic(),
        reason="anthropic package not installed",
    )
    def test_parses_valid_json_response(self) -> None:
        """LLMAgent correctly parses a well-formed Claude response."""
        agent = LLMAgent(TOOLS, api_key="sk-test-dummy")

        # Mock the anthropic client
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = json.dumps({
            "server": "corso",
            "action": "guard",
            "params": {"path": "/src"},
        })
        mock_response.content = [mock_content]

        with patch("anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_cls.return_value = mock_client

            obs = json.dumps({"last_result": None, "step_count": 0,
                              "token_budget_remaining": 4096,
                              "call_history_length": 0})
            action = agent.act(obs)

        parsed = json.loads(action)
        assert parsed["server"] == "corso"
        assert parsed["action"] == "guard"
        assert parsed["params"] == {"path": "/src"}

    @pytest.mark.skipif(
        not _has_anthropic(),
        reason="anthropic package not installed",
    )
    def test_fallback_on_unparseable_response(self) -> None:
        """LLMAgent falls back to first available tool on bad response."""
        agent = LLMAgent(TOOLS, api_key="sk-test-dummy")

        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "I think we should scan for vulnerabilities..."
        mock_response.content = [mock_content]

        with patch("anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_cls.return_value = mock_client

            obs = json.dumps({"last_result": None, "step_count": 0,
                              "token_budget_remaining": 4096,
                              "call_history_length": 0})
            action = agent.act(obs)

        tool_key = _action_tool_key(action)
        assert tool_key in TOOLS

    def test_reset_clears_conversation(self) -> None:
        """reset() clears the internal conversation history."""
        agent = LLMAgent(TOOLS, api_key="sk-test-dummy")

        # Simulate a conversation by directly setting state
        agent._conversation = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "response"},
        ]

        agent.reset()
        assert agent._conversation == []
