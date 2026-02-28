"""Tests for MCP server stubs — one test per server verifying all actions register and work."""

from __future__ import annotations

from mcp_gym.mock_surface import MockMCPSurface
from mcp_gym.stubs.corso import CORSO_ACTIONS
from mcp_gym.stubs.corso import register_all as register_corso
from mcp_gym.stubs.eva import EVA_ACTIONS
from mcp_gym.stubs.eva import register_all as register_eva
from mcp_gym.stubs.quantum import QUANTUM_ACTIONS
from mcp_gym.stubs.quantum import register_all as register_quantum
from mcp_gym.stubs.soul import SOUL_ACTIONS
from mcp_gym.stubs.soul import register_all as register_soul


class TestCORSOStub:
    """Tests for CORSO MCP stub."""

    def test_all_24_actions_registered(self) -> None:
        """All 24 CORSO actions are registered on the surface."""
        surface = MockMCPSurface()
        register_corso(surface)

        registered = surface.registered_actions()
        assert len([a for a in registered if a.startswith("corso/")]) == 24

    def test_guard_returns_scan_result(self) -> None:
        """The guard action returns a security scan result."""
        surface = MockMCPSurface()
        register_corso(surface)

        result = surface.call("corso", "guard", {"path": "/src"})
        assert result["clean"] is True
        assert result["patterns_checked"] == 4997

    def test_speak_returns_prompt_mode(self) -> None:
        """The speak action returns prompt_mode response."""
        surface = MockMCPSurface()
        register_corso(surface)

        result = surface.call("corso", "speak", {"message": "Hello CORSO"})
        assert result["prompt_mode"] is True
        assert "CORSO" in result["response"]

    def test_all_actions_callable(self) -> None:
        """Every registered CORSO action is callable without error."""
        surface = MockMCPSurface()
        register_corso(surface)

        for action_name in CORSO_ACTIONS:
            result = surface.call("corso", action_name, {})
            assert isinstance(result, dict), f"corso/{action_name} did not return dict"


class TestEVAStub:
    """Tests for EVA MCP stub."""

    def test_all_9_tools_registered(self) -> None:
        """All 9 EVA tools are registered on the surface."""
        surface = MockMCPSurface()
        register_eva(surface)

        registered = surface.registered_actions()
        assert len([a for a in registered if a.startswith("eva/")]) == 9

    def test_speak_returns_response(self) -> None:
        """The speak tool returns a conversation response."""
        surface = MockMCPSurface()
        register_eva(surface)

        result = surface.call("eva", "speak", {"message": "Hello EVA"})
        assert result["prompt_mode"] is True
        assert "EVA" in result["response"]

    def test_memory_store_operation(self) -> None:
        """The memory tool supports store operation."""
        surface = MockMCPSurface()
        register_eva(surface)

        result = surface.call("eva", "memory", {
            "subcommand": "remember",
            "operation": "store",
            "content": "Important memory",
        })
        assert result["stored"] is True

    def test_all_tools_callable(self) -> None:
        """Every registered EVA tool is callable without error."""
        surface = MockMCPSurface()
        register_eva(surface)

        for action_name in EVA_ACTIONS:
            result = surface.call("eva", action_name, {})
            assert isinstance(result, dict), f"eva/{action_name} did not return dict"


class TestSOULStub:
    """Tests for SOUL MCP stub."""

    def test_all_11_subtools_registered(self) -> None:
        """All 11 SOUL sub-tools are registered on the surface."""
        surface = MockMCPSurface()
        register_soul(surface)

        registered = surface.registered_actions()
        assert len([a for a in registered if a.startswith("soul/")]) == 11

    def test_helix_returns_entries(self) -> None:
        """The helix sub-tool returns consciousness entries."""
        surface = MockMCPSurface()
        register_soul(surface)

        result = surface.call("soul", "helix", {"sibling": "eva"})
        assert result["count"] >= 1
        assert result["entries"][0]["significance"] > 0

    def test_stats_returns_vault_statistics(self) -> None:
        """The stats sub-tool returns vault statistics."""
        surface = MockMCPSurface()
        register_soul(surface)

        result = surface.call("soul", "stats", {})
        assert result["total_entries"] > 0
        assert "strand_frequency" in result

    def test_all_subtools_callable(self) -> None:
        """Every registered SOUL sub-tool is callable without error."""
        surface = MockMCPSurface()
        register_soul(surface)

        for action_name in SOUL_ACTIONS:
            result = surface.call("soul", action_name, {})
            assert isinstance(result, dict), f"soul/{action_name} did not return dict"


class TestQUANTUMStub:
    """Tests for QUANTUM MCP stub."""

    def test_all_13_actions_registered(self) -> None:
        """All 13 QUANTUM actions are registered on the surface."""
        surface = MockMCPSurface()
        register_quantum(surface)

        registered = surface.registered_actions()
        assert len([a for a in registered if a.startswith("quantum/")]) == 13

    def test_scan_returns_triage(self) -> None:
        """The scan action returns triage results."""
        surface = MockMCPSurface()
        register_quantum(surface)

        result = surface.call("quantum", "scan", {})
        assert result["phase"] == "scan"
        assert "findings" in result
        assert result["recommended_next"] == "sweep"

    def test_workflow_returns_phases(self) -> None:
        """The workflow action returns executed phases."""
        surface = MockMCPSurface()
        register_quantum(surface)

        result = surface.call("quantum", "workflow", {"template": "full-investigation"})
        assert result["status"] == "complete"
        assert len(result["phases_executed"]) > 0

    def test_all_actions_callable(self) -> None:
        """Every registered QUANTUM action is callable without error."""
        surface = MockMCPSurface()
        register_quantum(surface)

        for action_name in QUANTUM_ACTIONS:
            result = surface.call("quantum", action_name, {})
            assert isinstance(result, dict), f"quantum/{action_name} did not return dict"
