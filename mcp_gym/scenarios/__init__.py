"""Scenario system for MCP Agent Gym.

Provides YAML-based scenario definitions, a loader with validation,
and a state machine that drives multi-phase evaluation episodes.
"""

from mcp_gym.scenarios.loader import load_all_scenarios, load_scenario, validate_scenario
from mcp_gym.scenarios.schema import (
    CriterionDef,
    PhaseDefinition,
    ScenarioDefinition,
)
from mcp_gym.scenarios.state_machine import ScenarioStateMachine, StepResult

__all__ = [
    "CriterionDef",
    "PhaseDefinition",
    "ScenarioDefinition",
    "ScenarioStateMachine",
    "StepResult",
    "load_all_scenarios",
    "load_scenario",
    "validate_scenario",
]
