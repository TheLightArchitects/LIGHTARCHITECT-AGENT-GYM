"""Pydantic models for YAML scenario schema.

Defines the structure of scenario definitions, phases, and criteria
used to drive multi-phase evaluation episodes.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CriterionDef(BaseModel):
    """A single success or failure criterion.

    Supported types:
        - tool_called: Check if a specific server/action was called.
        - tool_not_called: Check that a specific server/action was NOT called.
        - file_exists: Check if a file exists in the virtual filesystem.
        - file_contains: Check if a file contains a regex pattern.
        - param_matches: Check if a tool was called with specific param values.
    """

    type: str
    server: str | None = None
    action: str | None = None
    path: str | None = None
    pattern: str | None = None
    params: dict[str, Any] | None = None


class PhaseDefinition(BaseModel):
    """A single phase within a scenario.

    Each phase defines its own available tools, entry state (filesystem
    contents and context variables), success/failure criteria, step limit,
    and transitions to other phases.
    """

    name: str
    description: str
    available_tools: list[str] = Field(default_factory=list)
    entry_state: dict[str, Any] = Field(default_factory=dict)
    success_criteria: list[CriterionDef] = Field(default_factory=list)
    failure_criteria: list[CriterionDef] = Field(default_factory=list)
    max_steps: int = 10
    transitions: dict[str, str | None] = Field(default_factory=dict)


class ScenarioDefinition(BaseModel):
    """Top-level scenario definition loaded from YAML.

    A scenario consists of one or more phases, each with its own tool
    availability, success criteria, and transitions. The state machine
    drives the agent through phases in sequence.
    """

    name: str
    domain: str
    description: str
    token_budget: int = 2048
    phases: list[PhaseDefinition]
    expected_sequence: list[dict[str, Any]] = Field(default_factory=list)
    forbidden_actions: list[str] = Field(default_factory=list)
    escalation_points: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
