"""YAML scenario loader and validator.

Loads ScenarioDefinition objects from YAML files and validates that
all referenced tools are registered on the mock surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from mcp_gym.scenarios.schema import ScenarioDefinition

if TYPE_CHECKING:
    from mcp_gym.mock_surface import MockMCPSurface


def load_scenario(path: Path) -> ScenarioDefinition:
    """Load a single scenario from a YAML file.

    Args:
        path: Path to a YAML file containing one scenario definition.

    Returns:
        Parsed and validated ScenarioDefinition.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If the YAML is malformed.
        pydantic.ValidationError: If the data does not match the schema.
    """
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    return ScenarioDefinition.model_validate(raw)


def load_all_scenarios(directory: Path) -> list[ScenarioDefinition]:
    """Load all YAML scenarios from a directory.

    Scans for files matching ``*.yaml`` and ``*.yml`` (non-recursive).

    Args:
        directory: Directory containing scenario YAML files.

    Returns:
        List of parsed ScenarioDefinition objects, sorted by name.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    if not directory.is_dir():
        msg = f"Scenario directory does not exist: {directory}"
        raise FileNotFoundError(msg)

    scenarios: list[ScenarioDefinition] = []
    for ext in ("*.yaml", "*.yml"):
        for yaml_path in sorted(directory.glob(ext)):
            scenarios.append(load_scenario(yaml_path))

    scenarios.sort(key=lambda s: s.name)
    return scenarios


def validate_scenario(
    scenario: ScenarioDefinition,
    surface: MockMCPSurface,
) -> list[str]:
    """Validate that all tools referenced in a scenario are registered.

    Checks ``available_tools`` in every phase and ``forbidden_actions``
    at the scenario level against the surface's registered actions.

    Args:
        scenario: The scenario definition to validate.
        surface: The mock surface whose registry to check against.

    Returns:
        List of validation error strings. Empty if valid.
    """
    registered = set(surface.registered_actions())
    errors: list[str] = []

    for phase in scenario.phases:
        for tool in phase.available_tools:
            if tool not in registered:
                errors.append(
                    f"Phase '{phase.name}': tool '{tool}' not registered on surface"
                )

    for action in scenario.forbidden_actions:
        if action not in registered:
            errors.append(
                f"Forbidden action '{action}' not registered on surface "
                f"(may be intentional but cannot be enforced)"
            )

    return errors
