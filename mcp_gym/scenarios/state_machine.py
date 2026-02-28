"""ScenarioStateMachine — drives multi-phase evaluation episodes.

Tracks phase transitions, evaluates success/failure criteria against
the accumulated call history, and manages filesystem state per phase.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from mcp_gym.scenarios.schema import CriterionDef, PhaseDefinition

if TYPE_CHECKING:
    from mcp_gym.mock_surface import MockMCPSurface
    from mcp_gym.scenarios.schema import ScenarioDefinition


class StepResult(BaseModel):
    """Result of processing a single agent step through the state machine."""

    phase_changed: bool = False
    new_phase: str | None = None
    scenario_complete: bool = False
    scenario_failed: bool = False
    criteria_met: list[str] = Field(default_factory=list)
    feedback: str = ""


class ScenarioStateMachine:
    """Drives an agent through a multi-phase scenario.

    Initializes filesystem from each phase's entry_state, checks
    success/failure criteria after each step, and handles transitions.
    """

    def __init__(
        self,
        scenario: ScenarioDefinition,
        surface: MockMCPSurface,
    ) -> None:
        self._scenario = scenario
        self._surface = surface
        self._phase_idx: int = 0
        self._phase_step_count: int = 0
        self._phase_call_history: list[dict[str, Any]] = []
        self._completed: bool = False
        self._failed: bool = False

    @property
    def current_phase(self) -> PhaseDefinition:
        """Return the current phase definition."""
        return self._scenario.phases[self._phase_idx]

    @property
    def is_complete(self) -> bool:
        """Return True if the scenario has been completed (success or failure)."""
        return self._completed or self._failed

    @property
    def phase_index(self) -> int:
        """Return the zero-based index of the current phase."""
        return self._phase_idx

    def reset(self) -> dict[str, Any]:
        """Reset to the first phase. Returns initial observation context.

        Clears the surface state, applies the first phase's entry_state
        to the virtual filesystem, and returns context for the agent.
        """
        self._phase_idx = 0
        self._phase_step_count = 0
        self._phase_call_history = []
        self._completed = False
        self._failed = False
        self._surface.reset()
        self._apply_entry_state(self.current_phase)

        return self._build_phase_context()

    def process_step(
        self,
        server: str,
        action: str,
        params: dict[str, Any],
        result: dict[str, Any],
    ) -> StepResult:
        """Process one agent step. Check criteria, handle transitions.

        Args:
            server: The MCP server that was called.
            action: The action name that was called.
            params: The parameters passed to the action.
            result: The result dict returned by the mock surface.

        Returns:
            StepResult describing what happened.
        """
        if self.is_complete:
            return StepResult(
                scenario_complete=self._completed,
                scenario_failed=self._failed,
                feedback="Scenario already finished.",
            )

        self._phase_step_count += 1
        self._phase_call_history.append({
            "server": server,
            "action": action,
            "params": params,
            "result": result,
        })

        # Check forbidden actions at scenario level
        action_key = f"{server}/{action}"
        if action_key in self._scenario.forbidden_actions:
            self._failed = True
            return StepResult(
                scenario_failed=True,
                feedback=f"Forbidden action used: {action_key}",
            )

        # Check failure criteria first
        phase = self.current_phase
        if phase.failure_criteria and self.check_criteria(phase.failure_criteria, self._phase_call_history):
            self._failed = True
            return StepResult(
                scenario_failed=True,
                feedback=f"Failure criteria met in phase '{phase.name}'.",
            )

        # Check success criteria
        met_criteria: list[str] = []
        success = self._evaluate_criteria_detailed(
            phase.success_criteria,
            self._phase_call_history,
            met_criteria,
        )

        if success:
            return self._handle_phase_success(phase, met_criteria)

        # Check max steps exceeded
        if self._phase_step_count >= phase.max_steps:
            self._failed = True
            return StepResult(
                scenario_failed=True,
                feedback=(
                    f"Max steps ({phase.max_steps}) exceeded "
                    f"in phase '{phase.name}'."
                ),
            )

        return StepResult(
            criteria_met=met_criteria,
            feedback=f"Phase '{phase.name}': step {self._phase_step_count}/{phase.max_steps}",
        )

    def check_criteria(
        self,
        criteria: list[CriterionDef],
        call_history: list[dict[str, Any]],
    ) -> bool:
        """Evaluate a list of criteria against call history.

        All criteria must be met (AND logic) for the check to pass.

        Args:
            criteria: List of criterion definitions to evaluate.
            call_history: List of call records to check against.

        Returns:
            True if ALL criteria are satisfied.
        """
        return all(
            self._evaluate_single_criterion(c, call_history) for c in criteria
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate_single_criterion(
        self,
        criterion: CriterionDef,
        call_history: list[dict[str, Any]],
    ) -> bool:
        """Evaluate one criterion against the call history and filesystem."""
        ctype = criterion.type

        if ctype == "tool_called":
            return self._check_tool_called(criterion, call_history)
        if ctype == "tool_not_called":
            return self._check_tool_not_called(criterion, call_history)
        if ctype == "file_exists":
            return self._check_file_exists(criterion)
        if ctype == "file_contains":
            return self._check_file_contains(criterion)
        if ctype == "param_matches":
            return self._check_param_matches(criterion, call_history)

        return False

    @staticmethod
    def _check_tool_called(
        criterion: CriterionDef,
        call_history: list[dict[str, Any]],
    ) -> bool:
        """Return True if the specified tool was called at least once."""
        return any(
            entry.get("server") == criterion.server
            and entry.get("action") == criterion.action
            for entry in call_history
        )

    @staticmethod
    def _check_tool_not_called(
        criterion: CriterionDef,
        call_history: list[dict[str, Any]],
    ) -> bool:
        """Return True if the specified tool was never called."""
        return not any(
            entry.get("server") == criterion.server
            and entry.get("action") == criterion.action
            for entry in call_history
        )

    def _check_file_exists(self, criterion: CriterionDef) -> bool:
        """Return True if the specified file exists in the virtual filesystem."""
        if criterion.path is None:
            return False
        return self._surface.filesystem.exists(criterion.path)

    def _check_file_contains(self, criterion: CriterionDef) -> bool:
        """Return True if the file exists and contains the regex pattern."""
        if criterion.path is None or criterion.pattern is None:
            return False
        content = self._surface.filesystem.read(criterion.path)
        if content is None:
            return False
        return bool(re.search(criterion.pattern, content))

    @staticmethod
    def _check_param_matches(
        criterion: CriterionDef,
        call_history: list[dict[str, Any]],
    ) -> bool:
        """Return True if a matching call has all specified param key-value pairs."""
        if criterion.params is None:
            return False
        for entry in call_history:
            if (
                entry.get("server") == criterion.server
                and entry.get("action") == criterion.action
            ):
                entry_params = entry.get("params", {})
                if all(
                    entry_params.get(k) == v
                    for k, v in criterion.params.items()
                ):
                    return True
        return False

    def _evaluate_criteria_detailed(
        self,
        criteria: list[CriterionDef],
        call_history: list[dict[str, Any]],
        met_out: list[str],
    ) -> bool:
        """Evaluate criteria, populating met_out with descriptions of met criteria.

        Returns True only if ALL criteria are met.
        """
        all_met = True
        for criterion in criteria:
            if self._evaluate_single_criterion(criterion, call_history):
                met_out.append(f"{criterion.type}:{criterion.server}/{criterion.action}")
            else:
                all_met = False
        return all_met

    def _handle_phase_success(
        self,
        phase: PhaseDefinition,
        met_criteria: list[str],
    ) -> StepResult:
        """Handle successful completion of a phase."""
        next_phase_name = phase.transitions.get("success")

        if next_phase_name is None:
            # No next phase -- scenario complete
            self._completed = True
            return StepResult(
                phase_changed=False,
                scenario_complete=True,
                criteria_met=met_criteria,
                feedback=f"Scenario complete! Final phase '{phase.name}' succeeded.",
            )

        # Find the next phase by name
        next_idx = self._find_phase_index(next_phase_name)
        if next_idx is None:
            self._failed = True
            return StepResult(
                scenario_failed=True,
                feedback=f"Transition target '{next_phase_name}' not found.",
            )

        self._phase_idx = next_idx
        self._phase_step_count = 0
        self._phase_call_history = []
        self._apply_entry_state(self.current_phase)

        return StepResult(
            phase_changed=True,
            new_phase=next_phase_name,
            criteria_met=met_criteria,
            feedback=f"Transitioned to phase '{next_phase_name}'.",
        )

    def _find_phase_index(self, name: str) -> int | None:
        """Find a phase index by name. Returns None if not found."""
        for idx, phase in enumerate(self._scenario.phases):
            if phase.name == name:
                return idx
        return None

    def _apply_entry_state(self, phase: PhaseDefinition) -> None:
        """Apply a phase's entry_state to the virtual filesystem.

        The entry_state dict can contain:
            - files: dict[str, str] mapping paths to content
            - context: dict of arbitrary context variables (stored as metadata)
        """
        files = phase.entry_state.get("files", {})
        for path, content in files.items():
            self._surface.filesystem.write(path, content)

    def _build_phase_context(self) -> dict[str, Any]:
        """Build an observation context dict for the current phase."""
        phase = self.current_phase
        return {
            "phase": phase.name,
            "description": phase.description,
            "available_tools": phase.available_tools,
            "max_steps": phase.max_steps,
            "files": list(self._surface.filesystem.snapshot().keys()),
        }
