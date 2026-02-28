"""Tests for the scenario system (schema, loader, state machine).

Covers:
    - ScenarioDefinition: Pydantic validation, required fields, defaults
    - PhaseDefinition: available_tools, entry_state, criteria, transitions
    - CriterionDef: all 5 types (tool_called, tool_not_called, file_exists,
      file_contains, param_matches)
    - load_scenario: load from YAML file, invalid YAML, missing file
    - load_all_scenarios: directory scanning, sorting by name
    - validate_scenario: tool registration checking, forbidden action validation
    - ScenarioStateMachine: multi-phase execution, phase transitions,
      success/failure criteria checking, forbidden action detection,
      max_steps exceeded, entry_state filesystem initialization,
      phase_changed tracking
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mcp_gym.mock_surface import MockMCPSurface
from mcp_gym.scenarios.loader import load_all_scenarios, load_scenario, validate_scenario
from mcp_gym.scenarios.schema import CriterionDef, PhaseDefinition, ScenarioDefinition
from mcp_gym.scenarios.state_machine import ScenarioStateMachine, StepResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_scenario_dict() -> dict:
    """Return a minimal valid scenario dict for testing."""
    return {
        "name": "test-scenario",
        "domain": "security",
        "description": "A test scenario.",
        "phases": [
            {
                "name": "phase-1",
                "description": "First phase.",
                "available_tools": ["corso/guard"],
                "success_criteria": [
                    {"type": "tool_called", "server": "corso", "action": "guard"},
                ],
                "max_steps": 5,
            },
        ],
    }


def _two_phase_scenario_dict() -> dict:
    """Return a scenario with two phases and a success transition."""
    return {
        "name": "two-phase-test",
        "domain": "investigation",
        "description": "Two-phase scenario with transitions.",
        "token_budget": 4096,
        "phases": [
            {
                "name": "recon",
                "description": "Reconnaissance phase.",
                "available_tools": ["corso/guard"],
                "success_criteria": [
                    {"type": "tool_called", "server": "corso", "action": "guard"},
                ],
                "max_steps": 3,
                "transitions": {"success": "analyze"},
            },
            {
                "name": "analyze",
                "description": "Analysis phase.",
                "available_tools": ["quantum/scan"],
                "entry_state": {
                    "files": {
                        "report.md": "# Analysis Report\n",
                    },
                },
                "success_criteria": [
                    {"type": "tool_called", "server": "quantum", "action": "scan"},
                    {"type": "file_contains", "path": "report.md", "pattern": "Analysis"},
                ],
                "max_steps": 5,
            },
        ],
        "forbidden_actions": ["quantum/destroy"],
    }


def _write_scenario_yaml(path: Path, data: dict) -> Path:
    """Write a scenario dict to a YAML file and return the path."""
    with open(path, "w") as fh:
        yaml.safe_dump(data, fh)
    return path


def _build_surface(*actions: str) -> MockMCPSurface:
    """Build a MockMCPSurface with the given actions registered."""
    surface = MockMCPSurface()
    for action in actions:
        surface.register(action, lambda p: {"ok": True})
    return surface


# ---------------------------------------------------------------------------
# ScenarioDefinition
# ---------------------------------------------------------------------------


class TestScenarioDefinition:
    """Tests for ScenarioDefinition Pydantic model."""

    def test_minimal_valid_scenario(self) -> None:
        """A minimal scenario with required fields is valid."""
        data = _minimal_scenario_dict()
        scenario = ScenarioDefinition.model_validate(data)
        assert scenario.name == "test-scenario"
        assert scenario.domain == "security"
        assert len(scenario.phases) == 1

    def test_default_values(self) -> None:
        """Optional fields have sensible defaults."""
        data = _minimal_scenario_dict()
        scenario = ScenarioDefinition.model_validate(data)
        assert scenario.token_budget == 2048
        assert scenario.expected_sequence == []
        assert scenario.forbidden_actions == []
        assert scenario.escalation_points == []
        assert scenario.metadata == {}

    def test_custom_token_budget(self) -> None:
        """Custom token_budget is preserved."""
        data = _minimal_scenario_dict()
        data["token_budget"] = 8192
        scenario = ScenarioDefinition.model_validate(data)
        assert scenario.token_budget == 8192

    def test_missing_name_raises(self) -> None:
        """Missing 'name' field raises ValidationError."""
        data = _minimal_scenario_dict()
        del data["name"]
        with pytest.raises(ValidationError):
            ScenarioDefinition.model_validate(data)

    def test_missing_domain_raises(self) -> None:
        """Missing 'domain' field raises ValidationError."""
        data = _minimal_scenario_dict()
        del data["domain"]
        with pytest.raises(ValidationError):
            ScenarioDefinition.model_validate(data)

    def test_missing_description_raises(self) -> None:
        """Missing 'description' field raises ValidationError."""
        data = _minimal_scenario_dict()
        del data["description"]
        with pytest.raises(ValidationError):
            ScenarioDefinition.model_validate(data)

    def test_missing_phases_raises(self) -> None:
        """Missing 'phases' field raises ValidationError."""
        data = _minimal_scenario_dict()
        del data["phases"]
        with pytest.raises(ValidationError):
            ScenarioDefinition.model_validate(data)

    def test_empty_phases_accepted(self) -> None:
        """An empty phases list is accepted by the schema (not by the state machine)."""
        data = _minimal_scenario_dict()
        data["phases"] = []
        scenario = ScenarioDefinition.model_validate(data)
        assert scenario.phases == []

    def test_forbidden_actions_populated(self) -> None:
        """forbidden_actions list is preserved."""
        data = _minimal_scenario_dict()
        data["forbidden_actions"] = ["quantum/destroy", "eva/delete"]
        scenario = ScenarioDefinition.model_validate(data)
        assert "quantum/destroy" in scenario.forbidden_actions

    def test_expected_sequence_populated(self) -> None:
        """expected_sequence is preserved."""
        data = _minimal_scenario_dict()
        data["expected_sequence"] = [{"server": "corso", "action": "guard"}]
        scenario = ScenarioDefinition.model_validate(data)
        assert len(scenario.expected_sequence) == 1

    def test_metadata_dict(self) -> None:
        """Arbitrary metadata dict is preserved."""
        data = _minimal_scenario_dict()
        data["metadata"] = {"author": "test", "version": "1.0"}
        scenario = ScenarioDefinition.model_validate(data)
        assert scenario.metadata["author"] == "test"


# ---------------------------------------------------------------------------
# PhaseDefinition
# ---------------------------------------------------------------------------


class TestPhaseDefinition:
    """Tests for PhaseDefinition Pydantic model."""

    def test_minimal_phase(self) -> None:
        """A phase with just name and description is valid."""
        phase = PhaseDefinition(name="init", description="Initialize.")
        assert phase.name == "init"
        assert phase.available_tools == []
        assert phase.entry_state == {}
        assert phase.success_criteria == []
        assert phase.failure_criteria == []
        assert phase.max_steps == 10
        assert phase.transitions == {}

    def test_available_tools(self) -> None:
        """available_tools list is preserved."""
        phase = PhaseDefinition(
            name="scan",
            description="Scan phase.",
            available_tools=["corso/guard", "quantum/scan"],
        )
        assert len(phase.available_tools) == 2
        assert "corso/guard" in phase.available_tools

    def test_entry_state_with_files(self) -> None:
        """entry_state can contain files dict."""
        phase = PhaseDefinition(
            name="analyze",
            description="Analysis.",
            entry_state={"files": {"report.md": "# Report"}},
        )
        assert phase.entry_state["files"]["report.md"] == "# Report"

    def test_success_criteria(self) -> None:
        """success_criteria list of CriterionDef is preserved."""
        phase = PhaseDefinition(
            name="check",
            description="Check phase.",
            success_criteria=[
                CriterionDef(type="tool_called", server="corso", action="guard"),
            ],
        )
        assert len(phase.success_criteria) == 1
        assert phase.success_criteria[0].type == "tool_called"

    def test_failure_criteria(self) -> None:
        """failure_criteria list is preserved."""
        phase = PhaseDefinition(
            name="check",
            description="Check phase.",
            failure_criteria=[
                CriterionDef(type="tool_called", server="quantum", action="destroy"),
            ],
        )
        assert len(phase.failure_criteria) == 1

    def test_transitions_dict(self) -> None:
        """transitions dict maps outcomes to phase names."""
        phase = PhaseDefinition(
            name="phase-1",
            description="First.",
            transitions={"success": "phase-2", "failure": None},
        )
        assert phase.transitions["success"] == "phase-2"
        assert phase.transitions["failure"] is None

    def test_custom_max_steps(self) -> None:
        """Custom max_steps overrides the default of 10."""
        phase = PhaseDefinition(name="p", description="d", max_steps=3)
        assert phase.max_steps == 3


# ---------------------------------------------------------------------------
# CriterionDef
# ---------------------------------------------------------------------------


class TestCriterionDef:
    """Tests for all 5 criterion types."""

    def test_tool_called(self) -> None:
        """tool_called criterion has server and action."""
        c = CriterionDef(type="tool_called", server="corso", action="guard")
        assert c.type == "tool_called"
        assert c.server == "corso"
        assert c.action == "guard"

    def test_tool_not_called(self) -> None:
        """tool_not_called criterion has server and action."""
        c = CriterionDef(type="tool_not_called", server="quantum", action="destroy")
        assert c.type == "tool_not_called"

    def test_file_exists(self) -> None:
        """file_exists criterion has a path."""
        c = CriterionDef(type="file_exists", path="report.md")
        assert c.type == "file_exists"
        assert c.path == "report.md"

    def test_file_contains(self) -> None:
        """file_contains criterion has path and pattern."""
        c = CriterionDef(type="file_contains", path="report.md", pattern="CLEAN")
        assert c.type == "file_contains"
        assert c.pattern == "CLEAN"

    def test_param_matches(self) -> None:
        """param_matches criterion has server, action, and params dict."""
        c = CriterionDef(
            type="param_matches",
            server="corso",
            action="guard",
            params={"path": "/src"},
        )
        assert c.type == "param_matches"
        assert c.params == {"path": "/src"}

    def test_optional_fields_default_none(self) -> None:
        """Optional fields default to None when not provided."""
        c = CriterionDef(type="tool_called")
        assert c.server is None
        assert c.action is None
        assert c.path is None
        assert c.pattern is None
        assert c.params is None


# ---------------------------------------------------------------------------
# load_scenario
# ---------------------------------------------------------------------------


class TestLoadScenario:
    """Tests for load_scenario YAML loading."""

    def test_load_valid_yaml(self, tmp_path: Path) -> None:
        """Load a valid scenario from a YAML file."""
        data = _minimal_scenario_dict()
        yaml_path = _write_scenario_yaml(tmp_path / "scenario.yaml", data)
        scenario = load_scenario(yaml_path)
        assert scenario.name == "test-scenario"
        assert scenario.domain == "security"
        assert len(scenario.phases) == 1

    def test_load_with_all_fields(self, tmp_path: Path) -> None:
        """Load a scenario with all optional fields populated."""
        data = _two_phase_scenario_dict()
        data["expected_sequence"] = [{"server": "corso", "action": "guard"}]
        data["escalation_points"] = ["corso/speak"]
        data["metadata"] = {"difficulty": "medium"}
        yaml_path = _write_scenario_yaml(tmp_path / "full.yaml", data)
        scenario = load_scenario(yaml_path)
        assert len(scenario.phases) == 2
        assert len(scenario.expected_sequence) == 1
        assert scenario.escalation_points == ["corso/speak"]
        assert scenario.metadata["difficulty"] == "medium"

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        """Loading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_scenario(tmp_path / "nonexistent.yaml")

    def test_load_invalid_yaml_raises(self, tmp_path: Path) -> None:
        """Loading a file with invalid YAML raises yaml.YAMLError."""
        bad_path = tmp_path / "bad.yaml"
        bad_path.write_text("name: test\n  bad_indent: [unclosed")
        with pytest.raises(yaml.YAMLError):
            load_scenario(bad_path)

    def test_load_valid_yaml_but_bad_schema_raises(self, tmp_path: Path) -> None:
        """Valid YAML that does not match the schema raises ValidationError."""
        bad_data = {"name": "test"}  # Missing required fields
        yaml_path = _write_scenario_yaml(tmp_path / "incomplete.yaml", bad_data)
        with pytest.raises(ValidationError):
            load_scenario(yaml_path)

    def test_load_yml_extension(self, tmp_path: Path) -> None:
        """load_scenario works with .yml extension (just a Path, no glob)."""
        data = _minimal_scenario_dict()
        yaml_path = _write_scenario_yaml(tmp_path / "scenario.yml", data)
        scenario = load_scenario(yaml_path)
        assert scenario.name == "test-scenario"


# ---------------------------------------------------------------------------
# load_all_scenarios
# ---------------------------------------------------------------------------


class TestLoadAllScenarios:
    """Tests for load_all_scenarios directory loading."""

    def test_load_from_directory(self, tmp_path: Path) -> None:
        """Load all YAML files from a directory."""
        for name in ("alpha.yaml", "beta.yaml", "gamma.yml"):
            data = _minimal_scenario_dict()
            data["name"] = name.split(".")[0]
            _write_scenario_yaml(tmp_path / name, data)

        scenarios = load_all_scenarios(tmp_path)
        assert len(scenarios) == 3

    def test_sorted_by_name(self, tmp_path: Path) -> None:
        """Scenarios are sorted by name (alphabetically)."""
        for name in ("zebra", "apple", "middle"):
            data = _minimal_scenario_dict()
            data["name"] = name
            _write_scenario_yaml(tmp_path / f"{name}.yaml", data)

        scenarios = load_all_scenarios(tmp_path)
        names = [s.name for s in scenarios]
        assert names == ["apple", "middle", "zebra"]

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Empty directory returns empty list."""
        scenarios = load_all_scenarios(tmp_path)
        assert scenarios == []

    def test_nonexistent_directory_raises(self, tmp_path: Path) -> None:
        """Non-existent directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            load_all_scenarios(tmp_path / "nonexistent")

    def test_ignores_non_yaml_files(self, tmp_path: Path) -> None:
        """Non-YAML files in the directory are ignored."""
        data = _minimal_scenario_dict()
        _write_scenario_yaml(tmp_path / "valid.yaml", data)
        (tmp_path / "readme.txt").write_text("Not a scenario")
        (tmp_path / "notes.json").write_text('{"key": "value"}')

        scenarios = load_all_scenarios(tmp_path)
        assert len(scenarios) == 1

    def test_both_yaml_and_yml_loaded(self, tmp_path: Path) -> None:
        """Both .yaml and .yml extensions are loaded."""
        data_a = _minimal_scenario_dict()
        data_a["name"] = "alpha"
        _write_scenario_yaml(tmp_path / "alpha.yaml", data_a)

        data_b = _minimal_scenario_dict()
        data_b["name"] = "beta"
        _write_scenario_yaml(tmp_path / "beta.yml", data_b)

        scenarios = load_all_scenarios(tmp_path)
        assert len(scenarios) == 2


# ---------------------------------------------------------------------------
# validate_scenario
# ---------------------------------------------------------------------------


class TestValidateScenario:
    """Tests for validate_scenario against MockMCPSurface."""

    def test_all_tools_registered_no_errors(self) -> None:
        """All phase tools and forbidden actions registered: no errors."""
        data = _minimal_scenario_dict()
        scenario = ScenarioDefinition.model_validate(data)
        surface = _build_surface("corso/guard")
        errors = validate_scenario(scenario, surface)
        assert errors == []

    def test_missing_phase_tool_reported(self) -> None:
        """An unregistered phase tool produces an error."""
        data = _minimal_scenario_dict()
        scenario = ScenarioDefinition.model_validate(data)
        surface = _build_surface()  # Nothing registered
        errors = validate_scenario(scenario, surface)
        assert len(errors) == 1
        assert "corso/guard" in errors[0]
        assert "not registered" in errors[0]

    def test_missing_forbidden_action_reported(self) -> None:
        """An unregistered forbidden action produces a warning error."""
        data = _minimal_scenario_dict()
        data["forbidden_actions"] = ["quantum/destroy"]
        scenario = ScenarioDefinition.model_validate(data)
        surface = _build_surface("corso/guard")  # Only guard registered
        errors = validate_scenario(scenario, surface)
        assert any("quantum/destroy" in e for e in errors)
        assert any("cannot be enforced" in e for e in errors)

    def test_multiple_phases_validated(self) -> None:
        """All phases are checked, not just the first one."""
        data = _two_phase_scenario_dict()
        scenario = ScenarioDefinition.model_validate(data)
        # Only register tools for phase 1, not phase 2
        surface = _build_surface("corso/guard")
        errors = validate_scenario(scenario, surface)
        # Should report quantum/scan and quantum/destroy as unregistered
        assert any("quantum/scan" in e for e in errors)
        assert any("quantum/destroy" in e for e in errors)

    def test_all_registered_including_forbidden(self) -> None:
        """When everything is registered (including forbidden), no errors."""
        data = _two_phase_scenario_dict()
        scenario = ScenarioDefinition.model_validate(data)
        surface = _build_surface("corso/guard", "quantum/scan", "quantum/destroy")
        errors = validate_scenario(scenario, surface)
        assert errors == []


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------


class TestStepResult:
    """Tests for StepResult data model."""

    def test_defaults(self) -> None:
        """StepResult has sensible defaults."""
        result = StepResult()
        assert result.phase_changed is False
        assert result.new_phase is None
        assert result.scenario_complete is False
        assert result.scenario_failed is False
        assert result.criteria_met == []
        assert result.feedback == ""

    def test_custom_values(self) -> None:
        """StepResult accepts custom values."""
        result = StepResult(
            phase_changed=True,
            new_phase="phase-2",
            criteria_met=["tool_called:corso/guard"],
            feedback="Transitioning.",
        )
        assert result.phase_changed is True
        assert result.new_phase == "phase-2"


# ---------------------------------------------------------------------------
# ScenarioStateMachine
# ---------------------------------------------------------------------------


class TestScenarioStateMachine:
    """Tests for the ScenarioStateMachine execution engine."""

    def _make_sm(
        self,
        scenario_data: dict | None = None,
        *extra_actions: str,
    ) -> tuple[ScenarioStateMachine, MockMCPSurface]:
        """Create a state machine and surface from scenario data."""
        data = scenario_data or _minimal_scenario_dict()
        scenario = ScenarioDefinition.model_validate(data)
        surface = _build_surface("corso/guard", "quantum/scan", "eva/speak", *extra_actions)
        sm = ScenarioStateMachine(scenario, surface)
        return sm, surface

    def test_reset_returns_phase_context(self) -> None:
        """reset() returns observation context for the first phase."""
        sm, _ = self._make_sm()
        ctx = sm.reset()
        assert ctx["phase"] == "phase-1"
        assert "description" in ctx
        assert "available_tools" in ctx
        assert "max_steps" in ctx

    def test_current_phase_after_reset(self) -> None:
        """After reset, current_phase is the first phase."""
        sm, _ = self._make_sm()
        sm.reset()
        assert sm.current_phase.name == "phase-1"
        assert sm.phase_index == 0

    def test_is_complete_false_after_reset(self) -> None:
        """After reset, scenario is not complete."""
        sm, _ = self._make_sm()
        sm.reset()
        assert sm.is_complete is False

    def test_single_phase_success(self) -> None:
        """Calling the expected tool in a single-phase scenario completes it."""
        sm, _ = self._make_sm()
        sm.reset()
        result = sm.process_step(
            server="corso", action="guard", params={}, result={"clean": True}
        )
        assert result.scenario_complete is True
        assert sm.is_complete is True
        assert "Scenario complete" in result.feedback

    def test_forbidden_action_fails_scenario(self) -> None:
        """Calling a forbidden action immediately fails the scenario."""
        data = _minimal_scenario_dict()
        data["forbidden_actions"] = ["quantum/destroy"]
        sm, _ = self._make_sm(data, "quantum/destroy")
        sm.reset()
        result = sm.process_step(
            server="quantum", action="destroy", params={}, result={}
        )
        assert result.scenario_failed is True
        assert "Forbidden action" in result.feedback
        assert sm.is_complete is True

    def test_max_steps_exceeded(self) -> None:
        """Exceeding max_steps in a phase fails the scenario."""
        data = _minimal_scenario_dict()
        # Increase max_steps to 2, but never satisfy success criteria
        data["phases"][0]["max_steps"] = 2
        data["phases"][0]["success_criteria"] = [
            {"type": "tool_called", "server": "eva", "action": "speak"},
        ]
        sm, _ = self._make_sm(data)
        sm.reset()

        # Step 1: call guard (not the expected tool)
        result1 = sm.process_step(
            server="corso", action="guard", params={}, result={}
        )
        assert result1.scenario_failed is False
        assert result1.scenario_complete is False

        # Step 2: call guard again (max_steps=2 reached)
        result2 = sm.process_step(
            server="corso", action="guard", params={}, result={}
        )
        assert result2.scenario_failed is True
        assert "Max steps" in result2.feedback

    def test_already_finished_returns_status(self) -> None:
        """Processing a step after completion returns the finished status."""
        sm, _ = self._make_sm()
        sm.reset()
        sm.process_step(server="corso", action="guard", params={}, result={})
        # Now scenario is complete; another step returns the same status
        result = sm.process_step(
            server="corso", action="guard", params={}, result={}
        )
        assert result.scenario_complete is True
        assert "already finished" in result.feedback

    def test_two_phase_transition(self) -> None:
        """Success in phase 1 transitions to phase 2."""
        data = _two_phase_scenario_dict()
        sm, _surface = self._make_sm(data)
        sm.reset()

        # Phase 1: call guard -> success, transition to "analyze"
        result = sm.process_step(
            server="corso", action="guard", params={}, result={}
        )
        assert result.phase_changed is True
        assert result.new_phase == "analyze"
        assert sm.current_phase.name == "analyze"
        assert sm.phase_index == 1

    def test_phase_transition_applies_entry_state(self) -> None:
        """Transitioning to a new phase applies that phase's entry_state files."""
        data = _two_phase_scenario_dict()
        sm, surface = self._make_sm(data)
        sm.reset()

        # Phase 1 success
        sm.process_step(server="corso", action="guard", params={}, result={})

        # Phase 2 entry_state should have written "report.md"
        assert surface.filesystem.exists("report.md")
        content = surface.filesystem.read("report.md")
        assert content is not None
        assert "Analysis Report" in content

    def test_phase_2_completion(self) -> None:
        """Completing all criteria in phase 2 finishes the scenario."""
        data = _two_phase_scenario_dict()
        sm, _surface = self._make_sm(data)
        sm.reset()

        # Phase 1 success
        sm.process_step(server="corso", action="guard", params={}, result={})

        # Phase 2: call quantum/scan (success_criteria: tool_called + file_contains)
        # file_contains is satisfied because entry_state wrote report.md with "Analysis"
        result = sm.process_step(
            server="quantum", action="scan", params={}, result={}
        )
        assert result.scenario_complete is True

    def test_failure_criteria_checked_before_success(self) -> None:
        """Failure criteria are checked before success criteria."""
        data = _minimal_scenario_dict()
        # Both failure and success criteria match on the same call
        data["phases"][0]["failure_criteria"] = [
            {"type": "tool_called", "server": "corso", "action": "guard"},
        ]
        sm, _ = self._make_sm(data)
        sm.reset()
        result = sm.process_step(
            server="corso", action="guard", params={}, result={}
        )
        # Failure criteria take precedence
        assert result.scenario_failed is True
        assert "Failure criteria met" in result.feedback

    def test_in_progress_step_feedback(self) -> None:
        """A step that does not complete or fail gives progress feedback."""
        data = _minimal_scenario_dict()
        data["phases"][0]["max_steps"] = 10
        data["phases"][0]["success_criteria"] = [
            {"type": "tool_called", "server": "eva", "action": "speak"},
        ]
        sm, _ = self._make_sm(data)
        sm.reset()
        result = sm.process_step(
            server="corso", action="guard", params={}, result={}
        )
        assert result.scenario_complete is False
        assert result.scenario_failed is False
        assert "step 1/10" in result.feedback

    def test_check_criteria_all_must_pass(self) -> None:
        """check_criteria returns True only if ALL criteria are met (AND logic)."""
        data = _minimal_scenario_dict()
        sm, _ = self._make_sm(data)
        sm.reset()

        criteria = [
            CriterionDef(type="tool_called", server="corso", action="guard"),
            CriterionDef(type="tool_called", server="eva", action="speak"),
        ]
        history = [
            {"server": "corso", "action": "guard"},
        ]
        # Only one of two criteria met
        assert sm.check_criteria(criteria, history) is False

        history.append({"server": "eva", "action": "speak"})
        assert sm.check_criteria(criteria, history) is True

    def test_criterion_tool_not_called(self) -> None:
        """tool_not_called passes when the tool was NOT in history."""
        data = _minimal_scenario_dict()
        sm, _ = self._make_sm(data)
        sm.reset()

        criteria = [
            CriterionDef(type="tool_not_called", server="quantum", action="destroy"),
        ]
        history = [{"server": "corso", "action": "guard"}]
        assert sm.check_criteria(criteria, history) is True

        # Now add the forbidden call
        history.append({"server": "quantum", "action": "destroy"})
        assert sm.check_criteria(criteria, history) is False

    def test_criterion_file_exists(self) -> None:
        """file_exists checks the virtual filesystem."""
        data = _minimal_scenario_dict()
        sm, surface = self._make_sm(data)
        sm.reset()

        criteria = [CriterionDef(type="file_exists", path="output.txt")]

        # File does not exist yet
        assert sm.check_criteria(criteria, []) is False

        # Create the file
        surface.filesystem.write("output.txt", "content")
        assert sm.check_criteria(criteria, []) is True

    def test_criterion_file_contains_regex(self) -> None:
        """file_contains uses regex matching on file content."""
        data = _minimal_scenario_dict()
        sm, surface = self._make_sm(data)
        sm.reset()
        surface.filesystem.write("log.txt", "ERROR: connection refused at 10:30")

        # Regex match
        criteria_match = [
            CriterionDef(type="file_contains", path="log.txt", pattern=r"ERROR:.*refused"),
        ]
        assert sm.check_criteria(criteria_match, []) is True

        # Regex no match
        criteria_no_match = [
            CriterionDef(type="file_contains", path="log.txt", pattern=r"SUCCESS"),
        ]
        assert sm.check_criteria(criteria_no_match, []) is False

    def test_criterion_file_contains_missing_file(self) -> None:
        """file_contains returns False when the file does not exist."""
        data = _minimal_scenario_dict()
        sm, _ = self._make_sm(data)
        sm.reset()

        criteria = [
            CriterionDef(type="file_contains", path="missing.txt", pattern="anything"),
        ]
        assert sm.check_criteria(criteria, []) is False

    def test_criterion_file_contains_no_path(self) -> None:
        """file_contains returns False when path is None."""
        data = _minimal_scenario_dict()
        sm, _ = self._make_sm(data)
        sm.reset()

        criteria = [
            CriterionDef(type="file_contains", path=None, pattern="anything"),
        ]
        assert sm.check_criteria(criteria, []) is False

    def test_criterion_file_exists_no_path(self) -> None:
        """file_exists returns False when path is None."""
        data = _minimal_scenario_dict()
        sm, _ = self._make_sm(data)
        sm.reset()

        criteria = [CriterionDef(type="file_exists", path=None)]
        assert sm.check_criteria(criteria, []) is False

    def test_criterion_param_matches(self) -> None:
        """param_matches checks server, action, and specific param key-values."""
        data = _minimal_scenario_dict()
        sm, _ = self._make_sm(data)
        sm.reset()

        criteria = [
            CriterionDef(
                type="param_matches",
                server="corso",
                action="guard",
                params={"path": "/src", "severity": "high"},
            ),
        ]

        # Partial match (missing severity) -> False
        history_partial = [
            {"server": "corso", "action": "guard", "params": {"path": "/src"}},
        ]
        assert sm.check_criteria(criteria, history_partial) is False

        # Full match
        history_full = [
            {
                "server": "corso",
                "action": "guard",
                "params": {"path": "/src", "severity": "high", "extra": "ignored"},
            },
        ]
        assert sm.check_criteria(criteria, history_full) is True

    def test_criterion_param_matches_no_params(self) -> None:
        """param_matches returns False when criterion params is None."""
        data = _minimal_scenario_dict()
        sm, _ = self._make_sm(data)
        sm.reset()

        criteria = [
            CriterionDef(type="param_matches", server="corso", action="guard", params=None),
        ]
        history = [{"server": "corso", "action": "guard", "params": {"path": "/src"}}]
        assert sm.check_criteria(criteria, history) is False

    def test_unknown_criterion_type_returns_false(self) -> None:
        """An unrecognized criterion type evaluates to False."""
        data = _minimal_scenario_dict()
        sm, _ = self._make_sm(data)
        sm.reset()

        criteria = [CriterionDef(type="unknown_type")]
        assert sm.check_criteria(criteria, []) is False

    def test_reset_clears_state(self) -> None:
        """Resetting the state machine returns to phase 0 and clears history."""
        data = _two_phase_scenario_dict()
        sm, _surface = self._make_sm(data)
        sm.reset()

        # Progress to phase 2
        sm.process_step(server="corso", action="guard", params={}, result={})
        assert sm.phase_index == 1

        # Reset
        ctx = sm.reset()
        assert sm.phase_index == 0
        assert sm.is_complete is False
        assert ctx["phase"] == "recon"

    def test_reset_applies_first_phase_entry_state(self) -> None:
        """reset() applies the first phase's entry_state to the filesystem."""
        data = _minimal_scenario_dict()
        data["phases"][0]["entry_state"] = {
            "files": {"config.yaml": "key: value"},
        }
        sm, surface = self._make_sm(data)
        sm.reset()
        assert surface.filesystem.exists("config.yaml")
        assert surface.filesystem.read("config.yaml") == "key: value"

    def test_reset_clears_filesystem(self) -> None:
        """reset() clears the filesystem from prior episodes."""
        data = _minimal_scenario_dict()
        sm, surface = self._make_sm(data)
        surface.filesystem.write("leftover.txt", "old data")
        sm.reset()
        assert not surface.filesystem.exists("leftover.txt")

    def test_transition_target_not_found_fails(self) -> None:
        """If a transition target phase name does not exist, scenario fails."""
        data = _minimal_scenario_dict()
        data["phases"][0]["success_criteria"] = [
            {"type": "tool_called", "server": "corso", "action": "guard"},
        ]
        data["phases"][0]["transitions"] = {"success": "nonexistent_phase"}
        sm, _ = self._make_sm(data)
        sm.reset()
        result = sm.process_step(
            server="corso", action="guard", params={}, result={}
        )
        assert result.scenario_failed is True
        assert "not found" in result.feedback

    def test_build_phase_context_includes_files(self) -> None:
        """Phase context includes list of files in the filesystem."""
        data = _minimal_scenario_dict()
        data["phases"][0]["entry_state"] = {
            "files": {"a.txt": "alpha", "b.txt": "beta"},
        }
        sm, _ = self._make_sm(data)
        ctx = sm.reset()
        assert "a.txt" in ctx["files"]
        assert "b.txt" in ctx["files"]

    def test_criteria_met_list_populated(self) -> None:
        """StepResult criteria_met list contains descriptions of met criteria."""
        data = _minimal_scenario_dict()
        data["phases"][0]["success_criteria"] = [
            {"type": "tool_called", "server": "corso", "action": "guard"},
        ]
        sm, _ = self._make_sm(data)
        sm.reset()
        result = sm.process_step(
            server="corso", action="guard", params={}, result={}
        )
        assert len(result.criteria_met) >= 1
        assert "tool_called:corso/guard" in result.criteria_met[0]

    def test_phase_step_count_resets_on_transition(self) -> None:
        """Step count resets to 0 when transitioning to a new phase."""
        data = _two_phase_scenario_dict()
        sm, _ = self._make_sm(data)
        sm.reset()

        # Phase 1 step
        sm.process_step(server="corso", action="guard", params={}, result={})

        # Now in phase 2: should be able to take up to max_steps=5
        # (not still counting from phase 1)
        for _ in range(4):
            result = sm.process_step(
                server="corso", action="guard", params={}, result={}
            )
            assert result.scenario_failed is False

        # Step 5 should fail (max_steps=5 for phase "analyze")
        result = sm.process_step(
            server="corso", action="guard", params={}, result={}
        )
        assert result.scenario_failed is True
        assert "Max steps" in result.feedback
