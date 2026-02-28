"""Action and observation space definitions for MCP Agent Gym.

Both action and observation spaces are text-based (JSON strings),
which aligns with how LLM agents naturally interact: producing
and consuming structured text.

Action space:
    JSON string with keys: server, action, params
    Example: '{"server": "corso", "action": "guard", "params": {"path": "/src"}}'

Observation space:
    JSON string with keys:
    - last_result: the MCP response from previous action (or null)
    - step_count: current step number
    - token_budget_remaining: tokens left in the episode
    - call_history_length: number of actions taken so far
"""

from __future__ import annotations

from gymnasium import spaces

# Action space: agent produces a JSON string describing the MCP call
ACTION_SPACE = spaces.Text(
    min_length=2,
    max_length=2048,
)

# Observation space: environment returns a JSON string with state
OBSERVATION_SPACE = spaces.Text(
    min_length=0,
    max_length=8192,
)
