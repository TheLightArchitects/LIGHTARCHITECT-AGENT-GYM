"""MCP server stubs for deterministic evaluation.

Each module provides a register_all(surface) function that wires
mock handlers into the MockMCPSurface registry.
"""

from mcp_gym.stubs.corso import register_all as register_corso
from mcp_gym.stubs.eva import register_all as register_eva
from mcp_gym.stubs.quantum import register_all as register_quantum
from mcp_gym.stubs.soul import register_all as register_soul

__all__ = [
    "register_corso",
    "register_eva",
    "register_quantum",
    "register_soul",
]
