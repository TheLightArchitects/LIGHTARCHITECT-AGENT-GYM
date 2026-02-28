"""Core type definitions for MCP Agent Gym."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MCPServer(str, Enum):
    """Available MCP servers in the evaluation environment."""

    CORSO = "corso"
    EVA = "eva"
    SOUL = "soul"
    QUANTUM = "quantum"


class MCPAction(BaseModel):
    """An action targeting an MCP server."""

    server: MCPServer
    action: str
    params: dict[str, Any] = Field(default_factory=dict)


class MCPResult(BaseModel):
    """Result from an MCP tool call."""

    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class ToolNotAvailableError(Exception):
    """Raised when an unregistered MCP action is called."""

    def __init__(self, server: str, action: str) -> None:
        self.server = server
        self.action = action
        super().__init__(f"Tool not available: {server}/{action}")


class EpisodeConfig(BaseModel):
    """Configuration for a single evaluation episode."""

    max_steps: int = 20
    token_budget: int = 4096
    seed: int = 42
