# MCP Agent Evaluation Gym

Gymnasium-compatible environment for evaluating AI agent MCP tool-use behavior.

## Overview

This package provides a standardized evaluation framework for AI agents that interact with MCP (Model Context Protocol) servers. It wraps real MCP tool surfaces with deterministic mock implementations, enabling reproducible benchmarking of agent decision-making, tool selection, and multi-step reasoning.

## Architecture

```
MCPAgentEnv (gymnasium.Env)
  |
  +-- MockMCPSurface (registry pattern)
  |     |
  |     +-- VirtualFilesystem (in-memory state)
  |     +-- Call log (ordered action history)
  |
  +-- Stubs (deterministic mock handlers)
        +-- CORSO (24 actions)
        +-- EVA (9 tools)
        +-- SOUL (11 sub-tools)
        +-- QUANTUM (13 actions)
```

## Install

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from mcp_gym import make_env

env = make_env()
obs, info = env.reset(seed=42)

# Agent produces a JSON action string
import json
action = json.dumps({"server": "corso", "action": "guard", "params": {"path": "/src"}})
obs, reward, terminated, truncated, info = env.step(action)
```

## Tests

```bash
pytest tests/ -v
```

## Servers

| Server | Actions | Domain |
|--------|---------|--------|
| CORSO | 24 | Security, operations, code generation |
| EVA | 9 | Consciousness, memory, education |
| SOUL | 11 | Knowledge graph, helix, voice |
| QUANTUM | 13 | Investigation, forensics, research |
