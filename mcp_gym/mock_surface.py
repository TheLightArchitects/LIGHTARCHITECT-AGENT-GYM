"""MockMCPSurface and VirtualFilesystem for deterministic MCP evaluation."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

from mcp_gym.types import ToolNotAvailableError


class VirtualFilesystem:
    """In-memory file tree for scenario state.

    Provides a simple path-based filesystem where paths are forward-slash
    separated strings. No actual disk I/O occurs.
    """

    def __init__(self) -> None:
        self._tree: dict[str, str] = {}

    def write(self, path: str, content: str) -> None:
        """Write content to a path, creating or overwriting."""
        path = self._normalize(path)
        self._tree[path] = content

    def read(self, path: str) -> str | None:
        """Read content from a path. Returns None if not found."""
        path = self._normalize(path)
        return self._tree.get(path)

    def list_dir(self, path: str) -> list[str]:
        """List immediate children of a directory path.

        Returns file/directory names (not full paths) that are direct
        children of the given path.
        """
        path = self._normalize(path)
        if path and not path.endswith("/"):
            path += "/"

        children: set[str] = set()
        for key in self._tree:
            if key.startswith(path) and key != path:
                remainder = key[len(path) :]
                # Take only the first segment (immediate child)
                first_segment = remainder.split("/")[0]
                if first_segment:
                    children.add(first_segment)
        return sorted(children)

    def exists(self, path: str) -> bool:
        """Check if a path exists as a file."""
        path = self._normalize(path)
        return path in self._tree

    def delete(self, path: str) -> bool:
        """Delete a file at path. Returns True if it existed."""
        path = self._normalize(path)
        if path in self._tree:
            del self._tree[path]
            return True
        return False

    def reset(self) -> None:
        """Clear all files."""
        self._tree.clear()

    def snapshot(self) -> dict[str, str]:
        """Return a copy of the current file tree."""
        return copy.deepcopy(self._tree)

    @staticmethod
    def _normalize(path: str) -> str:
        """Normalize a path: strip leading/trailing slashes, collapse doubles."""
        path = path.strip("/")
        while "//" in path:
            path = path.replace("//", "/")
        return path


class MockMCPSurface:
    """Deterministic mock surface with registry pattern.

    Only mocks actions that are explicitly registered.
    Unregistered actions raise ToolNotAvailableError.
    Maintains an ordered call log for evaluation.
    """

    def __init__(self, filesystem: VirtualFilesystem | None = None) -> None:
        self._registry: dict[str, Callable[..., dict[str, Any]]] = {}
        self._filesystem = filesystem or VirtualFilesystem()
        self._call_log: list[dict[str, Any]] = []

    @property
    def filesystem(self) -> VirtualFilesystem:
        """Access the virtual filesystem."""
        return self._filesystem

    def register(self, action: str, handler: Callable[..., dict[str, Any]]) -> None:
        """Register a mock handler for an MCP action.

        Args:
            action: Key in the form 'server/action_name' (e.g., 'corso/guard').
            handler: Callable that takes (params: dict) and returns a dict response.
        """
        self._registry[action] = handler

    def call(self, server: str, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """Call a registered mock action.

        Args:
            server: MCP server name (e.g., 'corso').
            action: Action name (e.g., 'guard').
            params: Parameters dict for the action.

        Returns:
            Response dict from the mock handler.

        Raises:
            ToolNotAvailableError: If the server/action combination is not registered.
        """
        key = f"{server}/{action}"
        if key not in self._registry:
            self._call_log.append({
                "server": server,
                "action": action,
                "params": params,
                "result": None,
                "error": f"ToolNotAvailableError: {key}",
            })
            raise ToolNotAvailableError(server, action)

        handler = self._registry[key]
        result = handler(params)

        self._call_log.append({
            "server": server,
            "action": action,
            "params": params,
            "result": result,
            "error": None,
        })

        return result

    def get_call_log(self) -> list[dict[str, Any]]:
        """Return the ordered log of all calls made (including failed ones)."""
        return list(self._call_log)

    def reset(self) -> None:
        """Reset state for new episode. Clears call log and filesystem."""
        self._call_log.clear()
        self._filesystem.reset()

    def registered_actions(self) -> list[str]:
        """Return a sorted list of all registered action keys."""
        return sorted(self._registry.keys())
