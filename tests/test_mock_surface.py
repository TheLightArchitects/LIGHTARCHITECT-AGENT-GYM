"""Tests for MockMCPSurface and VirtualFilesystem."""

from __future__ import annotations

import pytest

from mcp_gym.mock_surface import MockMCPSurface, VirtualFilesystem
from mcp_gym.types import ToolNotAvailableError


class TestVirtualFilesystem:
    """Tests for VirtualFilesystem CRUD operations."""

    def test_write_and_read(self) -> None:
        """Write a file and read it back."""
        vfs = VirtualFilesystem()
        vfs.write("src/main.rs", "fn main() {}")
        assert vfs.read("src/main.rs") == "fn main() {}"

    def test_read_nonexistent_returns_none(self) -> None:
        """Reading a path that does not exist returns None."""
        vfs = VirtualFilesystem()
        assert vfs.read("does/not/exist.txt") is None

    def test_exists(self) -> None:
        """Exists returns True for written files, False otherwise."""
        vfs = VirtualFilesystem()
        assert not vfs.exists("file.txt")
        vfs.write("file.txt", "content")
        assert vfs.exists("file.txt")

    def test_delete(self) -> None:
        """Delete removes a file and returns True; returns False if absent."""
        vfs = VirtualFilesystem()
        vfs.write("temp.txt", "temporary")
        assert vfs.delete("temp.txt") is True
        assert vfs.exists("temp.txt") is False
        assert vfs.delete("temp.txt") is False

    def test_list_dir(self) -> None:
        """List directory returns immediate children."""
        vfs = VirtualFilesystem()
        vfs.write("src/main.rs", "fn main() {}")
        vfs.write("src/lib.rs", "pub mod core;")
        vfs.write("src/core/mod.rs", "pub struct Core;")
        vfs.write("Cargo.toml", "[package]")

        # Root level
        root_children = vfs.list_dir("")
        assert "src" in root_children
        assert "Cargo.toml" in root_children

        # src level
        src_children = vfs.list_dir("src")
        assert "main.rs" in src_children
        assert "lib.rs" in src_children
        assert "core" in src_children

    def test_overwrite(self) -> None:
        """Writing to an existing path overwrites the content."""
        vfs = VirtualFilesystem()
        vfs.write("file.txt", "version 1")
        vfs.write("file.txt", "version 2")
        assert vfs.read("file.txt") == "version 2"

    def test_reset_clears_all(self) -> None:
        """Reset clears the entire filesystem."""
        vfs = VirtualFilesystem()
        vfs.write("a.txt", "a")
        vfs.write("b.txt", "b")
        vfs.reset()
        assert vfs.read("a.txt") is None
        assert vfs.read("b.txt") is None

    def test_snapshot_returns_copy(self) -> None:
        """Snapshot returns a deep copy that is independent of the original."""
        vfs = VirtualFilesystem()
        vfs.write("file.txt", "original")
        snap = vfs.snapshot()
        vfs.write("file.txt", "modified")
        assert snap["file.txt"] == "original"
        assert vfs.read("file.txt") == "modified"

    def test_normalize_paths(self) -> None:
        """Paths are normalized (stripped slashes, collapsed doubles)."""
        vfs = VirtualFilesystem()
        vfs.write("/src//main.rs/", "content")
        assert vfs.read("src/main.rs") == "content"


class TestMockMCPSurface:
    """Tests for MockMCPSurface registration and call mechanics."""

    def test_register_and_call(self) -> None:
        """Register a handler and call it successfully."""
        surface = MockMCPSurface()
        surface.register("corso/guard", lambda params: {"clean": True, "path": params.get("path")})

        result = surface.call("corso", "guard", {"path": "/src"})
        assert result["clean"] is True
        assert result["path"] == "/src"

    def test_unregistered_action_raises(self) -> None:
        """Calling an unregistered action raises ToolNotAvailableError."""
        surface = MockMCPSurface()
        with pytest.raises(ToolNotAvailableError) as exc_info:
            surface.call("corso", "nonexistent", {})
        assert "corso" in str(exc_info.value)
        assert "nonexistent" in str(exc_info.value)

    def test_call_log_records_success(self) -> None:
        """Successful calls are recorded in the call log."""
        surface = MockMCPSurface()
        surface.register("eva/speak", lambda params: {"response": "ok"})
        surface.call("eva", "speak", {"message": "hello"})

        log = surface.get_call_log()
        assert len(log) == 1
        assert log[0]["server"] == "eva"
        assert log[0]["action"] == "speak"
        assert log[0]["error"] is None
        assert log[0]["result"] == {"response": "ok"}

    def test_call_log_records_failures(self) -> None:
        """Failed calls (ToolNotAvailableError) are also recorded in the call log."""
        surface = MockMCPSurface()
        with pytest.raises(ToolNotAvailableError):
            surface.call("soul", "missing", {})

        log = surface.get_call_log()
        assert len(log) == 1
        assert log[0]["result"] is None
        assert "ToolNotAvailableError" in log[0]["error"]

    def test_reset_clears_log_and_filesystem(self) -> None:
        """Reset clears call log and filesystem but preserves registrations."""
        vfs = VirtualFilesystem()
        vfs.write("test.txt", "data")
        surface = MockMCPSurface(filesystem=vfs)
        surface.register("corso/guard", lambda params: {"clean": True})
        surface.call("corso", "guard", {})

        surface.reset()

        assert surface.get_call_log() == []
        assert not surface.filesystem.exists("test.txt")
        # Registrations survive reset
        result = surface.call("corso", "guard", {})
        assert result["clean"] is True

    def test_registered_actions_list(self) -> None:
        """registered_actions returns sorted list of all keys."""
        surface = MockMCPSurface()
        surface.register("eva/speak", lambda p: {})
        surface.register("corso/guard", lambda p: {})
        surface.register("corso/fetch", lambda p: {})

        actions = surface.registered_actions()
        assert actions == ["corso/fetch", "corso/guard", "eva/speak"]

    def test_multiple_calls_ordered(self) -> None:
        """Multiple calls appear in the log in order."""
        surface = MockMCPSurface()
        surface.register("corso/guard", lambda p: {"step": "guard"})
        surface.register("corso/fetch", lambda p: {"step": "fetch"})

        surface.call("corso", "guard", {})
        surface.call("corso", "fetch", {})

        log = surface.get_call_log()
        assert len(log) == 2
        assert log[0]["action"] == "guard"
        assert log[1]["action"] == "fetch"
