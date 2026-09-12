import fnmatch
import os
import re
from pathlib import Path
from typing import Annotated, Callable, Final, Self, Sequence

from pydantic import Field

from pytoy_llm.tools.errors import ToolError, ToolErrorKind
from pytoy_llm.tools.workspace_explorer.discovery import WorkspaceDiscovery
from pytoy_llm.tools.workspace_explorer.inspection import WorkspaceInspection
from pytoy_llm.tools.workspace_explorer.models import (
    WorkspaceAccess,
)
from pytoy_llm.tools.workspace_explorer.search import WorkspaceSearch

DEFAULT_EXCLUDE_PATTERNS = [
    ".venv",
    "node_modules",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    "*.egg-info",
]


class WorkspaceExplorer:
    """
    Provide safe workspace exploration tools for LLM agents.

    This class provides read-only operations to inspect files,
    search contents, and understand project structure.
    It does not modify workspace files.

    Every path is interpreted relative to the workspace root.
    Files outside the workspace are never accessible.

    Common generated directories are excluded by default.
    """

    DEFAULT_EXCLUDE_PATTERNS: Final[frozenset[str]] = frozenset(DEFAULT_EXCLUDE_PATTERNS)

    def __init__(self, access: WorkspaceAccess) -> None:
        self.access = access

        self.discovery = WorkspaceDiscovery(self.access)
        self.inspection = WorkspaceInspection(self.access)
        self.search = WorkspaceSearch(self.access)

    @property
    def workspace(self) -> Path:
        return self.access.workspace

    @classmethod
    def from_any(
        cls,
        workspace: Path | str,
        excludes: set[str] | frozenset[str] | Sequence[str] | None = None,
    ) -> Self:
        excludes = frozenset(cls.DEFAULT_EXCLUDE_PATTERNS if excludes is None else excludes)
        workspace = Path(workspace).resolve()
        access = WorkspaceAccess.from_any(workspace=workspace, excludes=excludes)
        return cls(access=access)

    @property
    def tools(self) -> Sequence[Callable]:
        return [
            *self.discovery.tools,
            *self.inspection.tools,
            *self.search.tools,
        ]
