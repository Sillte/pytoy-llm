import fnmatch
import os
import re
from pathlib import Path
from typing import Annotated, Callable, Sequence

from pydantic import Field

from pytoy_llm.tools.errors import ToolError, ToolErrorKind
from pytoy_llm.tools.workspace_explorer.discovery import WorkspaceDiscovery
from pytoy_llm.tools.workspace_explorer.inspection import WorkspaceInspection
from pytoy_llm.tools.workspace_explorer.models import (
    WorkspaceAccess,
)
from pytoy_llm.tools.workspace_explorer.search import WorkspaceSearch

DEFAULT_EXCLUDE_NAMES = [".venv", "node_modules", ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__"]


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

    def __init__(
        self,
        workspace: Path,
        excludes: Sequence[str] | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.excludes = set(excludes or DEFAULT_EXCLUDE_NAMES)
        self.access = WorkspaceAccess.from_any(workspace=workspace, excludes=frozenset(self.excludes))
        self.discovery = WorkspaceDiscovery(self.access)
        self.inspection = WorkspaceInspection(self.access)
        self.search = WorkspaceSearch(self.access)

    @property
    def tools(self) -> Sequence[Callable]:
        return [
            *self.discovery.tools,
            *self.inspection.tools,
            *self.search.tools,
        ]
