from pathlib import Path
from typing import Annotated, Self

from pydantic import Field

from pytoy_llm.materials.text_files import PathGatherer
from pytoy_llm.tools.errors import ToolError, ToolErrorKind
from pytoy_llm.tools.workspace_explorer.models import DirectoryInfo, FileInfo, WorkspaceAccess
from pytoy_llm.tools.workspace_explorer.semantic_types import GlobPattern, WorkspacePath

DEFAULT_EXCLUDE_NAMES = [".venv", "node_modules", ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__"]


class WorkspaceDiscovery:
    """
    Provide safe workspace discovery tools for LLM agents.

    This class provides read-only operations to discover files and understand project structure.
    It does not modify workspace files.

    Every path is interpreted relative to the workspace root.
    Files outside the workspace are never accessible.
    """

    def __init__(self, access: WorkspaceAccess) -> None:
        self.access = access
        self.workspace = access.workspace.resolve()
        self.excludes = access.excludes

    @classmethod
    def from_any(cls, workspace: Path | str, excludes: frozenset[str] | None = None) -> Self:
        return cls(access=WorkspaceAccess.from_any(workspace=workspace, excludes=excludes))

    @property
    def tools(
        self,
    ):
        return [self.find_paths]

    def find_paths(
        self,
        collection_root: WorkspacePath,
        patterns: Annotated[
            tuple[GlobPattern, ...], Field(description="Glob pattern matched against paths relative to `collection_root`.")
        ] = ("*",),
    ) -> list[FileInfo | DirectoryInfo] | ToolError:
        """
        Find files and directories matching a glob pattern within the workspace.

        Use this tool to discover candidate paths before inspecting their contents.
        It returns path metadata only; file contents are not read or included.

        Args:
            collection_root:
                Directory relative to the workspace root from which the search starts.
                Returned paths are relative to the workspace root.

            patterns:
                Tuple of glob pattern matched against paths relative to `collection_root`.

        Returns:
            A tuple of `FileInfo` and `DirectoryInfo` objects matching the pattern.

        Notes:
            - Paths outside the workspace are never accessible.
            - The search traverses the collection root recursively.
            - File contents are never read or included.
        """

        try:
            root = self.access.to_absolute(collection_root)
        except Exception as e:
            return ToolError(kind=ToolErrorKind.INVALID_ARGUMENT, msg=f"Invalid `{collection_root=}`; {e}")

        try:
            paths = PathGatherer().gather(root=root, max_depth=None, excludes=self.excludes, target="all", patterns=patterns)
        except ValueError as e:
            return ToolError(kind=ToolErrorKind.INVALID_ARGUMENT, msg=f"`{collection_root=}` is invalid; {e}")

        def to_model(path: Path) -> FileInfo | DirectoryInfo:
            if path.is_dir():
                return DirectoryInfo.from_absolute_path(path, self.workspace)
            return FileInfo.from_absolute_path(path, self.workspace)

        return [to_model(path) for path in paths]

    def tree(
        self,
        collection_root: WorkspacePath = ".",
    ) -> str | ToolError:
        """ """
        try:
            root = self.access.to_absolute(collection_root)
        except Exception as e:
            return ToolError(kind=ToolErrorKind.INVALID_ARGUMENT, msg=f"Invalid `{collection_root=}`; {e}")

        try:
            paths = PathGatherer().gather(root=root, max_depth=None, excludes=self.excludes, target="all")
        except ValueError as e:
            return ToolError(kind=ToolErrorKind.INVALID_ARGUMENT, msg=f"`{collection_root=}` is invalid; {e}")
