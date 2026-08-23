from pathlib import Path
from typing import Annotated, Self, Sequence

from pydantic import Field

from pytoy_llm.foundation.paths import PathGatherer, PathTree
from pytoy_llm.tools.errors import ToolError, ToolErrorKind
from pytoy_llm.tools.workspace_explorer.models import DirectoryInfo, FileInfo, WorkspaceAccess
from pytoy_llm.tools.workspace_explorer.semantic_types import GlobPattern, MaxResults, WorkspacePath

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
        return [self.find_paths, self.tree, self.recent_files]

    def find_paths(
        self,
        collection_root: WorkspacePath,
        patterns: Annotated[
            GlobPattern | Sequence[GlobPattern],
            Field(description="Glob pattern matched against paths relative to `collection_root`."),
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
                Glob pattern or glob patterns matched against paths relative to `collection_root`.

        Returns:
            A list of `FileInfo` and `DirectoryInfo` objects matching the pattern.

        Notes:
            - Paths outside the workspace are never accessible.
            - The search traverses the collection root recursively.
            - File contents are never read or included.
        """
        if isinstance(patterns, str):
            patterns = [patterns]

        root = self.access.resolve(collection_root)
        if isinstance(root, ToolError):
            return root

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
        """
        Render a workspace-relative tree for paths under a collection root.

        Paths are collected recursively from `collection_root`, while the
        resulting tree is structured relative to the workspace root.

        Use this tool to understand the structure of a workspace or a specific
        directory before investigating individual files.

        Args:
            collection_root:
                Directory relative to the workspace root from which paths are
                collected.

        Returns:
            A plain-text directory tree containing paths under `collection_root`,
            represented relative to the workspace root, or a `ToolError` if the
            collection root is invalid.

        Notes:
            - Paths outside the workspace are never accessible.
            - File contents are never read or included.
        """
        try:
            root = self.access.resolve(collection_root)
            if isinstance(root, ToolError):
                return root
        except Exception as e:
            return ToolError(kind=ToolErrorKind.INVALID_ARGUMENT, msg=f"Invalid `{collection_root=}`; {e}")

        try:
            paths = PathGatherer().gather(root=root, max_depth=None, excludes=self.excludes, target="all")
        except ValueError as e:
            return ToolError(kind=ToolErrorKind.INVALID_ARGUMENT, msg=f"`{collection_root=}` is invalid; {e}")

        if not paths:
            if root == self.workspace:
                return ""
            tree = PathTree.from_paths([root], root_path=self.workspace)
            return tree.render(include_root=True)

        try:
            tree = PathTree.from_paths(paths, root_path=self.workspace)
        except ValueError as e:
            return ToolError(kind=ToolErrorKind.UNKNOWN, msg=f"`{collection_root=}` is invalid in `PathTree`; {e}")

        return tree.render(include_root=False)

    def recent_files(
        self,
        collection_root: WorkspacePath = ".",
        max_results: MaxResults = 10,
    ) -> list[FileInfo] | ToolError:
        """
        Find recently modified files within the workspace.

        Use this tool to identify files that have been modified most recently,
        especially when investigating recent work or deciding where to inspect first.

        FileInfos are sorted by last modification time in descending order.

        Args:
            collection_root:
                Directory relative to the workspace root from which the search starts.

            max_results:
                Maximum number of files to return.

        Returns:
            FileInfo objects sorted from newest to oldest by modification time.
        """

        root = self.access.resolve(collection_root)
        if isinstance(root, ToolError):
            return root

        try:
            paths = PathGatherer().gather(root=root, max_depth=None, excludes=self.excludes, target="file")
        except ValueError as e:
            return ToolError(kind=ToolErrorKind.INVALID_ARGUMENT, msg=f"`{collection_root=}` is invalid; {e}")
        file_infos = sorted(
            (FileInfo.from_absolute_path(path, self.workspace) for path in paths),
            key=lambda file_info: file_info.modified,
            reverse=True,
        )
        return file_infos[:max_results]


if __name__ == "__main__":
    explorer = WorkspaceDiscovery.from_any(Path("../../../../"))
    print(explorer.tree("."))
