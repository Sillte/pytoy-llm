from pathlib import Path
from typing import Annotated, Self, Sequence

from pydantic import Field

from pytoy_llm.foundation.paths import PathGatherer, PathTree
from pytoy_llm.tools.errors import ToolError, ToolErrorKind
from pytoy_llm.tools.workspace_explorer.models import DirectoryInfo, FileInfo, WorkspaceAccess
from pytoy_llm.tools.workspace_explorer.semantic_types import (
    GlobPattern,
    MaxResults,
    WorkspaceDirectoryPath,
    WorkspaceDirectoryPivot,
)


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
        return [self.workspace_find_directories_and_files, self.workspace_tree, self.recent_files]

    def workspace_find_directories_and_files(
        self,
        collection_root: WorkspaceDirectoryPivot,
        patterns: Annotated[
            GlobPattern | Sequence[GlobPattern],
            Field(description="Glob pattern matched against paths relative to `collection_root`."),
        ] = ("*",),
    ) -> list[DirectoryInfo | FileInfo] | ToolError:
        """Find directories and files matching glob patterns under a directory.

        Returns metadata for matching paths without reading file contents.
        """

        if isinstance(patterns, str):
            patterns = [patterns]

        root = self.access.resolve(collection_root)
        if isinstance(root, ToolError):
            return root

        try:
            paths = tuple(
                path
                for path in PathGatherer().gather(
                    root=root,
                    max_depth=None,
                    excludes=self.excludes,
                    target="all",
                    patterns=patterns,
                )
                if self.access.is_within_workspace(path)
            )
        except ValueError as e:
            return ToolError(
                kind=ToolErrorKind.INVALID_ARGUMENT, msg=f"`{collection_root=}` is invalid; {e}"
            )
        except PermissionError as e:
            return ToolError(
                kind=ToolErrorKind.PERMISSION_DENIED,
                msg=f"Could not inspect `{collection_root}`: {e}",
                retry=False,
            )
        except OSError as e:
            return ToolError(
                kind=ToolErrorKind.IO_ERROR, msg=f"Could not inspect `{collection_root}`: {e}"
            )

        def to_model(path: Path) -> FileInfo | DirectoryInfo:
            if path.is_dir():
                return DirectoryInfo.from_absolute_path(path, self.workspace)
            return FileInfo.from_absolute_path(path, self.workspace)

        try:
            return [to_model(path) for path in paths]
        except PermissionError as e:
            return ToolError(
                kind=ToolErrorKind.PERMISSION_DENIED,
                msg=f"Could not read metadata under `{collection_root}`: {e}",
                retry=False,
            )
        except OSError as e:
            return ToolError(
                kind=ToolErrorKind.IO_ERROR,
                msg=f"Could not read metadata under `{collection_root}`: {e}",
            )

    def workspace_tree(
        self,
        collection_root: WorkspaceDirectoryPath = ".",
    ) -> str | ToolError:
        """Render the directory and file structure under a workspace directory.

        Use this to inspect workspace structure without reading file contents.

        Args:
            collection_root:
                Directory relative to the workspace root from which paths are
                collected.

        Returns:
            A plain-text directory tree containing paths under `collection_root`,
            represented relative to the workspace root, or a `ToolError` if the
            collection root is invalid.
        """
        root = self.access.resolve(collection_root)
        if isinstance(root, ToolError):
            return root

        try:
            paths = tuple(
                path
                for path in PathGatherer().gather(
                    root=root, max_depth=None, excludes=self.excludes, target="all"
                )
                if self.access.is_within_workspace(path)
            )
        except ValueError as e:
            return ToolError(
                kind=ToolErrorKind.INVALID_ARGUMENT, msg=f"`{collection_root=}` is invalid; {e}"
            )
        except OSError as e:
            return ToolError(
                kind=ToolErrorKind.IO_ERROR,
                msg=f"Could not inspect `{collection_root}`: {e}",
                retry=True,
            )

        if not paths:
            if root == self.workspace:
                return ""
            try:
                tree = PathTree.from_paths([root], root_path=self.workspace)
                return tree.render(include_root=True)
            except PermissionError as e:
                return ToolError(
                    kind=ToolErrorKind.PERMISSION_DENIED,
                    msg=f"Could not read `{collection_root}`: {e}",
                    retry=False,
                )
            except OSError as e:
                return ToolError(
                    kind=ToolErrorKind.IO_ERROR, msg=f"Could not read `{collection_root}`: {e}"
                )

        try:
            tree = PathTree.from_paths(paths, root_path=self.workspace)
        except ValueError as e:
            return ToolError(
                kind=ToolErrorKind.UNKNOWN,
                msg=f"`{collection_root=}` is invalid in `PathTree`; {e}",
            )
        except PermissionError as e:
            return ToolError(
                kind=ToolErrorKind.PERMISSION_DENIED,
                msg=f"Could not read `{collection_root}`: {e}",
                retry=False,
            )
        except OSError as e:
            return ToolError(
                kind=ToolErrorKind.IO_ERROR, msg=f"Could not read `{collection_root}`: {e}"
            )

        return tree.render(include_root=False)

    def recent_files(
        self,
        collection_root: WorkspaceDirectoryPath = ".",
        max_results: MaxResults = 10,
    ) -> list[FileInfo] | ToolError:
        """Find the most recently modified files under a workspace directory.

        Results are sorted by modification time, newest first.
        """
        root = self.access.resolve(collection_root)
        if isinstance(root, ToolError):
            return root

        try:
            paths = tuple(
                path
                for path in PathGatherer().gather(
                    root=root, max_depth=None, excludes=self.excludes, target="file"
                )
                if self.access.is_within_workspace(path)
            )
        except ValueError as e:
            return ToolError(
                kind=ToolErrorKind.INVALID_ARGUMENT, msg=f"`{collection_root=}` is invalid; {e}"
            )
        except PermissionError as e:
            return ToolError(
                kind=ToolErrorKind.PERMISSION_DENIED,
                msg=f"Could not inspect `{collection_root}`: {e}",
                retry=False,
            )
        except OSError as e:
            return ToolError(
                kind=ToolErrorKind.IO_ERROR, msg=f"Could not inspect `{collection_root}`: {e}"
            )
        try:
            file_infos = sorted(
                (FileInfo.from_absolute_path(path, self.workspace) for path in paths),
                key=lambda file_info: file_info.modified_at,
                reverse=True,
            )
        except PermissionError as e:
            return ToolError(
                kind=ToolErrorKind.PERMISSION_DENIED,
                msg=f"Could not read file metadata under `{collection_root}`: {e}",
                retry=False,
            )
        except OSError as e:
            return ToolError(
                kind=ToolErrorKind.IO_ERROR,
                msg=f"Could not read file metadata under `{collection_root}`: {e}",
            )
        return file_infos[:max_results]


if __name__ == "__main__":
    explorer = WorkspaceDiscovery.from_any(Path("../../../../"))
    print(explorer.workspace_tree("."))
