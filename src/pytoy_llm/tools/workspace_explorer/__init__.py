import fnmatch
import os
import re
from pathlib import Path
from typing import Annotated, Callable, Sequence

from pydantic import Field

from pytoy_llm.tools.errors import ToolError, ToolErrorKind
from pytoy_llm.tools.workspace_explorer.models import FileContent, FileInfo, FilePartContent, GrepMatch, TreeNode
from pytoy_llm.tools.workspace_explorer.semantic_types import (
    FileGlob,
    LineNumber,
    MaxDepth,
    MaxResults,
    SearchPattern,
    WorkspacePath,
)

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
        exclude_names: Sequence[str] | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.exclude_names = set(exclude_names or DEFAULT_EXCLUDE_NAMES)

    @property
    def tools(self) -> Sequence[Callable]:
        return [
            self.list_files,
            self.read_file,
            self.read_file_range,
            self.find_files,
            self.tree,
            self.grep,
        ]

    # ---------------------------------------------------------

    def _resolve(self, path: WorkspacePath) -> Path | ToolError:
        p = (self.workspace / path).resolve()

        if not p.is_relative_to(self.workspace):
            return ToolError(kind=ToolErrorKind.INVALID_ARGUMENT, msg="Path is outside workspace", retry=False)
        return p

    def _excluded(self, path: Path) -> bool:
        return path.name in self.exclude_names

    # ---------------------------------------------------------

    def list_files(
        self,
        path: WorkspacePath = "",
        recursive: Annotated[
            bool,
            Field(description="Whether to traverse subdirectories recursively."),
        ] = False,
    ) -> list[FileInfo] | ToolError:
        """
        List files and directories under a directory in the workspace.

        Use this tool when you need to understand the structure
        of a workspace or inspect available files.

        This tool provides metadata only.
        It does not return file contents.

        Typical usage:
            - Explore an unfamiliar project.
            - Check which files exist before reading them.
            - Inspect a directory before choosing another tool.

        Prefer:
            - tree() when you need a hierarchical overview.
            - find_files() when searching by filename pattern.
            - grep() when searching by file content.

        Args:
            path:
                Relative path from the workspace root.
                Empty path means the workspace root.

            recursive:
                Whether to traverse all nested directories.
                For large workspaces, prefer tree() or find_files()
                when possible.

        Returns:
            A list of FileInfo objects.

        ToolError when:
            - the path is outside the workspace
            - the directory cannot be accessed
            - another filesystem error occurs

        """

        root = self._resolve(path)
        if isinstance(root, ToolError):
            return root

        if not recursive:
            try:
                return [FileInfo.from_absolute_path(p, self.workspace) for p in sorted(root.iterdir()) if not self._excluded(p)]
            except OSError as exc:
                return ToolError(kind=ToolErrorKind.UNKNOWN, msg=str(exc), retry=False)

        result: list[FileInfo] = []

        for current, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if not self._excluded(Path(current) / d)]

            for d in dirs:
                result.append(FileInfo.from_absolute_path(Path(current) / d, self.workspace))

            for f in files:
                p = Path(current) / f

                if self._excluded(p):
                    continue

                result.append(FileInfo.from_absolute_path(p, self.workspace))

        return sorted(result, key=lambda x: x.path)

    # ---------------------------------------------------------

    def read_file(self, path: WorkspacePath, max_bytes: int = 1024 * 1024) -> FileContent | ToolError:
        """
        Read the complete contents of a text file.

        Use this tool when the exact file content is required.
        Prefer grep() or find_files() first when the target file
        is not known yet.

        Args:
            path:
                Path relative to the workspace root.

            max_bytes:
                Maximum allowed file size in bytes.
                Large files should be inspected with read_file_range()
                when only a portion of the content is needed.

        Returns:
            FileContent containing the requested file contents.

            ToolError when:
                - the file does not exist
                - the file is outside the workspace
                - the file exceeds the size limit
                - the path is invalid
                - the file cannot be read

        Notes:
            This tool reads the file as UTF-8 text.
            Binary files are not supported.
        """

        abs_path = self._resolve(path)
        if isinstance(abs_path, ToolError):
            return abs_path

        try:
            size = abs_path.stat().st_size
        except OSError as exc:
            return ToolError(kind=ToolErrorKind.UNKNOWN, msg=str(exc))
        else:
            if max_bytes < size:
                return ToolError(
                    kind=ToolErrorKind.RESOURCE_LIMIT, msg="The target file is too big.", suggestion="Use different tools."
                )

        try:
            return FileContent(path=path, content=abs_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            return ToolError(kind=ToolErrorKind.NOT_FOUND, msg=str(exc), retry=None)
        except OSError as exc:
            return ToolError(kind=ToolErrorKind.UNKNOWN, msg=str(exc))

    # ---------------------------------------------------------

    def read_file_range(
        self,
        path: WorkspacePath,
        start_line: LineNumber,
        end_line: LineNumber,
    ) -> FilePartContent | ToolError:
        """
        Read a specific line range from a text file.

        Use this tool when:
            - the target file is already known
            - only a portion of the file needs to be inspected

        Prefer read_file() when the entire file content is required.
        Prefer grep() or find_files() when the target file is not known yet.

        Args:
            path:
                Path relative to the workspace root.

            start_line:
                the start line number of the range. Zero-based and inclusive.

            end_line:
                the end line number of the range. Zero-based and exclusive.

        Returns:
            FilePartContent containing the requested file contents.

            The returned range follows Python slice semantics:
            start_line is inclusive and end_line is exclusive.

        ToolError when:
            - the file does not exist
            - the file is outside the workspace
            - the operation cannot be completed

        Notes:
            This tool reads the file as UTF-8 text.
            Binary files are not supported.

        """
        abs_path = self._resolve(path)
        if isinstance(abs_path, ToolError):
            return abs_path
        try:
            lines = abs_path.read_text(encoding="utf-8").splitlines()
            return FilePartContent(
                path=path, content="\n".join(lines[start_line:end_line]), start_line=start_line, end_line=end_line
            )
        except FileNotFoundError as exc:
            return ToolError(kind=ToolErrorKind.NOT_FOUND, msg=str(exc), retry=None)
        except OSError as exc:
            return ToolError(kind=ToolErrorKind.UNKNOWN, msg=str(exc))

    # ---------------------------------------------------------

    def find_files(
        self,
        pattern: SearchPattern,
        path: WorkspacePath = "",
    ) -> list[FileInfo] | ToolError:
        """
        Find files recursively by file name pattern.

        This tool searches files under `path`.
        The pattern is matched against the file name only,
        not the full relative path.

        Examples:

            *.py
            *.md
            test_*.py

        Use this tool when you need to locate multiple files
        whose names satisfy a certain pattern.

        Args:
            pattern:
                File name glob pattern.

            path:
                Directory relative to the workspace root.
                Search starts from this directory.

        Returns:
            A list of matching files.

            ToolError when:
                - the path is outside the workspace
                - the search directory cannot be accessed
                - another filesystem error occurs.

        Notes:
            - Matching is performed against file names only.
            - Excluded directories are skipped.
        """

        root = self._resolve(path)
        if isinstance(root, ToolError):
            return root

        result = []

        try:
            for current, dirs, files in os.walk(root):
                dirs[:] = [d for d in dirs if not self._excluded(Path(current) / d)]

                for file_name in files:
                    file_path = Path(current) / file_name

                    if self._excluded(file_path):
                        continue

                    if fnmatch.fnmatch(file_name, pattern):
                        result.append(
                            FileInfo.from_absolute_path(
                                file_path,
                                self.workspace,
                            )
                        )

        except OSError as exc:
            return ToolError(
                kind=ToolErrorKind.UNKNOWN,
                msg=str(exc),
                retry=False,
            )

        return sorted(result, key=lambda x: x.path)

    # ---------------------------------------------------------

    def grep(
        self,
        pattern: SearchPattern,
        path: WorkspacePath = "",
        file_pattern: FileGlob = "*",
        regex: Annotated[
            bool,
            Field(description="Interpret the search pattern as a regular expression."),
        ] = False,
        case_sensitive: Annotated[
            bool,
            Field(description="Whether the search is case-sensitive."),
        ] = False,
        max_results: MaxResults = 100,
    ) -> list[GrepMatch] | ToolError:
        """
        Search text contents inside workspace files.

        This tool searches file contents line by line
        using plain text matching or regular expressions.

        Prefer to use this tool before read_file()
        when you need to locate symbols, functions,
        classes, configuration keys, or other text.

        Prefer specific `pattern` and `file_pattern`
        to avoid unnecessary file scanning.

        The search is performed recursively under `path`.

        Args:
            pattern:
                Text to search for.
                When regex=True, this is interpreted
                as a regular expression.

            file_pattern:
                File name glob pattern used to filter searched files.
                Examples: '*.py', '*.ts'.

            regex:
                Whether to interpret `pattern` as a regular expression.

            case_sensitive:
                Whether matching should be case-sensitive.

            max_results:
                Maximum number of returned matches.

        Returns:
            A list of `GrepMatch`.

            Each match contains:
                - relative file path
                - zero-based line number
                - zero-based column index
                - entire line containing the match

        ToolError when:
            - the path is outside the workspace
            - regex compilation fails when regex=True
            - another filesystem error occurs

        Notes:
            - Search is recursive.
            - Excluded directories are skipped.
            - Results are limited by max_results.
        """

        root = self._resolve(path)
        if isinstance(root, ToolError):
            return root
        flags = 0 if case_sensitive else re.IGNORECASE

        compiled = None
        if regex:
            try:
                compiled = re.compile(pattern, flags)
            except re.error as exc:
                return ToolError(
                    kind=ToolErrorKind.INVALID_ARGUMENT,
                    msg=str(exc),
                    retry=False,
                )

        results: list[GrepMatch] = []

        try:
            for current, dirs, files in os.walk(root):
                if len(results) >= max_results:
                    break

                dirs[:] = [d for d in dirs if not self._excluded(Path(current) / d)]

                for file_name in files:
                    if len(results) >= max_results:
                        break

                    file_path = Path(current) / file_name

                    if self._excluded(file_path):
                        continue

                    if not fnmatch.fnmatch(file_name, file_pattern):
                        continue

                    try:
                        lines = file_path.read_text(
                            encoding="utf-8",
                            errors="ignore",
                        ).splitlines()

                    except OSError:
                        continue

                    for lineno, line in enumerate(lines):
                        if regex:
                            assert compiled is not None
                            match = compiled.search(line)

                            if match:
                                results.append(
                                    GrepMatch(
                                        path=str(file_path.relative_to(self.workspace)),
                                        line=lineno,
                                        column=match.start(),
                                        text=line,
                                    )
                                )

                        else:
                            source = line if case_sensitive else line.lower()
                            target = pattern if case_sensitive else pattern.lower()

                            idx = source.find(target)

                            if idx != -1:
                                results.append(
                                    GrepMatch(
                                        path=str(file_path.relative_to(self.workspace)),
                                        line=lineno,
                                        column=idx,
                                        text=line,
                                    )
                                )

                        if len(results) >= max_results:
                            break

        except OSError as exc:
            return ToolError(
                kind=ToolErrorKind.UNKNOWN,
                msg=str(exc),
                retry=False,
            )

        return results

    def tree(
        self,
        path: WorkspacePath = "",
        max_depth: MaxDepth = 3,
    ) -> TreeNode | ToolError:
        """
        Return a directory tree for workspace exploration.

        This tool helps understand the project structure
        before reading individual files.

        Use this tool when:
            - the project layout is not known yet
            - you need to discover directories or files
            - you need to decide which files should be inspected next

        The tree is generated recursively under `path`.
        Directory expansion stops when `max_depth` is reached.

        Args:
            path:
                Directory path relative to the workspace root.
                The tree starts from this directory.

            max_depth:
                Maximum depth of directories to traverse.
                The root directory is depth 0.
                Deeper directories are not included.

        Returns:
            A `TreeNode` representing the directory hierarchy.

            Each node contains:
                - file or directory name
                - relative path
                - whether it is a directory
                - immediate child nodes for directories

        ToolError when:
            - the path is outside the workspace
            - the directory cannot be accessed
            - another filesystem error occurs

        Notes:
            - Excluded files and directories are omitted.
            - Use read_file() after identifying target files.
        """

        try:
            root = self._resolve(path)
            if isinstance(root, ToolError):
                return root

            def build_node(
                current: Path,
                depth: int,
            ) -> TreeNode:

                if current.is_dir() and depth < max_depth:
                    children = []

                    for child in sorted(
                        current.iterdir(),
                        key=lambda p: (
                            not p.is_dir(),
                            p.name.lower(),
                        ),
                    ):
                        if self._excluded(child):
                            continue

                        children.append(
                            build_node(
                                child,
                                depth + 1,
                            )
                        )

                else:
                    children = []

                return TreeNode(
                    name=current.name,
                    path=str(current.relative_to(self.workspace)),
                    is_dir=current.is_dir(),
                    children=children,
                )

            return build_node(root, 0)

        except OSError as exc:
            return ToolError(kind=ToolErrorKind.UNKNOWN, msg=str(exc))
