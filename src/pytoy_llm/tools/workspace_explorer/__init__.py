import fnmatch
import os
import re
from pathlib import Path
from typing import Annotated, Callable, Sequence

from pydantic import Field

from pytoy_llm.tools.errors import ToolError, ToolErrorKind
from pytoy_llm.tools.workspace_explorer.models import (
    DirectoryInfo,
    FileContent,
    FileInfo,
    FilePartContent,
    GrepContext,
    GrepMatch,
    TreeNode,
)
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
            self.read_files,
            self.read_file_range,
            self.find_files,
            self.find_directories,
            self.tree,
            self.grep,
            self.grep_context,
            self.recent_files,
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

    def read_file(
        self,
        path: WorkspacePath,
        max_lines: int | None = 20,
    ) -> FilePartContent | FileContent | ToolError:
        """
        Read a text file, returning its complete contents when it fits
        within the line limit, otherwise, returning a partial content.


        Args:
            path:
                Path relative to the workspace root.

            max_lines:
                Maximum number of lines to read.
                - integer: return at most this many lines.
                - null: read the entire file.

        Returns:
            FileContent:
                The complete file contents.

            FilePartContent:
                The first `max_lines` lines of the file.

            ToolError:
                When the file does not exist, is outside the workspace,
                cannot be read, or the argument is invalid.

        Notes:
            This tool reads the file as UTF-8 text.
        """
        if max_lines is not None and max_lines < 1:
            return ToolError(
                kind=ToolErrorKind.INVALID_ARGUMENT,
                msg="`max_lines` must be greater than or equal to 1.",
                retry=False,
            )

        abs_path = self._resolve(path)
        if isinstance(abs_path, ToolError):
            return abs_path

        if max_lines is None:
            return FileContent(path=path, content=abs_path.read_text(encoding="utf8"))

        try:
            with abs_path.open("r", encoding="utf-8") as f:
                lines = []

                for _ in range(max_lines):
                    line = f.readline()

                    if line == "":
                        return FileContent(
                            path=path,
                            content="".join(lines),
                        )

                    lines.append(line)

                if f.readline() == "":
                    return FileContent(
                        path=path,
                        content="".join(lines),
                    )

                return FilePartContent(
                    path=path,
                    content="".join(lines),
                    start_line=0,
                    end_line=max_lines,
                )

        except FileNotFoundError as exc:
            return ToolError(
                kind=ToolErrorKind.NOT_FOUND,
                msg=str(exc),
                retry=None,
            )
        except OSError as exc:
            return ToolError(
                kind=ToolErrorKind.UNKNOWN,
                msg=str(exc),
            )

    # ---------------------------------------------------------
    def read_files(
        self,
        paths: Sequence[WorkspacePath],
        max_lines: int | None = 10,
    ) -> list[FileContent | FilePartContent] | ToolError:
        """
        Read multiple text files from the workspace.

        Each file is read using the same line limit as read_file().
        Args:
            paths:
                File paths relative to the workspace root.

            max_lines:
                - integer: return at most this many lines.
                - null: read the entire file.

        Returns:
            A list containing FileContent or FilePartContent for each
            requested file, in the same order as `paths`.

            ToolError when:
                - `max_lines` is invalid
                - any path is outside the workspace
                - any requested file does not exist
                - any file cannot be read

        Notes:
            - The line limit applies independently to each file.
            - FileContent is returned when the complete file fits within
              `max_lines`.
            - FilePartContent is returned when the file exceeds `max_lines`.
            - Binary files are not supported.
        """
        if max_lines is not None and max_lines < 1:
            return ToolError(
                kind=ToolErrorKind.INVALID_ARGUMENT,
                msg="`max_lines` must be greater than or equal to 1.",
                retry=False,
            )

        results: list[FileContent | FilePartContent] = []

        for path in paths:
            result = self.read_file(
                path=path,
                max_lines=max_lines,
            )

            if isinstance(result, ToolError):
                return result

            results.append(result)

        return results

    def read_file_range(
        self,
        path: WorkspacePath,
        start_line: LineNumber,
        end_line: LineNumber,
    ) -> FilePartContent | ToolError:
        """
        Read a specific line range from a text file.
        Use this tool when surrounding context is needed around a specific part of a file.

        Args:
            path:
                Path relative to the workspace root.

            start_line:
                the start line number of the range. Zero-based and inclusive.

            end_line:
                the end line number of the range. Zero-based and exclusive.

        Returns:
            FilePartContent containing the requested partial file content.
            Start_line is inclusive and end_line is exclusive.

        ToolError when:
            - the file does not exist
            - the file is outside the workspace
            - the operation cannot be completed

        Notes:
            This tool reads the file as UTF-8 text.
        """
        if end_line <= start_line:
            return ToolError(
                kind=ToolErrorKind.INVALID_ARGUMENT,
                msg="end_line must be greater than start_line.",
                retry=False,
            )
        if start_line < 0:
            return ToolError(kind=ToolErrorKind.INVALID_ARGUMENT, msg="start_line must be a non-negative integer.")
        abs_path = self._resolve(path)
        if isinstance(abs_path, ToolError):
            return abs_path
        try:
            lines = abs_path.read_text(encoding="utf-8").splitlines()

            if len(lines) < end_line:
                return ToolError(
                    kind=ToolErrorKind.INVALID_ARGUMENT,
                    msg=f"The file has {len(lines)}, so `{end_line=}` is out of range.",
                )
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

    def find_directories(
        self,
        pattern: SearchPattern,
        path: WorkspacePath = "",
    ) -> list[DirectoryInfo] | ToolError:
        """
        Find directories recursively by directory name pattern.

        Use this tool when you need to locate directories
        whose names satisfy a certain pattern.

        Args:
            pattern:
                Directory name glob pattern.

            path:
                Directory relative to the workspace root.
                Search starts from this directory.

        Returns:
            A list of matching directories.

            ToolError when:
                - the path is outside the workspace
                - the search directory cannot be accessed
                - another filesystem error occurs.

        Notes:
            - Matching is performed against directory names only.
            - Excluded directories are skipped.
        """

        root = self._resolve(path)
        if isinstance(root, ToolError):
            return root

        result = []

        try:
            for current, dirs, _files in os.walk(root):
                dirs[:] = [d for d in dirs if not self._excluded(Path(current) / d)]

                for dir_name in dirs:
                    dir_path = Path(current) / dir_name

                    if fnmatch.fnmatch(dir_name, pattern):
                        result.append(
                            DirectoryInfo.from_absolute_path(
                                dir_path,
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

    def grep_context(
        self,
        pattern: SearchPattern,
        path: WorkspacePath = "",
        file_pattern: FileGlob = "*",
        context_lines: Annotated[
            int,
            Field(
                ge=0,
                description=("Number of lines to include before and after each match."),
            ),
        ] = 3,
        regex: Annotated[
            bool,
            Field(description="Interpret the search pattern as a regular expression."),
        ] = False,
        case_sensitive: Annotated[
            bool,
            Field(description="Whether the search is case-sensitive."),
        ] = False,
        max_results: MaxResults = 100,
    ) -> list[GrepContext] | ToolError:
        """
        Search workspace files and return the surrounding context of each match.

        This tool combines content search with local file context. Each returned
        GrepContext contains a contiguous range of file contents together with
        the grep matches contained in that range.

        Args:
            pattern:
                Text to search for. When regex=True, this is interpreted as a
                regular expression.

            path:
                Directory path relative to the workspace root. Search starts from
                this directory.

            file_pattern:
                File name glob pattern used to filter searched files.
                Examples: '*.py', '*.ts'.

            context_lines:
                Number of lines to include before and after each matching line.
                A value of 0 returns only the matching lines.

            regex:
                Whether to interpret pattern as a regular expression.

            case_sensitive:
                Whether matching should be case-sensitive.

            max_results:
                Maximum number of grep matches to process.

        Returns:
            A list of GrepContext objects.

            Each GrepContext represents one contiguous region of a file and
            contains:
                - the relative file path
                - the content of that region
                - the zero-based start line, inclusive
                - the zero-based end line, exclusive
                - the grep matches contained in the region

            Overlapping or adjacent context ranges from nearby matches are merged into a
            single GrepContext.

        ToolError when:
            - the search path is outside the workspace
            - the search pattern is invalid when regex=True
            - the search cannot be completed
            - a matched file cannot be read

        Notes:
            - Line numbers are zero-based.
            - start_line is inclusive and end_line is exclusive.
            - Context ranges are merged when they overlap.
            - The returned content preserves the original file text, including
              line endings.
            - The search semantics are the same as grep().
        """

        matches = self.grep(
            pattern=pattern,
            path=path,
            file_pattern=file_pattern,
            regex=regex,
            case_sensitive=case_sensitive,
            max_results=max_results,
        )

        if isinstance(matches, ToolError):
            return matches

        if not matches:
            return []

        # Group matches by file.
        matches_by_path: dict[str, list[GrepMatch]] = {}

        for match in matches:
            matches_by_path.setdefault(str(match.path), []).append(match)

        contexts: list[GrepContext] = []

        for match_path, file_matches in matches_by_path.items():
            file_matches.sort(key=lambda match: match.line)

            # Create context ranges around each match.
            ranges: list[tuple[int, int, list[GrepMatch]]] = []

            for match in file_matches:
                start_line = max(0, match.line - context_lines)
                end_line = match.line + context_lines + 1

                ranges.append(
                    (
                        start_line,
                        end_line,
                        [match],
                    )
                )

            # Merge overlapping ranges.
            merged: list[tuple[int, int, list[GrepMatch]]] = []

            for start_line, end_line, range_matches in ranges:
                if not merged:
                    merged.append(
                        (
                            start_line,
                            end_line,
                            range_matches,
                        )
                    )
                    continue

                previous_start, previous_end, previous_matches = merged[-1]

                if start_line <= previous_end:
                    merged[-1] = (
                        previous_start,
                        max(previous_end, end_line),
                        previous_matches + range_matches,
                    )
                else:
                    merged.append(
                        (
                            start_line,
                            end_line,
                            range_matches,
                        )
                    )

            # Read each merged range.
            for start_line, end_line, range_matches in merged:
                content = self.read_file_range(
                    path=match_path,
                    start_line=start_line,
                    end_line=end_line,
                )

                if isinstance(content, ToolError):
                    return content

                contexts.append(
                    GrepContext(
                        path=content.path,
                        content=content.content,
                        start_line=content.start_line,
                        end_line=content.end_line,
                        matches=range_matches,
                    )
                )

        return contexts

    def tree(
        self,
        path: WorkspacePath = "",
        max_depth: MaxDepth = 3,
    ) -> TreeNode | ToolError:
        """
        Return a directory tree for workspace exploration.

        This tool helps understand the project structure
        before reading individual files.

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

    def recent_files(
        self,
        path: WorkspacePath = "",
        max_results: MaxResults = 20,
    ) -> list[FileInfo] | ToolError:
        """
        Return recently modified files in the workspace.

        This tool orders files by filesystem modification time,
        with the most recently modified files first.

        Args:
            path:
                Directory path relative to the workspace root.
                Search starts from this directory.

            max_results:
                Maximum number of files to return.

        Returns:
            A list of FileInfo objects ordered from most recently
            modified to least recently modified.

        ToolError when:
            - the path is outside the workspace
            - max_results is invalid
            - the directory cannot be accessed
            - another filesystem error occurs

        Notes:
            - Search is recursive.
            - Excluded directories are skipped.
            - Modification time is based on filesystem metadata.
            - This tool does not determine whether a file is tracked by Git.
        """

        if max_results < 1:
            return ToolError(
                kind=ToolErrorKind.INVALID_ARGUMENT,
                msg="`max_results` must be greater than or equal to 1.",
                retry=False,
            )

        root = self._resolve(path)
        if isinstance(root, ToolError):
            return root

        result: list[tuple[float, FileInfo]] = []

        try:
            for current, dirs, files in os.walk(root):
                dirs[:] = [d for d in dirs if not self._excluded(Path(current) / d)]

                for file_name in files:
                    file_path = Path(current) / file_name

                    if self._excluded(file_path):
                        continue

                    try:
                        mtime = file_path.stat().st_mtime
                        info = FileInfo.from_absolute_path(
                            file_path,
                            self.workspace,
                        )
                    except OSError:
                        continue

                    result.append((mtime, info))

        except OSError as exc:
            return ToolError(
                kind=ToolErrorKind.UNKNOWN,
                msg=str(exc),
                retry=False,
            )

        result.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [info for _, info in result[:max_results]]
