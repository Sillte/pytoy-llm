from pathlib import Path
from typing import Self, Sequence

from pytoy_llm.tools.errors import ToolError, ToolErrorKind
from pytoy_llm.tools.workspace_explorer.models import FileContent, FilePartContent, WorkspaceAccess
from pytoy_llm.tools.workspace_explorer.semantic_types import LineNumber, WorkspacePath


class WorkspaceInspection:
    """
    Provide safe workspace inspection tools for LLM agents.

    This class provides read-only operations for inspecting the contents
    of files within the workspace.
    It does not modify workspace files.

    Every path is interpreted relative to the workspace root.
    Files outside the workspace are never accessible.
    """

    def __init__(self, access: WorkspaceAccess) -> None:
        self.access = access
        self.workspace = access.workspace.resolve()

    @classmethod
    def from_any(cls, workspace: Path | str, excludes: frozenset[str] | None = None) -> Self:
        return cls(access=WorkspaceAccess.from_any(workspace=workspace, excludes=excludes))

    @property
    def tools(self):
        return [self.read_file, self.read_files, self.read_file_range]

    def read_file(
        self,
        path: WorkspacePath,
        max_lines: int | None = 20,
    ) -> FilePartContent | FileContent | ToolError:
        """
        Read the beginning of a text file, or the entire file when requested.

        By default, only the first `max_lines` lines are returned to keep
        large files bounded. If the file is longer than `max_lines`, the
        result is `FilePartContent`, not the complete file.

        Set `max_lines=None` to request the entire file.

        Use `read_file_range` when you need a specific line range rather
        than the beginning of the file.

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
            - This tool reads the file as UTF-8 text.
            - This tool does NOT accept `start_line` nor `end_line` unlike `read_file_range`.
        """
        if max_lines is not None and max_lines < 1:
            return ToolError(
                kind=ToolErrorKind.INVALID_ARGUMENT,
                msg="`max_lines` must be greater than or equal to 1.",
                retry=False,
            )

        abs_path = self.access.resolve(path)
        if isinstance(abs_path, ToolError):
            return abs_path

        try:
            if not abs_path.exists():
                return ToolError(
                    kind=ToolErrorKind.NOT_FOUND,
                    msg=f"{path=} does not exist.",
                    retry=False,
                )
            if max_lines is None:
                return FileContent(path=path, content=abs_path.read_text(encoding="utf8"))

            with abs_path.open("r", encoding="utf8") as f:
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
        except Exception as exc:
            return ToolError(
                kind=ToolErrorKind.UNKNOWN,
                msg=str(exc),
            )

    def read_files(
        self,
        paths: Sequence[WorkspacePath],
        max_lines: int | None = 10,
    ) -> list[FileContent | FilePartContent] | ToolError:
        """
        Read the beginning of multiple text files, or the entire files when requested.

        The same `max_lines` limit is applied independently to each file.
        If a file is longer than `max_lines`, its result is `FilePartContent`
        rather than the complete file.

        Set `max_lines=None` to request the complete contents of every file.

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
            - FileContent is returned when the complete file fits within `max_lines`.
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
        Read a specific range of lines from a text file.

        Use this tool when you need to inspect lines that are not necessarily
        at the beginning of the file, or when a previous search result
        identifies a specific location of interest.

        `start_line` is zero-based and inclusive.
        `end_line` is zero-based and exclusive.

        Unlike `read_file`, this tool reads only the requested line range.

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
        abs_path = self.access.resolve(path)
        if isinstance(abs_path, ToolError):
            return abs_path
        try:
            content = abs_path.read_text(encoding="utf8")
            lines = content.splitlines(keepends=True)

            if len(lines) < end_line:
                return ToolError(
                    kind=ToolErrorKind.INVALID_ARGUMENT,
                    msg=f"`{end_line=}` is out of range; the file has {len(lines)} lines.",
                )
            return FilePartContent(
                path=path, content="".join(lines[start_line:end_line]), start_line=start_line, end_line=end_line
            )
        except FileNotFoundError as exc:
            return ToolError(kind=ToolErrorKind.NOT_FOUND, msg=str(exc), retry=None)
        except OSError as exc:
            return ToolError(kind=ToolErrorKind.UNKNOWN, msg=str(exc))
