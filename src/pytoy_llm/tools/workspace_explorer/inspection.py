from pathlib import Path
from typing import Self, Sequence

from pytoy_llm.tools.errors import ToolError, ToolErrorKind
from pytoy_llm.tools.workspace_explorer.models import (
    FileContent,
    FilePartContent,
    PartialFilesReadResult,
    WorkspaceAccess,
)
from pytoy_llm.tools.workspace_explorer.semantic_types import LineNumber, MaxBytes, WorkspacePath


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
        return [self.read_text_file, self.read_text_files, self.read_text_file_range]

    def read_text_file(
        self,
        path: WorkspacePath,
        max_lines: int | None = 20,
        max_bytes: MaxBytes | None = 1_000_000,
    ) -> FilePartContent | FileContent | ToolError:
        """
        Read the beginning of a text file, or the entire file when requested.

        By default, only the first `max_lines` lines are returned to keep
        large files bounded. If the file is longer than `max_lines`, the
        result is `FilePartContent`, not the complete file.

        Set `max_lines` to be `null` to request the entire file.

        Use `read_text_file_range` when you need a specific line range rather
        than the beginning of the file.

        Args:
            path:
                Path relative to the workspace root.

            max_lines:
                Maximum number of lines to read.
                - integer: return at most this many lines.
                - null: read the entire file.

            max_bytes:
                Acceptable maximum size of the file in bytes.
                - integer: If the file exceeds `max_bytes`, return `ToolError`.
                - null: No file-size limit is applied.

        Returns:
            FileContent:
                The complete file contents.

            FilePartContent:
                The first `max_lines` lines of the file.

            ToolError:
                When the file does not exist, is outside the workspace,
                cannot be read, or the argument is invalid.

        Notes:
            - This tool does NOT accept `start_line` nor `end_line` unlike `read_text_file_range`.
        """
        if max_lines is not None and max_lines < 1:
            return ToolError(
                kind=ToolErrorKind.INVALID_ARGUMENT,
                msg="`max_lines` must be greater than or equal to 1.",
                retry=False,
            )
        if max_bytes is not None and max_bytes < 1:
            return ToolError(
                kind=ToolErrorKind.INVALID_ARGUMENT,
                msg="`max_bytes` must be greater than or equal to 1.",
                retry=False,
            )

        text = self.access.read_text(path, max_bytes=max_bytes)
        if isinstance(text, ToolError):
            return text
        if max_lines is None:
            return FileContent(path=path, content=text)
        lines = text.splitlines(keepends=True)
        if len(lines) <= max_lines:
            return FileContent(path=path, content=text)
        return FilePartContent(
            path=path, content="".join(lines[:max_lines]), start_line=0, end_line=max_lines
        )

    def read_text_files(
        self,
        paths: Sequence[WorkspacePath],
        max_lines: int | None = 10,
        max_bytes: MaxBytes | None = 1_000_000,
    ) -> list[FileContent | FilePartContent] | PartialFilesReadResult | ToolError:
        """
        Read the beginning of multiple text files, or the entire files when requested.

        The same `max_lines` limit is applied independently to each file.
        If a file is longer than `max_lines`, its result is `FilePartContent`
        rather than the complete file.

        Set `max_lines=null` to request the complete contents of every file.

        Args:
            paths:
                File paths relative to the workspace root.

            max_lines:
                - integer: return at most this many lines.
                - null: read the entire file.

            max_bytes:
                Acceptable maximum size of the file in bytes.
                - integer: If the file exceeds `max_bytes`, return `ToolError`.
                - null: No file-size limit is applied.

        Returns:
            A list containing FileContent or FilePartContent for each
            requested file, in the same order as `paths`.

            ToolError when:
                - Entire operation is invalid.

            PartialFilesReadResult when:
                - At least one file failed to be read. Successful and failed
                  file results are returned separately.

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
        if max_bytes is not None and max_bytes < 1:
            return ToolError(
                kind=ToolErrorKind.INVALID_ARGUMENT,
                msg="`max_bytes` must be greater than or equal to 1.",
                retry=False,
            )

        successes: list[FileContent | FilePartContent] = []
        failures: dict[WorkspacePath, ToolError] = {}

        for path in paths:
            result = self.read_text_file(
                path=path,
                max_lines=max_lines,
                max_bytes=max_bytes,
            )

            if isinstance(result, ToolError):
                failures[path] = result
            else:
                successes.append(result)

        if failures:
            return PartialFilesReadResult(successes=successes, failures=failures)
        else:
            return successes

    def read_text_file_range(
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

        Unlike `read_text_file`, this tool returns only the requested line range.

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
        """
        if end_line <= start_line:
            return ToolError(
                kind=ToolErrorKind.INVALID_ARGUMENT,
                msg="end_line must be greater than start_line.",
                retry=False,
            )
        if start_line < 0:
            return ToolError(
                kind=ToolErrorKind.INVALID_ARGUMENT,
                msg="start_line must be a non-negative integer.",
            )
        text = self.access.read_text(path, max_bytes=None)
        if isinstance(text, ToolError):
            return text

        lines = text.splitlines(keepends=True)
        if len(lines) < end_line:
            return ToolError(
                kind=ToolErrorKind.INVALID_ARGUMENT,
                msg=f"`{end_line=}` is out of range; the file has {len(lines)} lines.",
            )
        return FilePartContent(
            path=path,
            content="".join(lines[start_line:end_line]),
            start_line=start_line,
            end_line=end_line,
        )
