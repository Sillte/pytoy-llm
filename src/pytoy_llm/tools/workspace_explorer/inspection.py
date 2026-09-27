from pathlib import Path
from typing import Self, Sequence

from pytoy_llm.tools.errors import ToolError, ToolErrorKind
from pytoy_llm.tools.workspace_explorer.models import (
    FileContent,
    FilePartContent,
    PartialFilesReadResult,
    WorkspaceAccess,
)
from pytoy_llm.tools.workspace_explorer.semantic_types import (
    LineNumber,
    MaxBytes,
    WorkspaceFilePath,
)


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
        return [
            self.workspace_read_text_file,
            self.workspace_read_text_files,
            self.workspace_read_text_file_range,
        ]

    def workspace_read_text_file(
        self,
        workspace_file_path: WorkspaceFilePath,
        max_lines: int | None = 20,
        max_bytes: MaxBytes | None = 1_000_000,
    ) -> FilePartContent | FileContent | ToolError:
        """
        Read the beginning of a text file, or the entire file when requested.

        By default, only the first `max_lines` lines are returned to keep
        large files bounded. If the file is longer than `max_lines`, the
        result is `FilePartContent`, not the complete file.

        Set `max_lines` to `null` to request the entire file.

        Args:
            workspace_file_path:
                File path relative to the workspace root.

            max_lines:
                Maximum number of lines to read.
                - integer: return at most this many lines.
                - null: read the entire file.

            max_bytes:
                Maximum acceptable file size in bytes.
                - integer: return `ToolError` if the file exceeds this limit.
                - null: no file-size limit is applied.

        Returns:
            FileContent when the complete file is returned.
            FilePartContent when only the first `max_lines` lines are returned.
            ToolError when the file cannot be read or the arguments are invalid.
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

        text = self.access.read_text(workspace_file_path, max_bytes=max_bytes)
        if isinstance(text, ToolError):
            return text
        if max_lines is None:
            return FileContent(workspace_file_path=workspace_file_path, content=text)
        lines = text.splitlines(keepends=True)
        if len(lines) <= max_lines:
            return FileContent(workspace_file_path=workspace_file_path, content=text)
        return FilePartContent(
            workspace_file_path=workspace_file_path,
            content="".join(lines[:max_lines]),
            start_line=0,
            end_line=max_lines,
        )

    def workspace_read_text_files(
        self,
        workspace_file_paths: Sequence[WorkspaceFilePath],
        max_lines: int | None = 10,
        max_bytes: MaxBytes | None = 1_000_000,
    ) -> list[FileContent | FilePartContent] | PartialFilesReadResult | ToolError:
        """Read the beginning of multiple text files, or the complete files when requested.

        The `max_lines` and `max_bytes` limits are applied independently to each file.
        If any file fails, successful and failed results are returned separately.

        Set `max_lines` to `null` to request the complete contents of every file.

        Args:
            workspace_file_paths:
                File paths relative to the workspace root.

            max_lines:
                - integer: return at most this many lines.
                - null: read the entire file.

            max_bytes:
                Maximum acceptable file size in bytes.
                - integer: If the file exceeds `max_bytes`, return `ToolError`.
                - null: No file-size limit is applied.

        Returns:
            A list of file contents when all requested files are read successfully.

            PartialFilesReadResult when one or more files cannot be read.

            ToolError when the entire operation is invalid.

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
        failures: dict[WorkspaceFilePath, ToolError] = {}

        for path in workspace_file_paths:
            result = self.workspace_read_text_file(
                workspace_file_path=path,
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

    def workspace_read_text_file_range(
        self,
        workspace_file_path: WorkspaceFilePath,
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

        Args:
            workspace_file_path:
                File path relative to the workspace root.

            start_line:
                Zero-based start line of the range, inclusive.

            end_line:
                Zero-based end line of the range, exclusive.

        Returns:
            FilePartContent containing the requested line range.
            ToolError if the file cannot be read or the range is invalid.
        """
        if end_line <= start_line:
            return ToolError(
                kind=ToolErrorKind.INVALID_ARGUMENT,
                msg="end_line must be greater than start_line.",
                retry=False,
            )
        text = self.access.read_text(workspace_file_path, max_bytes=None)
        if isinstance(text, ToolError):
            return text

        lines = text.splitlines(keepends=True)
        if len(lines) < end_line:
            return ToolError(
                kind=ToolErrorKind.INVALID_ARGUMENT,
                msg=f"`{end_line=}` is out of range; the file has {len(lines)} lines.",
            )
        return FilePartContent(
            workspace_file_path=workspace_file_path,
            content="".join(lines[start_line:end_line]),
            start_line=start_line,
            end_line=end_line,
        )
