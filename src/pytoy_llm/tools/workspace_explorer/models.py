from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Self, Sequence

from pydantic import AwareDatetime, BaseModel, Field

from pytoy_llm.foundation.paths import PathGatherer
from pytoy_llm.tools.errors import ToolError, ToolErrorKind
from pytoy_llm.tools.workspace_explorer.semantic_types import (
    FileGlob,
    LineNumber,
    MaxBytes,
    WorkspaceDirectoryPath,
    WorkspaceDirectoryPivot,
    WorkspaceFilePath,
)


class FileInfo(BaseModel, frozen=True):
    """Metadata of a file inside the workspace."""

    workspace_file_path: WorkspaceFilePath = Field(
        description="Relative path from the workspace root."
    )
    size: int = Field(description="File size in bytes.")

    modified_at: AwareDatetime = Field(description="Last modification timestamp.")

    @classmethod
    def from_relative_path(cls, relative_path: WorkspaceFilePath, workspace_root: Path) -> Self:
        abs_path = workspace_root / relative_path
        stat = abs_path.stat()
        return cls(
            workspace_file_path=abs_path.relative_to(workspace_root).as_posix(),
            size=stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
        )

    @classmethod
    def from_absolute_path(cls, absolute_path: Path, workspace_root: Path) -> Self:
        absolute_path = Path(absolute_path)
        stat = absolute_path.stat()
        return cls(
            workspace_file_path=absolute_path.relative_to(workspace_root).as_posix(),
            size=stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
        )


class DirectoryInfo(BaseModel, frozen=True):
    """Metadata of a directory inside the workspace."""

    workspace_directory_path: WorkspaceDirectoryPath = Field(
        description="Relative path from the workspace root."
    )
    modified_at: AwareDatetime = Field(description="Last modification timestamp.")

    @classmethod
    def from_relative_path(
        cls, relative_path: WorkspaceDirectoryPath, workspace_root: Path
    ) -> "DirectoryInfo":
        abs_path = workspace_root / relative_path
        stat = abs_path.stat()
        return cls(
            workspace_directory_path=abs_path.relative_to(workspace_root).as_posix(),
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
        )

    @classmethod
    def from_absolute_path(cls, absolute_path: Path, workspace_root: Path) -> "DirectoryInfo":
        absolute_path = Path(absolute_path)
        stat = absolute_path.stat()
        return cls(
            workspace_directory_path=absolute_path.relative_to(workspace_root).as_posix(),
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
        )


class FileContent(BaseModel, frozen=True):
    """The content of file."""

    workspace_file_path: WorkspaceFilePath = Field(
        description="Relative path from the workspace root."
    )
    content: str = Field(description="The content of the file.")


class FilePartContent(BaseModel, frozen=True):
    """A portion of a file."""

    workspace_file_path: WorkspaceFilePath = Field(
        description="Relative path from the workspace root."
    )
    content: str = Field(description="The content of the file.")
    start_line: LineNumber = Field(description="The start line number of the content, inclusive.")
    end_line: LineNumber = Field(description="The end line number of the content, exclusive.")


class PartialFilesReadResult(BaseModel, frozen=True):
    """The result of a batch file-read operation where one or more files failed."""

    status: Literal["partial-success"] = Field(
        default="partial-success",
        description=(
            "Indicates that one or more requested files could not be read; "
            "successful and failed results are returned separately."
        ),
    )
    successes: list[FilePartContent | FileContent] = Field(
        description="Success result of file-read."
    )
    failures: dict[WorkspaceFilePath, ToolError] = Field(description="Failure errors of file-read.")


class GrepMatch(BaseModel, frozen=True):
    """One grep match."""

    workspace_file_path: WorkspaceFilePath = Field(description="Relative path of the matched file.")

    line: LineNumber = Field(description="Zero-based line number.")

    column: int = Field(description="Zero-based column index.")

    text: str = Field(description="Entire line containing the match.")


class GrepMatchContext(BaseModel, frozen=True):
    """A portion of a file containing one or more grep matches."""

    workspace_file_path: WorkspaceFilePath = Field(description="Relative path of the file.")

    content: str = Field(description="The portion of the file surrounding the grep matches.")

    start_line: int = Field(ge=0, description="Zero-based start line of the content, inclusive.")

    end_line: int = Field(ge=0, description="Zero-based end line of the content, exclusive.")

    matches: list[GrepMatch] = Field(description="Grep matches contained in this context.")


DEFAULT_EXCLUDED_PATTERNS = frozenset(
    {
        ".pytoy",
        ".venv",
        "venv",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        "*.egg-info",
    }
)


@dataclass(frozen=True)
class WorkspaceAccess:
    workspace: Path
    excludes: frozenset[str] = DEFAULT_EXCLUDED_PATTERNS
    text_encodings: tuple[str, ...] = ("utf-8-sig", "utf-8", "cp932")

    @classmethod
    def from_any(cls, workspace: Path | str, excludes: frozenset[str] | None = None) -> Self:
        if excludes is None:
            excludes = DEFAULT_EXCLUDED_PATTERNS
        return cls(workspace=Path(workspace).resolve(), excludes=excludes)

    def resolve(self, path: WorkspaceDirectoryPath | WorkspaceFilePath) -> Path | ToolError:
        abs_path = (self.workspace / path).resolve(strict=False)
        if not abs_path.is_relative_to(self.workspace):
            return ToolError(
                kind=ToolErrorKind.INVALID_ARGUMENT,
                msg=f"{path=} is outside workspace",
                retry=False,
            )
        return abs_path

    def _to_workspace_file_path(self, path: Path) -> WorkspaceFilePath:
        return path.relative_to(self.workspace).as_posix()

    def is_within_workspace(self, path: Path) -> bool:
        return path.resolve(strict=False).is_relative_to(self.workspace)

    def read_text(
        self, path: WorkspaceFilePath, max_bytes: MaxBytes | None = None
    ) -> str | ToolError:
        abs_path = self.resolve(path)
        if isinstance(abs_path, ToolError):
            return abs_path

        try:
            if not abs_path.exists():
                return ToolError(
                    kind=ToolErrorKind.NOT_FOUND,
                    msg=f"{path=} does not exist.",
                    retry=False,
                )

            if not abs_path.is_file():
                return ToolError(
                    kind=ToolErrorKind.INVALID_ARGUMENT,
                    msg=f"{path=} must be a file.",
                    retry=False,
                )

            if max_bytes is not None and abs_path.stat().st_size > max_bytes:
                return ToolError(
                    kind=ToolErrorKind.RESOURCE_LIMIT,
                    msg=f"{path=} exceeds the {max_bytes} byte limit.",
                    retry=False,
                )

            for encoding in self.text_encodings:
                try:
                    return abs_path.read_text(encoding=encoding)
                except UnicodeDecodeError:
                    continue

            return ToolError(
                kind=ToolErrorKind.DECODE_ERROR,
                msg=f"Could not decode {path=} using configured text encodings.",
                retry=False,
            )

        except PermissionError as exc:
            return ToolError(
                kind=ToolErrorKind.PERMISSION_DENIED,
                msg=f"Could not read {path=}: {exc}",
                retry=False,
            )
        except OSError as exc:
            return ToolError(
                kind=ToolErrorKind.IO_ERROR,
                msg=f"Could not read {path=}: {exc}",
            )
        except Exception as exc:
            return ToolError(
                kind=ToolErrorKind.UNKNOWN,
                msg=f"Could not read {path=}: {exc}",
            )

    def gather_file_paths(
        self,
        collection_root: WorkspaceDirectoryPivot,
        file_globs: Sequence[FileGlob],
        max_file_bytes: MaxBytes | None,
    ) -> Sequence[WorkspaceFilePath] | ToolError:
        root = self.resolve(collection_root)
        if isinstance(root, ToolError):
            return root
        if not root.exists():
            return ToolError(
                kind=ToolErrorKind.NOT_FOUND,
                msg=f"`{collection_root=}` does not exist.",
                retry=None,
            )
        if not root.is_dir():
            return ToolError(
                kind=ToolErrorKind.INVALID_ARGUMENT,
                msg=f"`{collection_root=}` must be a directory.",
                retry=None,
            )

        def _is_target_path(path: Path) -> bool:
            try:
                return self.is_within_workspace(path) and (
                    max_file_bytes is None or path.stat().st_size <= max_file_bytes
                )
            except OSError:
                return False

        return tuple(
            self._to_workspace_file_path(path)
            for path in PathGatherer().gather(
                root=root,
                patterns=file_globs,
                excludes=self.excludes,
                target="file",
            )
            if _is_target_path(path)
        )
