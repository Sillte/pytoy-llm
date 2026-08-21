from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from pytoy_llm.tools.workspace_explorer.semantic_types import WorkspacePath


class FileInfo(BaseModel, frozen=True):
    """Metadata of a workspace file or directory."""

    path: WorkspacePath = Field(description="Relative path from the workspace root.")
    size: int = Field(description="File size in bytes. Directories may report platform-dependent values.")

    modified: datetime = Field(description="Last modification timestamp.")

    @classmethod
    def from_relative_path(cls, relative_path: WorkspacePath, workspace_root: Path) -> "FileInfo":
        abs_path = workspace_root / relative_path
        stat = abs_path.stat()
        return cls(
            path=abs_path.relative_to(workspace_root).as_posix(),
            size=stat.st_size,
            modified=datetime.fromtimestamp(stat.st_mtime),
        )

    @classmethod
    def from_absolute_path(cls, absolute_path: Path, workspace_root: Path) -> "FileInfo":
        absolute_path = Path(absolute_path)
        stat = absolute_path.stat()
        return cls(
            path=absolute_path.relative_to(workspace_root).as_posix(),
            size=stat.st_size,
            modified=datetime.fromtimestamp(stat.st_mtime),
        )


class DirectoryInfo(BaseModel, frozen=True):
    path: WorkspacePath = Field(description="Relative path from the workspace root.")
    modified: datetime = Field(description="Last modification timestamp.")

    @classmethod
    def from_relative_path(cls, relative_path: WorkspacePath, workspace_root: Path) -> "DirectoryInfo":
        abs_path = workspace_root / relative_path
        stat = abs_path.stat()
        return cls(
            path=abs_path.relative_to(workspace_root).as_posix(),
            modified=datetime.fromtimestamp(stat.st_mtime),
        )

    @classmethod
    def from_absolute_path(cls, absolute_path: Path, workspace_root: Path) -> "DirectoryInfo":
        absolute_path = Path(absolute_path)
        stat = absolute_path.stat()
        return cls(
            path=absolute_path.relative_to(workspace_root).as_posix(),
            modified=datetime.fromtimestamp(stat.st_mtime),
        )


class FileContent(BaseModel, frozen=True):
    """The content of file."""

    path: WorkspacePath = Field(description="Relative path from the workspace root.")
    content: str = Field(description="The content of the file.")


class FilePartContent(BaseModel, frozen=True):
    """The partial content of file, not the entire content of the file."""

    path: WorkspacePath = Field(description="Relative path from the workspace root.")
    content: str = Field(description="The content of the file.")
    start_line: int = Field(description="The start line number of the content, inclusive.")
    end_line: int = Field(description="The end line number of the content, exclusive.")


class GrepMatch(BaseModel, frozen=True):
    """One grep match."""

    path: WorkspacePath = Field(description="Relative path of the matched file.")

    line: int = Field(description="Zero-based line number.")

    column: int = Field(description="Zero-based column index.")

    text: str = Field(description="Entire line containing the match.")


class GrepContext(BaseModel, frozen=True):
    """A portion of a file containing one or more grep matches."""

    path: WorkspacePath = Field(description="Relative path of the file.")

    content: str = Field(description="The portion of the file surrounding the grep matches.")

    start_line: int = Field(ge=0, description="Zero-based start line of the content, inclusive.")

    end_line: int = Field(ge=0, description="Zero-based end line of the content, exclusive.")

    matches: list[GrepMatch] = Field(description="Grep matches contained in this context.")


class TreeNode(BaseModel, frozen=True):
    """Directory tree node."""

    name: str = Field(description="File or directory name.")

    path: WorkspacePath = Field(description="Relative path from the workspace root.")

    is_dir: bool = Field(description="Whether this node is a directory.")

    children: list["TreeNode"] = Field(default_factory=list, description="Immediate child nodes.")
