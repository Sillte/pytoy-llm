from pydantic import BaseModel, Field
from typing import Sequence, Annotated
from pathlib import Path

# -------------------------
# LineRange
# -------------------------
class LineRange(BaseModel, frozen=True):
    """0-based, exclusive range [start, end).

    Example:
        LineRange(0, 1) -> Only line 0.
        LineRange(0, 0) -> Just before line 0 (Insertion Point)
    """
    start: Annotated[int, Field(description="Start line index (inclusive, 0-based)")]
    end: Annotated[int, Field(description="End line index (exclusive, 0-based)")]

    @property
    def count(self) -> int:
        """Number of lines in the range."""
        return self.end - self.start

# -------------------------
# AtomicChange
# -------------------------
class AtomicChange(BaseModel, frozen=True):
    """Represents an atomic text change in a file."""
    range: Annotated[LineRange, Field(description="Line range in the original file that is changed")]
    old_lines: Annotated[Sequence[str], Field(description="Original lines that were replaced or deleted")]
    new_lines: Annotated[Sequence[str], Field(description="New lines that were added in place of old_lines")]

# -------------------------
# File Operations
# -------------------------
class FileAdd(BaseModel, frozen=True):
    """Represents a file creation."""
    path: Annotated[Path, Field(description="Path to the new file")]
    lines: Annotated[Sequence[str], Field(description="Content lines of the new file")]

class FileDelete(BaseModel, frozen=True):
    """Represents a file deletion."""
    path: Annotated[Path, Field(description="Path to the deleted file")]
    old_lines: Annotated[Sequence[str], Field(description="Content lines of the deleted file")]

class FileModify(BaseModel, frozen=True):
    """Represents modifications to an existing file."""
    path: Annotated[Path, Field(description="Path to the modified file")]
    atomic_changes: Annotated[Sequence[AtomicChange], Field(description="Sequence of atomic changes applied to the file")]

FileOperation = FileAdd | FileDelete | FileModify

# -------------------------
# FileDiff
# -------------------------
class FileDiff(BaseModel, frozen=True):
    """Represents a single file change in a diff."""
    operation: Annotated[FileOperation, Field(description="The type of file operation (add, delete, modify)")]
    timestamp: Annotated[float, Field(description="Time when the change occurred (commit time or file mtime)")]
    location: Annotated[Sequence[str], Field(description="Relative path from the workspace root as sequence of directories")]

    @property
    def path(self) -> Path:
        """Convenience accessor for the path of the file affected by this diff."""
        return self.operation.path

# -------------------------
# DiffContainer
# -------------------------
class DiffContainer(BaseModel, frozen=True):
    """Container for multiple file diffs, with a root location context."""
    root_location: Annotated[Sequence[str], Field(description="Workspace-relative root location for this diff set")]
    file_diffs: Annotated[Sequence[FileDiff], Field(description="List of file diffs contained in this container")]
