from pydantic import BaseModel, Field
from typing import Sequence, Annotated, Literal, assert_never
from pathlib import Path
from pytoy_llm.materials.core import ModelSectionData, TextSectionData, StructuredText


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
    
    @property
    def structured_text(self) -> str:
        return f"<atomic-change lines={self.range.count} range={self.range.start}-{self.range.end}>"

# -------------------------
# File Operations
# -------------------------
class FileAdd(BaseModel, frozen=True):
    """Represents a file creation."""
    path: Annotated[Path, Field(description="Relative path to the new file from the git root.")]
    lines: Annotated[Sequence[str], Field(description="Content lines of the new file")]
    op_type: Literal["Add"] = "Add"
    

class FileDelete(BaseModel, frozen=True):
    """Represents a file deletion."""
    path: Annotated[Path, Field(description="Relative path to the deleted file from the git root.")]
    old_lines: Annotated[Sequence[str], Field(description="Content lines of the deleted file")]
    op_type: Literal["Delete"] = "Delete"

class FileModify(BaseModel, frozen=True):
    """Represents modifications to an existing file."""
    path: Annotated[Path, Field(description="Relative path to the modified file from the git root.")]
    atomic_changes: Annotated[Sequence[AtomicChange], Field(description="Sequence of atomic changes applied to the file")]
    op_type: Literal["Modify"] = "Modify"


type FileOperation = FileAdd | FileDelete | FileModify

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
    
    @property
    def structured_text(self) -> StructuredText:
        header: str = ("<header>\n"
                  f"<timestamp>{self.timestamp}</timestamp>\n"
                  f"<location>{self.location}</location>\n"
                  "/<header>\n")
        op = self.operation
        match op.op_type:
            case "Add":
                content = f"<add lines={len(op.lines)}>"
            case "Delete":
                content = f"<delete lines={len(op.old_lines)}>"
            case "Modify":
                atomics = "\n".join([item.structured_text for item in op.atomic_changes])
                content = f"<modify changes={atomics}>"
            case _:
                assert_never(op.op_type)
        body = f"<operation>{content}</operation>"
        return f"<diff>\n{header}\n{body}\n</diff>"


class DiffBundle(BaseModel, frozen=True):
    """Bundle for multiple file diffs, with a root location context."""
    root_location: Annotated[Sequence[str], Field(description="Workspace-relative root location for this diff set")]
    file_diffs: Annotated[Sequence[FileDiff], Field(description="List of file diffs contained in this container")]

    @property
    def bundle_kind(self) -> str:
        return "DiffBundle"

    @property
    def text_section_data(self) -> TextSectionData:
        """LLM-friendly structured text for the diff bundle."""
        import time
        timestamp = time.time()
        body = "\n\n".join([fd.structured_text for fd in self.file_diffs])
        
        structured_text = (
            f"===DiffBundle===\n"
            f"- Root location: {self.root_location}\n"
            f"- Timestamp: {timestamp}\n"
            f"===Files===\n"
            f"{body}"
        )
        description = "Structured representation of file diffs in this bundle."
        return TextSectionData(
            bundle_kind=self.bundle_kind,
            structured_text=structured_text,
            description=description
        )

    @property
    def model_section_data(self) -> ModelSectionData:
        """Raw BaseModel representation of the diff bundle."""
        description = "JSON-like representation of file diffs for model consumption."
        return ModelSectionData(
            bundle_kind=self.bundle_kind,
            description=description,
            data=[self] 
        )


class GitDiffBundleQuery(BaseModel, frozen=True):
    from_rev: str | Literal["index"] | Literal["head"]= "head"
    to_rev: str |  Literal["working-tree"] | Literal["index"] | None = "working-tree"
