from pathlib import Path
from pydantic import BaseModel, Field
from typing import Annotated
from collections.abc import Sequence
from datetime import datetime
import time


class TextFile(BaseModel, frozen=True):
    """Represents a single text file in the workspace."""
    path: Annotated[Path, Field(description="Absolute path to the file")]
    timestamp: Annotated[float, Field(description="Last modification time of the file (epoch seconds)")]
    location: Annotated[Sequence[str], Field(description="File path relative to the workspace root, as sequence of directories")]

    @property
    def text(self) -> str:
        """Full content of the file as string."""
        return self.path.read_text(encoding="utf8")

    @classmethod
    def from_path(cls, path: Path, workspace: Path) -> "TextFile":
        """Create a TextFile instance from a path and workspace."""
        relative = path.relative_to(workspace)
        location = relative.parts
        return cls(path=path, timestamp=path.stat().st_mtime, location=location)

    def to_structure_text(self) -> str:
        """Return a structured representation suitable for LLM consumption."""
        return (
            f"<file>\n"
            f"<file-meta>\n"
            f"- location: {self.location}\n"
            f"- timestamp: {self.timestamp} / {datetime.fromtimestamp(self.timestamp)}\n"
            f"</file-meta>\n"
            f"<file-body>\n"
            f"<<<BEGIN FILE>>>\n"
            f"{self.text}\n"
            f"<<<END FILE>>>\n"
            f"</file-body>\n"
            f"</file>"
        )


class TextFilesContainer(BaseModel, frozen=True):
    """Container holding multiple text files under a common root."""
    root_location: Annotated[Sequence[str], Field(description="Workspace-relative root location for this set of documents")]
    text_files: Annotated[Sequence[TextFile], Field(description="List of text files contained in this container")]

    @property
    def text_result(self) -> str:
        """Returns a structured text representation of the documents for LLM consumption."""
        timestamp = time.time()
        bodies = [elem.to_structure_text() for elem in self.text_files]
        return (
            f"===Document===\n"
            f"- Root location: {self.root_location}\n"
            f"- Current Timestamp: {timestamp} / {datetime.fromtimestamp(timestamp)}\n"
            f"===Files===\n"
            f"{'\n\n'.join(bodies)}"
        )

    @classmethod
    def based_on_workspace(
        cls, 
        text_files: Sequence[TextFile], 
        cwd: Path, 
        workspace: Path
    ) -> "TextFilesContainer":
        """Creates a DocumentsContainer with relative root location based on the current working directory."""
        rel = cwd.relative_to(workspace)
        root_location = rel.parts
        return cls(root_location=root_location, text_files=text_files)
