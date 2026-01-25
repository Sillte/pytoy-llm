from pathlib import Path
from pydantic import BaseModel, Field
from typing import Annotated, Self
from collections.abc import Sequence
from datetime import datetime
import time
from pytoy_llm.materials.core import StructuredText
from pytoy_llm.materials.core import ModelSectionData, TextSectionData


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

    @property
    def structured_text(self) -> StructuredText:
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


class TextFileData(BaseModel, frozen=True):
    """Represents a single text file in the workspace."""
    path: Annotated[Path, Field(description="Absolute path to the file")]
    timestamp: Annotated[float, Field(description="Last modification time of the file (epoch seconds)")]
    location: Annotated[Sequence[str], Field(description="File path relative to the workspace root, as sequence of directories")]
    text: Annotated[str, Field(description="The contents of file.")]

    @classmethod
    def from_text_file(cls, text_file: TextFile) -> Self:
        return cls(path=text_file.path,
                    timestamp=text_file.timestamp,
                    location=text_file.location,
                    text=text_file.text)
        
class TextFileBundleData(BaseModel, frozen=True):
    root_location: Annotated[Sequence[str], Field(description="Workspace-relative root location for this set of documents")]
    text_data: Annotated[Sequence[TextFileData], Field(description="List of text data contained in this bundle")]

    @classmethod
    def from_bundle(cls, bundle: "TextFileBundle") -> Self:
        text_data = [TextFileData.from_text_file(item) for item in bundle.text_files]
        return cls(root_location=bundle.root_location,
                   text_data=text_data)


class TextFileBundle(BaseModel, frozen=True):
    """Container holding multiple text files under a common root."""
    root_location: Annotated[Sequence[str], Field(description="Workspace-relative root location for this set of documents")]
    text_files: Annotated[Sequence[TextFile], Field(description="List of text files contained in this bundle")]

    @property
    def bundle_kind(self):
        return "TextFileBundle"
    
    @property
    def text_section_data(self) -> TextSectionData:
        bundle_kind = self.bundle_kind
        structured_text =  self.structured_text
        description = self.description
        return TextSectionData(bundle_kind=bundle_kind,
                                         structured_text=structured_text,
                                         description=description)

    @property
    def model_section_data(self) -> ModelSectionData:
        # Note: `TextFileBundleData` requires a memory space of 
        # text data. 
        # If we would like to use the big data, 
        # `chunk` or `iter` iteration is necessary regarding `data`.
        return ModelSectionData(bundle_kind=self.bundle_kind,
                                description=self.description,
                                data=[TextFileBundleData.from_bundle(self)])
        
    @property
    def description(self) -> str:
        description = ("This section contains of files.\n"
                       "`location` corresponds to the place of file.\n"
                       "timestamp: corresponds to the latest modified time."
                        )
        return description

    @property
    def structured_text(self) -> str:
        """Returns a structured text representation of the documents for LLM consumption."""
        timestamp = time.time()
        bodies = [elem.structured_text for elem in self.text_files]
        return (
            f"===Documents===\n"
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
    ) -> "TextFileBundle":
        """Creates a DocumentsContainer with relative root location based on the current working directory."""
        rel = cwd.relative_to(workspace)
        root_location = rel.parts
        return cls(root_location=root_location, text_files=text_files)
    

class TextFileBundleQuery(BaseModel, frozen=True):
    """Currently, no-queries, however, we can set these parameters later."""
