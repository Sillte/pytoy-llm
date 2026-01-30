from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel, Field, BeforeValidator
from typing import Annotated, Self, Literal, Mapping, get_type_hints
from collections.abc import Sequence
from datetime import datetime
import time
import uuid
from pytoy_llm.materials.core import StructuredText
from pytoy_llm.materials.core import ModelSectionData, TextSectionData


TextFilePath = Annotated[
    Path,
    Field(description="Relative path"),
    BeforeValidator(lambda v: Path(v) if not Path(v).is_absolute() else ValueError("Path must be relative"))
]

TextFileID = Annotated[
    str,
    Field(description="Unique identifier."),
]


class TextFileLocator(BaseModel):
    """Represents a locator for a text file entity within a workspace."""
    
    id: TextFileID = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this file entity")
    
    path: Annotated[
        TextFilePath,
        Field(description="Path relative to the workspace root"),
        BeforeValidator(lambda v: Path(v) if not Path(v).is_absolute() else ValueError("Path must be relative"))
    ]
    
    timestamp: float = Field(..., description="Last modification time of the file (epoch seconds)")

    def absolute_path(self, workspace_root: Path) -> Path:
        """Return the absolute path within the workspace."""
        return workspace_root / self.path
    
    @classmethod
    def from_path(cls, abs_path: str | Path, workspace: str | Path) -> Self:
        relative = Path(abs_path).relative_to(workspace)
        return cls(timestamp=Path(abs_path).stat().st_mtime, path=relative)

EntryType = Annotated[
    Literal["All", "Summary"],
    Field(description="Specifies the type of the entry: 'All' means full text, 'Summary' means a condensed version")
]
    
    
class TextFileContent(BaseModel):
    """One instance of `text` file.
    """
    id: Annotated[TextFileID, Field(description="Unique identifier for this file entity")]
    
    entry: Annotated[
        EntryType,
        Field(description="Specifies the type of this entry (full text or summary)")
    ]
    
    body: Annotated[
        str,
        Field(description="The actual text content of this instance, either full or summarized depending on `entry`")
    ]
    
    @classmethod
    def from_locator(cls, locator: TextFileLocator, workspace: str | Path) -> Self:
        path = Path(workspace) / locator.path
        return cls(id=locator.id, body=path.read_text(encoding="utf8"), entry="All")

class TextFileCollection(BaseModel):
    locators: Annotated[
        Mapping[str, TextFileLocator],
        Field(description="Mapping of locators for the text files in this collection.")
    ]
    
    contents: Annotated[
        Mapping[str, TextFileContent],
        Field(description="Mapping of id to TextFileContent.")
    ]
    

    @property
    def structured_text(self) -> str:
        """Returns a structured representation of all instances suitable for LLM consumption.

        Includes:
        - Locator info (id, path, timestamp)
        - Entry info (id, entry_type, body)
        """
        lines = ["===Tag Description==="]

        # Locator の説明を追加
        lines.append("* Locator info:")
        lines.append(f"  - id: {TextFileLocator.model_fields['id'].description}")
        lines.append(f"  - path: {TextFileLocator.model_fields['path'].description}")
        lines.append(f"  - timestamp: {TextFileLocator.model_fields['timestamp'].description}")
        lines.append(f"  - readable_timestamp: human-readable timestamp")

        # EntryType の説明
        entry_desc = TextFileContent.model_fields['entry'].description
        lines.append(f"* entry_type: {entry_desc}")

        # Body の説明
        lines.append(f"* body: {TextFileContent.model_fields['body'].description}\n")
        lines.append("  - body is wrapped between <<<BEGIN>>> and <<<END>>>  \n")

        lines.append("===Instances===")
        for instance_id, instance in self.contents.items():
            locator = self.locators.get(instance_id)
            path_info = locator.path.as_posix() if locator else "UNKNOWN_PATH"
            timestamp_info = locator.timestamp if locator else 0.0
            readable_ts = datetime.fromtimestamp(timestamp_info).isoformat() if locator else "UNKNOWN_TIME"

            lines.append(
                f"<entry id={instance_id} entry_type={instance.entry} "
                f"path={path_info} timestamp={timestamp_info} readable_timestamp={readable_ts}>"
            )
            lines.append("<<<BEGIN>>>")
            lines.append(instance.body)
            lines.append("<<<END>>>")
            lines.append("</entry>\n")  # entry間に1行空行を入れる

        return "\n".join(lines)
    
    @property
    def instances(self) -> "Sequence[TextFileInstance]":
        ids = self.locators.keys()

        return [TextFileInstance(id=id_,
                                  locator=self.locators[id_],
                                  content=self.contents[id_])
                                  for id_ in ids]




class TextFileInstance(BaseModel):
    """Represents a single bundle data item linking a locator and an instance."""
    
    id: Annotated[
        TextFileID,
        Field(description="Unique identifier for this bundle data model")
    ]
    
    locator: Annotated[
        "TextFileLocator",
        Field(description="Locator object for the text file")
    ]
    
    content: Annotated[
        "TextFileContent",
        Field(description="Content of the text (full or summary)")
    ]


class TextFileBundle(BaseModel, frozen=True):
    """Container holding multiple text files under a common root."""
    collection: Annotated[TextFileCollection, Field(description="Collection of Texts")]
    description: Annotated[str, Field(description="Explanation about the collection.")] = "Collection of textfiles."
    bundle_kind: Annotated[str, Field(description="Type of `bundle`")] = "TextFileBundle"

    @property
    def text_section_data(self) -> TextSectionData:
        structured_text =  self.collection.structured_text
        bundle_kind = self.bundle_kind
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
                                data=self.collection.instances)
        
    

class TextFileBundleQuery(BaseModel, frozen=True):
    """Currently, no-queries, however, we can set these parameters later."""
