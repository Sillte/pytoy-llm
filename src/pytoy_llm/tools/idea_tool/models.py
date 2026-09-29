from pathlib import Path
from typing import Sequence

from pydantic import AwareDatetime, BaseModel, Field

from pytoy_llm.idea.domain.uri import Uri
from pytoy_llm.tools.workspace_explorer.semantic_types import WorkspaceFilePath

from .semantic_types import IdeaNoteMetadata, IdeaNotePath, IdeaSpacePath


class WorkspaceLinkModel(BaseModel, frozen=True):
    """A link from an IdeaNote to a local workspace file."""

    idea_note_path: IdeaNotePath = Field(
        description="Path of the source IdeaNote, relative to the IdeaSpace root.",
        examples=["knowledge/python.md"],
    )
    workspace_file_path: WorkspaceFilePath = Field(
        description="Path of the referenced file, relative to the workspace root.",
        examples=["src/pytoy_llm/idea/note.py"],
    )

    workspace_root_folder: Path = Field(description="Root folder of Workspace")

    @property
    def uri_string(self) -> str:
        return f"workspace:///{self.workspace_file_path.strip('/')}"


class RemoteLinkModel(BaseModel, frozen=True):
    """A link from an IdeaNote to a remote resource."""

    idea_note_path: IdeaNotePath = Field(
        description="Path of the source IdeaNote, relative to the IdeaSpace root.",
        examples=["knowledge/python.md"],
    )
    uri: Uri = Field(
        description="URI of the referenced remote resource.",
        examples=["https://example.com/reference"],
    )

    @property
    def uri_string(self) -> str:
        return str(self.uri)


class IdeaNoteLinkModel(BaseModel, frozen=True):
    """A link from one IdeaNote to another IdeaNote."""

    source_idea_note_path: IdeaNotePath = Field(
        description="Path of the source IdeaNote, relative to the IdeaSpace root.",
        examples=["knowledge/python.md"],
    )
    target_idea_note_path: IdeaNotePath = Field(
        description="Path of the target IdeaNote, relative to the IdeaSpace root.",
        examples=["knowledge/programming.md"],
    )

    idea_space_directory_folder: Path = Field(description="Root directory of IdeaSpace")

    @property
    def uri_string(self) -> str:
        return f"idea:///{self.target_idea_note_path}"


class UnresolvedLinkModel(BaseModel, frozen=True):
    """A link which is not available."""

    idea_note_path: IdeaNotePath = Field(
        description="Path of the source IdeaNote, relative to the IdeaSpace root.",
        examples=["knowledge/python.md"],
    )
    uri: Uri = Field(
        description="The original link target that could not be resolved.",
        examples=["../../non-existent-folder/src/note.md"],
    )
    reason: str | None = Field(description="A reason why this link is unavailable, if given.")


class IdeaNoteModel(BaseModel, frozen=True):
    """A knowledge note exposed to an LLM."""

    idea_note_path: IdeaNotePath = Field(
        description="Path of the IdeaNote, relative to the IdeaSpace root.",
        examples=["knowledge/python.md"],
    )

    modified_at: AwareDatetime = Field(
        description="The last modification time of the physical note file."
    )

    body: str = Field(
        description="Markdown body of the IdeaNote, excluding its frontmatter.",
    )

    metadata: IdeaNoteMetadata = Field(
        description="YAML frontmatter metadata of the IdeaNote.",
    )

    idea_note_links: Sequence[IdeaNoteLinkModel] = Field(
        default=(),
        description="Links from this note to other IdeaNotes.",
    )

    workspace_local_links: Sequence[WorkspaceLinkModel] = Field(
        default=(),
        description="Links from this note to local workspace files.",
    )

    remote_links: Sequence[RemoteLinkModel] = Field(
        default=(),
        description="Links from this note to remote resources.",
    )

    unresolved_links: Sequence[UnresolvedLinkModel] = Field(
        default=(),
        description="Links which are unavailable.",
    )


class IdeaSpaceConventionModel(BaseModel, frozen=True):
    """A convention that defines how IdeaNotes within an IdeaSpace are organized."""

    applied_to: IdeaSpacePath = Field(
        description=(
            "Path within the IdeaSpace, relative to the IdeaSpace root. "
            "The convention applies to all files under this path."
        ),
        examples=[".", "knowledge"],
    )

    idea_note: IdeaNoteModel = Field(description="IdeaNote that represents the convention.")


class IdeaSpaceToolMetaModel(BaseModel, frozen=True):
    """Metadata regarding tools operating on the IdeaSpace."""

    last_llm_started_at: AwareDatetime | None = Field(
        description="The time when the latest LLM interaction started.",
        default=None,
    )
    last_llm_finished_at: AwareDatetime | None = Field(
        description="The time when the latest LLM interaction finished.",
        default=None,
    )


class IdeaSpaceToolWorkingContextModel(BaseModel, frozen=True):
    """Context for working with an IdeaSpace."""

    idea_space_tool_meta: IdeaSpaceToolMetaModel = Field(
        description="Metadata describing the current tool-related state of the IdeaSpace."
    )
