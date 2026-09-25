from typing import Sequence

from pydantic import AwareDatetime, BaseModel, Field, JsonValue

from pytoy_llm.tools.workspace_explorer.semantic_types import WorkspacePath

from .semantic_types import IdeaSpacePath


class LocalLinkModel(BaseModel, frozen=True):
    """A link from an IdeaNote to a local workspace file."""

    path: IdeaSpacePath = Field(
        description="Path of the source IdeaNote, relative to the IdeaSpace root.",
        examples=["knowledge/python.md"],
    )
    reference_path: WorkspacePath = Field(
        description="Path of the referenced file, relative to the workspace root.",
        examples=["src/pytoy_llm/idea/note.py"],
    )


class RemoteLinkModel(BaseModel, frozen=True):
    """A link from an IdeaNote to a remote resource."""

    path: IdeaSpacePath = Field(
        description="Path of the source IdeaNote, relative to the IdeaSpace root.",
        examples=["knowledge/python.md"],
    )
    uri: str = Field(
        description="URI of the referenced remote resource.",
        examples=["https://example.com/reference"],
    )


class IdeaNoteLinkModel(BaseModel, frozen=True):
    """A link from one IdeaNote to another IdeaNote."""

    path: IdeaSpacePath = Field(
        description="Path of the source IdeaNote, relative to the IdeaSpace root.",
        examples=["knowledge/python.md"],
    )
    target_path: IdeaSpacePath = Field(
        description="Path of the target IdeaNote, relative to the IdeaSpace root.",
        examples=["knowledge/programming.md"],
    )


class UnresolvedLinkModel(BaseModel, frozen=True):
    """A link which is not available."""

    path: IdeaSpacePath = Field(
        description="Path of the source IdeaNote, relative to the IdeaSpace root.",
        examples=["knowledge/python.md"],
    )
    uri: str = Field(
        description="Uri of `links` which is not available.",
        examples=["../../../src/pytoy_llm/idea/note.py"],
    )
    reason: str | None = Field(description="A reason why this link is unavailable, if given.")


class IdeaNoteModel(BaseModel, frozen=True):
    """A knowledge note exposed to an LLM."""

    path: IdeaSpacePath = Field(
        description="Path of the IdeaNote, relative to the IdeaSpace root.",
        examples=["knowledge/python.md"],
    )

    modified_at: AwareDatetime = Field(
        description="The last modification time of the physical note file."
    )

    body: str = Field(
        description="Markdown body of the IdeaNote, excluding its frontmatter.",
    )
    metadata: dict[str, JsonValue] | None = Field(
        default=None,
        description="Frontmatter metadata of the IdeaNote, if present.",
    )
    note_links: Sequence[IdeaNoteLinkModel] = Field(
        default=(),
        description="Links from this note to other IdeaNotes.",
    )
    local_links: Sequence[LocalLinkModel] = Field(
        default=(),
        description="Links from this note to local workspace files.",
    )
    remote_links: Sequence[RemoteLinkModel] = Field(
        default=(),
        description="Links from this note to remote resources.",
    )

    unresolved_links: Sequence[UnresolvedLinkModel] = Field(
        default=(),
        description="Links which is unavailable.",
    )


class IdeaSpaceConventionModel(BaseModel, frozen=True):
    """A convention that defines how IdeaNotes within an IdeaSpace are organized."""

    applied_to: IdeaSpacePath = Field(
        description=(
            "Path within the IdeaSpace, relative to the IdeaSpace root. "
            "The convention applies to all files under this path."
        ),
        examples=[".", "./knowledge"],
    )
    note: IdeaNoteModel = Field(description="IdeaNote that represents the convention.")


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

    tool_meta: IdeaSpaceToolMetaModel = Field(
        description="Metadata describing the current tool-related state of the IdeaSpace."
    )
