from typing import Sequence

from pydantic import BaseModel, Field, JsonValue

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


class IdeaNoteModel(BaseModel, frozen=True):
    """A knowledge note exposed to an LLM."""

    path: IdeaSpacePath = Field(
        description="Path of the IdeaNote, relative to the IdeaSpace root.",
        examples=["knowledge/python.md"],
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
