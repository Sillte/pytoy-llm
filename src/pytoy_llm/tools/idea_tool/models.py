from typing import Sequence

from pydantic import AwareDatetime, BaseModel, Field

from pytoy_llm.idea.domain.uri import Uri

from .semantic_types import (
    IdeaNoteMetadata,
    IdeaNotePath,
    IdeaNoteReference,
    LocalFilePath,
    Namespace,
)


class LocalLinkModel(BaseModel, frozen=True):
    """A link from an IdeaNote to a file in a configured local root.

    `scheme` identifies the kind of local link, and `namespace` identifies
    the configured root to which `path` is relative.

    `path` is a decoded filesystem path, not a percent-encoded URI path.
    `uri_string` percent-encodes it when constructing a URI.
    """

    scheme: str = Field(
        description="The scheme identifying the kind of local link.",
        examples=["workspace"],
    )
    namespace: Namespace = Field(
        description="The configured namespace containing the target file. When `namespace` is represented in the form of URI, it corresponds to Authority.",
        examples=["example.com", "authority"],
    )
    path: LocalFilePath = Field(
        description="A path relative to the root configured for the namespace.",
        examples=["src/llm/idea.note.py", "pyproject.toml"],
    )

    source_idea_note_reference: IdeaNoteReference

    @property
    def uri_string(self) -> str:
        from urllib.parse import quote

        path = quote(self.path.strip("/"), safe="/")
        if self.namespace:
            return f"{self.scheme}://{self.namespace}/{path}"
        return f"{self.scheme}:{path}"


class ExternalLinkModel(BaseModel, frozen=True):
    """A link from an IdeaNote to an external URI."""

    source_idea_note_reference: IdeaNoteReference

    uri: Uri = Field(
        description="External URI referenced by the IdeaNote.",
        examples=["https://example.com/reference"],
    )

    @property
    def uri_string(self) -> str:
        return str(self.uri)


class IdeaNoteLinkModel(BaseModel, frozen=True):
    """A link from one IdeaNote to another IdeaNote."""

    source_idea_note_reference: IdeaNoteReference
    target_idea_note_reference: IdeaNoteReference

    @property
    def uri_string(self) -> str:
        ref = self.target_idea_note_reference
        if ref.namespace:
            return f"idea://{ref.namespace}/{ref.idea_note_path}"
        else:
            return f"idea:/{ref.idea_note_path}"


class UnresolvedLinkModel(BaseModel, frozen=True):
    """A link which is not available."""

    source_idea_note_reference: IdeaNoteReference
    uri: Uri = Field(
        description="The original link target that could not be resolved.",
        examples=["../../non-existent-folder/src/note.md"],
    )
    reason: str | None = Field(description="A reason why this link is unavailable, if given.")


class IdeaNoteModel(BaseModel, frozen=True):
    """A IdeaNote exposed to an LLM."""

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

    local_links: Sequence[LocalLinkModel] = Field(
        default=(),
        description="Links from this note to local files.",
    )

    external_links: Sequence[ExternalLinkModel] = Field(
        default=(),
        description="Links from this note to external URIs.",
    )

    unresolved_links: Sequence[UnresolvedLinkModel] = Field(
        default=(),
        description="Links which are unavailable.",
    )


class IdeaSpaceConventionModel(BaseModel, frozen=True):
    """An IdeaNote that serves as the convention for an IdeaSpace.

    Convention is an IdeaNote that defines local conventions
    for organizing, interpreting, creating, or maintaining IdeaNotes within an IdeaSpace.

    The convention is defined by a `.convention.md` note located directly
    in an IdeaSpace. Its scope and inheritance are determined by the
    IdeaSpace hierarchy, rather than by fields in this model.

    The contained IdeaNoteModel exposes the convention's path, body,
    metadata, and links using the same representation as other IdeaNotes.
    """

    idea_note: IdeaNoteModel = Field(
        description="The IdeaNote that serves as the IdeaSpace's convention."
    )


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
