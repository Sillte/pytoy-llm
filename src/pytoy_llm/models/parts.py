from typing import Any, Literal

from pydantic import BaseModel, Field, TypeAdapter

type Role = Literal["user", "assistant"]


class BasePart(BaseModel, frozen=True):
    """One chunk of an LLM message represented by the domain model."""


class ContentPart(BasePart, frozen=True):
    """Content expressed by a participant in the message."""

    role: Role = Field(description="The role of the participant who expressed the content.")
    content: Any = Field(description="The content expressed by the participant.")


class TextPart(ContentPart, frozen=True):
    """Text content expressed by a participant in the message."""

    content: str = Field(description="The text expressed by the participant.")


class SystemPromptHistoryPart(BasePart, frozen=True):
    """A system prompt preserved as part of the interaction history."""

    content: str = Field(
        description="The system-level instruction or prompt preserved in the history."
    )


class ToolCallPart(BasePart, frozen=True):
    """A request made by the LLM to invoke a tool."""

    tool_name: str = Field(description="The name of the tool requested by the LLM.")
    call_id: str = Field(
        description="The identifier used to associate the tool call with its result."
    )
    args: str | dict[str, Any] | None = Field(
        default=None, description="The arguments supplied to the tool."
    )


class ToolResultPart(BasePart, frozen=True):
    """The result returned from a tool invocation."""

    tool_name: str = Field(
        description="The name of the tool that produced the result, when available."
    )
    call_id: str = Field(description="The identifier of the tool call that produced this result.")
    result: Any = Field(description="The result content returned by the tool.")


class OpaquePart(BasePart, frozen=True):
    """Data preserved in the interaction without assigning domain-specific semantics."""

    value: Any = Field(
        description="The raw value whose internal meaning is not modeled by this domain."
    )


Part = TextPart | ContentPart | ToolCallPart | ToolResultPart | SystemPromptHistoryPart | OpaquePart

PartAdapter = TypeAdapter(Part)
