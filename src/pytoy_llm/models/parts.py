from typing import Any, Literal

from pydantic import BaseModel, TypeAdapter

type Role = Literal["system", "user", "assistant"]


class BasePart(BaseModel, frozen=True): ...


class TextPart(BasePart, frozen=True):
    role: Role
    content: str


class AnyContentPart(BasePart, frozen=True):
    role: Role
    content: Any


class ToolCallRequestPart(BasePart, frozen=True):
    tool_name: str
    call_id: str
    args: str | dict[str, Any] | None = None


class ToolResultPart(BasePart, frozen=True):
    call_id: str
    content: Any
    tool_name: str | None = None


class OpaquePart(BasePart, frozen=True):
    value: Any


Part = TextPart | AnyContentPart | ToolCallRequestPart | ToolResultPart | OpaquePart
PartAdapter = TypeAdapter(Part)
