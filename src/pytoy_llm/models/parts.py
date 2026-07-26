from typing import Any, Literal

from pydantic import BaseModel, TypeAdapter

type Role = Literal["system", "user", "assistant"]


class BasePart(BaseModel, frozen=True): ...


class TextPart(BasePart, frozen=True):
    role: Role
    content: str


class OpaquePart(BasePart, frozen=True):
    value: Any


Part = TextPart | OpaquePart
PartAdapter = TypeAdapter(Part)
