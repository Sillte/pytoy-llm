from typing import Self

from pydantic import BaseModel


class SystemPrompt(BaseModel, frozen=True):
    content: str
    as_history: bool = True

    @classmethod
    def from_any(cls, content: str | Self, *, as_history: bool = False) -> Self:
        if isinstance(content, cls):
            return content
        elif isinstance(content, str):
            return cls(content=content, as_history=as_history)
        else:
            raise ValueError(f"Cannot convert {content} to SystemPrompt.")
