from collections.abc import Iterable
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict

type ReasoningEffort = Literal["none", "minimal", "low", "medium", "high"]
type Verbosity = Literal["low", "medium", "high"]


class LLMTokens(BaseModel, frozen=True):
    prompt: int
    completion: int
    total: int
    cache_read: int | None = None
    cache_write: int | None = None

    @classmethod
    def aggregate(cls, llm_tokens: Iterable[Self]) -> Self:
        tokens = list(llm_tokens)

        return cls(
            prompt=sum(x.prompt for x in tokens),
            completion=sum(x.completion for x in tokens),
            total=sum(x.total for x in tokens),
            cache_read=(
                sum(x.cache_read if x.cache_read else 0 for x in tokens)
                if all(x.cache_read is not None for x in tokens)
                else None
            ),
            cache_write=(
                sum(x.cache_write if x.cache_write else 0 for x in tokens)
                if all(x.cache_write is not None for x in tokens)
                else None
            ),
        )


class LLMOutputMeta(BaseModel, frozen=True):
    tokens: LLMTokens
    llm_calls: int = 1
    finish_reason: str | None = None


class LLMParam(BaseModel, frozen=True):
    model_config = ConfigDict(extra="allow")
    temperature: float | None = None
    max_tokens: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    reasoning_effort: ReasoningEffort | None = None
    verbosity: Verbosity | None = None
