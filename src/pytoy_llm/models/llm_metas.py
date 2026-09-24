from collections.abc import Iterable
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict

type ReasoningEffort = Literal["none", "minimal", "low", "medium", "high"]
type Verbosity = Literal["low", "medium", "high"]


class LLMTokens(BaseModel, frozen=True):
    prompt: int
    completion: int
    total: int

    @classmethod
    def aggregate(cls, llm_tokens: Iterable[Self]) -> Self:
        prompt = sum(elem.prompt for elem in llm_tokens)
        completion = sum(elem.completion for elem in llm_tokens)
        total = sum(elem.total for elem in llm_tokens)
        return cls(prompt=prompt, completion=completion, total=total)


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
