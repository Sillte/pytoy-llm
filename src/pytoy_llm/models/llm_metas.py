from pydantic import BaseModel, ConfigDict


class LLMTokens(BaseModel, frozen=True):
    prompt: int
    completion: int
    total: int
    model_config = ConfigDict(extra="allow")


class LLMOutputMeta(BaseModel, frozen=True):
    tokens: LLMTokens
    llm_calls: int = 1
    finish_reason: str | None = None


class LLMParam(BaseModel, frozen=True):
    temperature: float | None = None
    max_tokens: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
