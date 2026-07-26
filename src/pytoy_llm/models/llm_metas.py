from pydantic import BaseModel, ConfigDict
from pydantic_ai import ModelSettings


class LLMTokens(BaseModel, frozen=True):
    prompt: int
    completion: int
    total: int
    model_config = ConfigDict(extra="allow")


class LLMOutputMeta(BaseModel, frozen=True):
    tokens: LLMTokens
    llm_calls: int = 1
    finish_reason: str | None = None


class LLMConfig(BaseModel, frozen=True):
    temperature: float | None = None
    max_tokens: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None

    def to_litellm_kwargs(self) -> dict:
        return self.model_dump(exclude_none=True)

    def to_pydantic_model_settings(self) -> ModelSettings:
        from pydantic_ai import ModelSettings

        return ModelSettings(**self.model_dump(exclude_none=True))
