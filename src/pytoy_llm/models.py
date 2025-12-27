from typing import Annotated, Sequence, Any, Literal, Self, cast
import litellm
from abc import abstractmethod, ABC
from pydantic import BaseModel, Field, field_validator
from pydantic import StringConstraints

StrictStr = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]


class Connection(BaseModel, frozen=True):
    model: Annotated[
        StrictStr,
        Field(
            description="Model Name of LLM",
            examples=["gemini/gemini-2.0-flash", "gpt-4o"],
        ),
    ]
    base_url: Annotated[
        StrictStr,
        Field(
            description="Endpoint for LLM.",
            examples=["https://"],
        ),
    ]
    api_key: Annotated[
        StrictStr,
        Field(description="Credential Information for using LLM.", examples=["SECRET-KEY"]),
    ]

    @field_validator("base_url", mode="before")
    @classmethod
    def normalize_base_url(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip("/")
        else:
            return value


class InputMessage(BaseModel, frozen=True):
    role: Annotated[
        Literal["system", "user", "assistant"],
        Field(description="Role of Message in LLM"),
    ]
    content: Annotated[str, Field(description="Content of the message")]


class LLMMessageHistory(BaseModel, frozen=True):
    items: Annotated[Sequence[InputMessage], Field(description="")]


class CustomLLMOutputModel(BaseModel, ABC, frozen=True):
    @classmethod
    @abstractmethod
    def from_litellm_model_response(cls, response: litellm.ModelResponse) -> Self: ...


class LLMOutputModel(CustomLLMOutputModel, frozen=True):
    content: Annotated[str, Field(description="The main text content from LLM")]
    model: str
    usage: litellm.Usage | None

    @classmethod
    def from_litellm_model_response(cls, response: litellm.ModelResponse) -> Self:
        choices = cast(litellm.Choices, response.choices)
        choice = choices[0]
        usage = getattr(response, "usage", None)
        return cls(content=choice.message.content or "", model=response.model or "", usage=usage)


SyncOutputTypeStr = Literal["str", "all"]
type SyncOutputType = (
    type[litellm.ModelResponse]
    | type[CustomLLMOutputModel]
    | type[LLMOutputModel]
    | type[BaseModel]
    | type[str]
)
type SyncOutputMode = SyncOutputType | SyncOutputTypeStr
type SyncOutput = litellm.ModelResponse | CustomLLMOutputModel | LLMOutputModel | BaseModel | str
