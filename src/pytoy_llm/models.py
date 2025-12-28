from typing import Annotated, Sequence, Any, Literal, Self, cast, assert_never
import litellm
from abc import abstractmethod, ABC
from pydantic import BaseModel, Field, field_validator
from pydantic import StringConstraints
from pydantic.dataclasses import dataclass

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


SyncOutputFormatStr = Literal["str", "all"]
type SyncOutputType = (
    type[litellm.ModelResponse] | type[CustomLLMOutputModel] | type[BaseModel] | type[str]
)
type SyncOutput = litellm.ModelResponse | CustomLLMOutputModel | BaseModel | str


@dataclass
class SyncOutputFormat:
    output_type: Annotated[SyncOutputType, Field(description="The class type of output.")]

    @classmethod
    def from_str(cls, output_str: SyncOutputFormatStr) -> Self:
        match output_str:
            case "all":
                return cls(output_type=litellm.ModelResponse)
            case "str":
                return cls(output_type=str)
            case _:
                assert_never(output_str)

    @classmethod
    def from_type(cls, output_type: SyncOutputType) -> Self:
        return cls(output_type=output_type)

    @classmethod
    def from_any(cls, output_type: type | SyncOutputFormatStr) -> Self:
        if isinstance(output_type, str):
            return cls.from_str(output_str=output_type)
        else:
            return cls.from_type(output_type=output_type)

    @property
    def litellm_response_format(self) -> None | type[BaseModel]:
        """
        Determine the `response_format` parameter for `litellm.completion`.

        This value controls how the LLM is instructed to generate its output,
        and is derived from the *desired final output type* (`output_type`).

        - If the final output type is a Pydantic model that should be generated
          directly by the LLM, return that model type.
        - If the final output requires post-processing (e.g. CustomLLMOutputModel)
          return `None`.  -
        - If the final output requires the `litellm.ModelResponse`,
          return `None`.
        """

        if issubclass(self.output_type, BaseModel):
            if issubclass(self.output_type, CustomLLMOutputModel):
                return None
            elif issubclass(self.output_type, litellm.ModelResponse):
                return None
            else:
                return self.output_type
        else:
            return None
