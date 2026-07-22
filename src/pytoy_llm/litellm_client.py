from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    Mapping,
    Sequence,
    assert_never,
    cast,
    overload,
)

from pydantic import BaseModel

from pytoy_llm.connection_configuration import ConnectionConfiguration
from pytoy_llm.models import (
    Connection,
    InputMessage,
    LLMConfig,
    LLMOutputModel,
    ResultType,
)

if TYPE_CHECKING:
    from litellm import ModelResponse


class ModelResponseConverter[T: BaseModel | str]:
    def __init__(
        self,
        output_type: type[T],
        result_type: ResultType,
    ) -> None:
        self.output_type = output_type
        self.result_type: Final[ResultType] = result_type

    def convert(
        self,
        response: ModelResponse,
        input_messages: Sequence[InputMessage],
    ) -> T | ModelResponse | LLMOutputModel[T]:

        from litellm import Choices
        choices = cast(Choices, response.choices)
        choice = choices[0]
        raw_content = choice.message.content or ""

        match self.result_type:
            case "output":
                if self.output_type is str:
                    return cast(T, raw_content)

                elif issubclass(self.output_type, BaseModel):
                    return self.output_type.model_validate_json(raw_content)

                else:
                    raise ValueError(
                        "Unsupported output type",
                        self.output_type,
                    )

            case "pytoy-result":
                return LLMOutputModel.from_litellm_model_response(
                    response,
                    input_messages=input_messages,
                )

            case "native-result":
                return response

            case unexpected:
                assert_never(unexpected)


class PytoyLiteLLMClient:
    """
    LLM Client for `vim-pytoy`.

    As you know, `vim-pytoy` is a vim(neovim/neovim+vs-code).
    Hence, only text related functions are considered.
    """

    def __init__(
        self,
        connection: str | Connection,
        llm_config: LLMConfig | None = None,
    ) -> None:
        llm_config = llm_config or LLMConfig()

        if isinstance(connection, str):
            connection = ConnectionConfiguration().get_connection(connection)

        self._connection = connection
        self._llm_config = llm_config

    @property
    def connection(self) -> Connection:
        return self._connection

    @overload
    def completion[T: BaseModel | str](
        self,
        content: str | InputMessage | Sequence[InputMessage | str | Mapping],
        output_type: type[T],
        result_type: Literal["output"] = "output",
    ) -> T:
        ...

    @overload
    def completion[T: BaseModel | str](
        self,
        content: str | InputMessage | Sequence[InputMessage | str | Mapping],
        output_type: type[T],
        result_type: Literal["pytoy-result"],
    ) -> LLMOutputModel[T]:
        ...

    @overload
    def completion[T: BaseModel | str](
        self,
        content: str | InputMessage | Sequence[InputMessage | str | Mapping],
        output_type: type[T],
        result_type: Literal["native-result"],
    ) -> ModelResponse:
        ...

    def completion[T: BaseModel | str](
        self,
        content: str | InputMessage | Sequence[InputMessage | str | Mapping],
        output_type: type[T],
        result_type: ResultType = "output",
    ) -> T | ModelResponse | LLMOutputModel[T]:
        from litellm import ModelResponse, completion as litellm_completion

        messages = InputMessage.to_messages(content)

        response_format: type[BaseModel] | None

        if output_type is str:
            response_format = None
        else:
            response_format = cast(type[BaseModel], output_type)

        response = litellm_completion(
            model=self.connection.model,
            messages=[elem.model_dump() for elem in messages],
            api_key=self.connection.api_key,
            base_url=self.connection.base_url,
            response_format=response_format,
            **self._llm_config.to_litellm_kwargs(),
        )

        assert isinstance(response, ModelResponse)

        converter = ModelResponseConverter(
            output_type,
            result_type,
        )

        return converter.convert(
            response,
            input_messages=messages,
        )