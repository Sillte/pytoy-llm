from collections.abc import Mapping, Sequence
from itertools import chain
from typing import Any, Literal, cast, overload

from litellm import Choices, ModelResponse
from pydantic import BaseModel

from pytoy_llm.connection_configuration import ConnectionConfiguration
from pytoy_llm.litellm_client.adapter import LiteLLMMessageAdapter
from pytoy_llm.models import Connection, LLMConfig, LLMMessage, LLMOutputModel, ResultType


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
        messages: str | LLMMessage | Sequence[Mapping[str, Any]] | Sequence[LLMMessage],
        output_type: type[T],
        result_type: Literal["output"] = "output",
    ) -> T: ...

    @overload
    def completion[T: BaseModel | str](
        self,
        messages: str | LLMMessage | Sequence[Mapping[str, Any]] | Sequence[LLMMessage],
        output_type: type[T],
        result_type: Literal["pytoy-result"],
    ) -> LLMOutputModel[T]: ...

    @overload
    def completion[T: BaseModel | str](
        self,
        messages: str | LLMMessage | Sequence[Mapping[str, Any]] | Sequence[LLMMessage],
        output_type: type[T],
        result_type: Literal["native-result"],
    ) -> ModelResponse: ...

    def completion[T: BaseModel | str](
        self,
        messages: str | LLMMessage | Sequence[Mapping[str, Any]] | Sequence[LLMMessage],
        output_type: type[T],
        result_type: ResultType = "output",
    ) -> T | ModelResponse | LLMOutputModel[T]:
        from litellm import ModelResponse
        from litellm import completion as litellm_completion

        input_messages = LLMMessage.to_messages(messages)

        response_format: type[BaseModel] | None

        if output_type is str:
            response_format = None
        else:
            response_format = cast(type[BaseModel], output_type)

        message_adapter = LiteLLMMessageAdapter()
        chat_messages = [message_adapter.to_native(elem) for elem in input_messages]
        raw_messages = list(chain.from_iterable(chat_messages))

        response = litellm_completion(
            model=self.connection.model,
            messages=raw_messages,
            api_key=self.connection.api_key,
            base_url=self.connection.base_url,
            response_format=response_format,
            **self._llm_config.to_litellm_kwargs(),
        )

        assert isinstance(response, ModelResponse)

        match result_type:
            case "output":
                if not isinstance(response.choices[0], Choices):
                    raise ValueError(f"{response.choices[0]} cannot be recognized.")
                choice = response.choices[0]
                if issubclass(output_type, str):
                    return cast(T, choice.message.content)
                else:
                    return output_type.model_validate(choice.message.content)

            case "native-result":
                return response

            case "pytoy-result":
                return message_adapter.to_llm_model(input_messages=input_messages, llm_response=response)
