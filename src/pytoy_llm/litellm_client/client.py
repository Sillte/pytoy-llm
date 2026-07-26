from itertools import chain
from typing import cast

from litellm import ModelResponse
from pydantic import BaseModel

from pytoy_llm.connection_configuration import ConnectionConfiguration
from pytoy_llm.litellm_client.adapter import LiteLLMMessageAdapter
from pytoy_llm.models.connections import Connection
from pytoy_llm.models.llm_messages import LLMMessage, LLMMessagesLike, LLMResult
from pytoy_llm.models.llm_metas import LLMConfig


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

        self._connection: Connection = connection
        self._llm_config = llm_config

    @property
    def connection(self) -> Connection:
        return self._connection

    def completion[T: BaseModel | str](
        self,
        messages: LLMMessagesLike,
        output_type: type[T],
    ) -> T:
        result = self.completion_with_result(
            messages,
            output_type,
        )
        return result.output

    def completion_with_result[T: BaseModel | str](
        self,
        messages: LLMMessagesLike,
        output_type: type[T],
    ) -> LLMResult[T]:
        message_adapter = LiteLLMMessageAdapter()
        input_messages = LLMMessage.to_messages(messages)
        model_response = self.completion_with_native(input_messages, output_type)
        return message_adapter.to_llm_model(input_messages=input_messages, llm_response=model_response, output_type=output_type)

    def completion_with_native(
        self,
        messages: LLMMessagesLike,
        output_type: type[BaseModel] | type[str],
    ) -> ModelResponse:
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
        return response
