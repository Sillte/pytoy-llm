from typing import cast

from litellm import ModelResponse
from pydantic import BaseModel

from pytoy_llm.connection_configuration import ConnectionConfiguration
from pytoy_llm.litellm_client.adapter import LiteLLMMessageAdapter, LLMParamConverter
from pytoy_llm.litellm_client.event_handler import LiteLLMEventHandler
from pytoy_llm.models.connections import Connection
from pytoy_llm.models.llm_events import LLMEventEmitters
from pytoy_llm.models.llm_messages import LLMRequest, LLMRequestLike, LLMResult
from pytoy_llm.models.llm_metas import LLMParam


class PytoyLiteLLMClient:
    """
    LLM Client for `vim-pytoy`.

    As you know, `vim-pytoy` is a vim(neovim/neovim+vs-code).
    Hence, only text related functions are considered.
    """

    def __init__(
        self,
        connection: str | Connection,
        llm_param: LLMParam | None = None,
        event_emitters: LLMEventEmitters | None = None,
    ) -> None:

        if isinstance(connection, str):
            connection = ConnectionConfiguration().get_connection(connection)
        llm_param = llm_param or connection.llm_param or LLMParam()

        self._connection: Connection = connection
        self._llm_param = llm_param
        self._event_emitters = event_emitters or LLMEventEmitters()

    @property
    def connection(self) -> Connection:
        return self._connection

    def completion[T: BaseModel | str](
        self,
        request: LLMRequestLike,
        output_type: type[T],
    ) -> T:
        result = self.completion_with_result(
            request,
            output_type,
        )
        return result.output

    def completion_with_result[T: BaseModel | str](
        self,
        request: LLMRequestLike,
        output_type: type[T],
    ) -> LLMResult[T]:
        message_adapter = LiteLLMMessageAdapter()
        request = LLMRequest.from_any(request)
        model_response = self.completion_with_native(request, output_type)
        return message_adapter.to_llm_model(
            request=request, llm_response=model_response, output_type=output_type
        )

    def completion_with_native[T: BaseModel | str](
        self,
        request: LLMRequestLike,
        output_type: type[T],
    ) -> ModelResponse:
        from litellm import ModelResponse
        from litellm import completion as litellm_completion

        input_messages = LLMRequest.from_any(request)

        response_format: type[BaseModel] | None

        if output_type is str:
            response_format = None
        else:
            response_format = cast(type[BaseModel], output_type)

        message_adapter = LiteLLMMessageAdapter()

        raw_messages = message_adapter.to_native(input_messages)

        kwargs = LLMParamConverter().to_litellm_kwargs(self._llm_param)

        handler = LiteLLMEventHandler()
        event_emitters = self._event_emitters

        with handler.register(event_emitters) as metadata:
            response = litellm_completion(
                model=self.connection.model,
                messages=raw_messages,
                api_key=self.connection.api_key,
                base_url=self.connection.base_url,
                response_format=response_format,
                metadata=metadata,
                **kwargs,
            )

        assert isinstance(response, ModelResponse)
        return response
