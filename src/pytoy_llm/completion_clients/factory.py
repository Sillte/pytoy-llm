from typing import assert_never

from pytoy_llm.completion_clients.protocol import CompletionClient, CompletionClientFactory
from pytoy_llm.connection_configuration import ConnectionConfiguration
from pytoy_llm.models import CancelToken
from pytoy_llm.models.connections import Connection
from pytoy_llm.models.llm_events import LLMEventEmitters
from pytoy_llm.models.llm_metas import LLMParam


class DefaultCompletionClientFactory(CompletionClientFactory):
    def create(
        self,
        connection: str | Connection,
        llm_param: LLMParam | None = None,
        event_emitters: LLMEventEmitters | None = None,
        cancel_token: CancelToken | None = None,
    ) -> CompletionClient:
        if isinstance(connection, str):
            connection = ConnectionConfiguration().get_connection(connection)

        match connection.completion_backend:
            case "litellm":
                try:
                    from pytoy_llm.completion_clients.litellm_client import PytoyLiteLLMClient
                except ModuleNotFoundError as error:
                    if error.name != "litellm":
                        raise
                    raise ImportError(
                        "The LiteLLM backend requires the optional dependency. "
                        "Install it with `pip install pytoy-llm[litellm]`."
                    ) from error
                return PytoyLiteLLMClient(
                    connection,
                    llm_param=llm_param,
                    event_emitters=event_emitters,
                    cancel_token=cancel_token,
                )
            case "openai":
                from pytoy_llm.completion_clients.openai_client import PytoyOpenAIClient

                return PytoyOpenAIClient(
                    connection,
                    llm_param=llm_param,
                    event_emitters=event_emitters,
                    cancel_token=cancel_token,
                )
            case _:
                assert_never(connection.completion_backend)
