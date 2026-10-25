from pydantic import BaseModel

from pytoy_llm.completion_clients import (
    CompletionClient,
    CompletionClientFactory,
    DefaultCompletionClientFactory,
)
from pytoy_llm.connection_configuration import DEFAULT_NAME
from pytoy_llm.models import CancelToken, LLMRequestLike, LLMResult
from pytoy_llm.models.agent_metas import UsageLimit
from pytoy_llm.models.connections import Connection
from pytoy_llm.models.llm_activities import LLMActivitySink
from pytoy_llm.models.llm_events import LLMEventEmitters
from pytoy_llm.models.llm_metas import LLMParam
from pytoy_llm.models.llm_tools import LLMToolsLike
from pytoy_llm.pydantic_agent.agent import PytoyPydanticAIAgent


class LLMFacade:
    def __init__(
        self,
        connection: str | Connection | None = DEFAULT_NAME,
        llm_param: LLMParam | None = None,
        activity_sink: LLMActivitySink | None = None,
        event_emitters: LLMEventEmitters | None = None,
        *,
        completion_client_factory: CompletionClientFactory | None = None,
    ) -> None:
        self.connection: str | Connection | None = connection
        self.llm_param: LLMParam | None = llm_param
        self.event_emitters = event_emitters or LLMEventEmitters()
        self._completion_client_factory = (
            completion_client_factory
            if completion_client_factory is not None
            else DefaultCompletionClientFactory()
        )

        if activity_sink is not None:
            self.event_emitters.on_activity.subscribe(activity_sink.emit)

    def _resolve_connection(self) -> str | Connection:
        return self.connection or DEFAULT_NAME

    def _create_completion_client(self, cancel_token: CancelToken | None) -> CompletionClient:
        return self._completion_client_factory.create(
            connection=self._resolve_connection(),
            llm_param=self.llm_param,
            event_emitters=self.event_emitters,
            cancel_token=cancel_token,
        )

    def completion[T: BaseModel | str](
        self,
        request: LLMRequestLike,
        output_type: type[T],
        cancel_token: CancelToken | None = None,
    ) -> T:
        client = self._create_completion_client(cancel_token)
        return client.completion(request, output_type=output_type)

    def completion_with_result[T: BaseModel | str](
        self,
        request: LLMRequestLike,
        output_type: type[T],
        cancel_token: CancelToken | None = None,
    ) -> LLMResult[T]:
        client = self._create_completion_client(cancel_token)
        return client.completion_with_result(request, output_type=output_type)

    def run[T: BaseModel | str](
        self,
        request: LLMRequestLike,
        output_type: type[T],
        tools: LLMToolsLike = (),
        usage_limit: UsageLimit | None = None,
        cancel_token: CancelToken | None = None,
    ) -> T:
        """Alias of `run_agent` for better readability."""
        agent = PytoyPydanticAIAgent(
            self._resolve_connection(),
            llm_param=self.llm_param,
            event_emitters=self.event_emitters,
            cancel_token=cancel_token,
        )
        return agent.run(request, output_type=output_type, tools=tools, usage_limit=usage_limit)

    def run_with_result[T: BaseModel | str](
        self,
        request: LLMRequestLike,
        output_type: type[T],
        tools: LLMToolsLike = (),
        usage_limit: UsageLimit | None = None,
        cancel_token: CancelToken | None = None,
    ) -> LLMResult[T]:
        agent = PytoyPydanticAIAgent(
            self._resolve_connection(),
            llm_param=self.llm_param,
            event_emitters=self.event_emitters,
            cancel_token=cancel_token,
        )
        return agent.run_with_result(
            request, output_type=output_type, tools=tools, usage_limit=usage_limit
        )


if __name__ == "__main__":
    facade = LLMFacade()
    result = facade.completion_with_result("Hello", output_type=str)

    from pydantic import BaseModel

    class AModel(BaseModel):
        arg: int

        def is_valid(self) -> bool:
            return True

    facade = LLMFacade()
    result = facade.completion_with_result("Hello", output_type=AModel)
