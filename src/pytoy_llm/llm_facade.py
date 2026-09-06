from pydantic import BaseModel

from pytoy_llm.activity_sinks.protocol import ActivitySinkProtocol
from pytoy_llm.connection_configuration import DEFAULT_NAME
from pytoy_llm.litellm_client.client import PytoyLiteLLMClient
from pytoy_llm.models import LLMMessagesLike
from pytoy_llm.models.agent_metas import UsageLimit
from pytoy_llm.models.connections import Connection
from pytoy_llm.models.llm_events import LLMEventEmitters
from pytoy_llm.models.llm_messages import LLMResult
from pytoy_llm.models.llm_metas import LLMParam
from pytoy_llm.models.llm_tools import LLMToolsLike
from pytoy_llm.pydantic_agent.agent import PytoyPydanticAIAgent


class LLMFacade:
    def __init__(
        self,
        connection: str | Connection | None = DEFAULT_NAME,
        llm_param: LLMParam | None = None,
        activity_sink: ActivitySinkProtocol | None = None,
        event_emitters: LLMEventEmitters | None = None,
    ) -> None:
        self.connection: str | Connection | None = connection
        self.llm_param: LLMParam | None = llm_param
        self.activity_sink = activity_sink
        self.event_emitters = event_emitters

    def _resolve_connection(self) -> str | Connection:
        return self.connection or DEFAULT_NAME

    def completion[T: BaseModel | str](
        self,
        messages: LLMMessagesLike,
        output_type: type[T],
    ) -> T:
        if self.event_emitters is None:
            client = PytoyLiteLLMClient(self._resolve_connection(), llm_param=self.llm_param, activity_sink=self.activity_sink)
        else:
            client = PytoyLiteLLMClient(
                self._resolve_connection(),
                llm_param=self.llm_param,
                activity_sink=self.activity_sink,
                event_emitters=self.event_emitters,
            )
        return client.completion(messages, output_type=output_type)

    def completion_with_result[T: BaseModel | str](
        self,
        messages: LLMMessagesLike,
        output_type: type[T],
    ) -> LLMResult[T]:
        if self.event_emitters is None:
            client = PytoyLiteLLMClient(self._resolve_connection(), llm_param=self.llm_param, activity_sink=self.activity_sink)
        else:
            client = PytoyLiteLLMClient(
                self._resolve_connection(),
                llm_param=self.llm_param,
                activity_sink=self.activity_sink,
                event_emitters=self.event_emitters,
            )
        return client.completion_with_result(messages, output_type=output_type)

    def run[T: BaseModel | str](
        self,
        messages: LLMMessagesLike,
        output_type: type[T],
        tools: LLMToolsLike = (),
        usage_limit: UsageLimit | None = None,
    ) -> T:
        """Alias of `run_agent` for better readability."""
        if self.event_emitters is None:
            agent = PytoyPydanticAIAgent(self._resolve_connection(), llm_param=self.llm_param, activity_sink=self.activity_sink)
        else:
            agent = PytoyPydanticAIAgent(
                self._resolve_connection(),
                llm_param=self.llm_param,
                activity_sink=self.activity_sink,
                event_emitters=self.event_emitters,
            )
        return agent.run(messages, output_type=output_type, tools=tools, usage_limit=usage_limit)

    def run_with_result[T: BaseModel | str](
        self, messages: LLMMessagesLike, output_type: type[T], tools: LLMToolsLike = (), usage_limit: UsageLimit | None = None
    ) -> LLMResult[T]:
        if self.event_emitters is None:
            agent = PytoyPydanticAIAgent(self._resolve_connection(), llm_param=self.llm_param, activity_sink=self.activity_sink)
        else:
            agent = PytoyPydanticAIAgent(
                self._resolve_connection(),
                llm_param=self.llm_param,
                activity_sink=self.activity_sink,
                event_emitters=self.event_emitters,
            )
        return agent.run_with_result(messages, output_type=output_type, tools=tools, usage_limit=usage_limit)


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
