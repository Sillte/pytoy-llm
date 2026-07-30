from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Self

from pydantic import BaseModel
from pydantic_ai import (
    Agent,
    AgentRunResult,
    ModelMessage,
    ModelRequest,
    SystemPromptPart,
    UserPromptPart,
)

from pytoy_llm.connection_configuration import ConnectionConfiguration
from pytoy_llm.event_sinks import EventSinkProtocol, NullEventSink
from pytoy_llm.models import (
    LLMMessagesLike,
)
from pytoy_llm.models.connections import Connection
from pytoy_llm.models.llm_messages import LLMMessage, LLMResult
from pytoy_llm.models.llm_metas import LLMParam
from pytoy_llm.models.llm_tools import LLMTool
from pytoy_llm.pydantic_agent.adapter import PydanticAIMessageAdapter
from pytoy_llm.pydantic_agent.event_handler import EventHandler
from pytoy_llm.pydantic_agent.factory import PydanticAIModelFactory


@dataclass(frozen=True)
class CurrentModelRequestPair:
    user_prompt: str | None
    system_prompt: str | None

    def __post_init__(self):
        if self.user_prompt is None and self.system_prompt is None:
            raise ValueError("Either of `user` or `system` must be existent.")

    @classmethod
    def from_model_message(cls, model_message: ModelMessage) -> Self:
        if not isinstance(model_message, ModelRequest):
            raise ValueError("Current ModelMessage must be `ModelRequest`")
        user_parts = [part for part in model_message.parts if isinstance(part, UserPromptPart)]
        system_parts = [part for part in model_message.parts if isinstance(part, SystemPromptPart)]
        user_prompt = "\n\n".join(str(part.content) for part in user_parts) if user_parts else None
        if 1 < len(system_parts):
            raise ValueError("SystemPart must be less than 2.")
        system_prompt = system_parts[0].content if system_parts else None
        return cls(user_prompt=user_prompt, system_prompt=system_prompt)


class PytoyPydanticAIAgent:
    def __init__(
        self, connection: str | Connection, llm_param: LLMParam | None = None, event_sink: EventSinkProtocol | None = None
    ) -> None:
        llm_param = llm_param or LLMParam()

        if isinstance(connection, str):
            connection = ConnectionConfiguration().get_connection(connection)
        self._connection = connection
        self._llm_param = llm_param
        self._event_sink = event_sink

    def _normalize_tool(self, tool: LLMTool | Callable) -> Callable:
        if isinstance(tool, LLMTool):
            return tool.to_pydantic_tool()
        else:
            return tool

    def _make_agent(self, system_prompt: str | None | tuple, tools: Sequence[LLMTool | Callable]) -> Agent:
        system_prompt = system_prompt or tuple()
        model = PydanticAIModelFactory.create(self._connection, self._llm_param)
        tools = [self._normalize_tool(tool) for tool in tools]
        return Agent(model=model, system_prompt=system_prompt, tools=tools)

    def run[T: BaseModel | str](
        self,
        messages: LLMMessagesLike,
        output_type: type[T],
        tools: Sequence[LLMTool | Callable] = tuple(),
    ) -> T:
        result = self.run_with_native(messages=messages, output_type=output_type, tools=tools)
        return result.output

    def run_with_native[T: BaseModel | str](
        self, messages: LLMMessagesLike, output_type: type[T], tools: Sequence[LLMTool | Callable] = tuple()
    ) -> AgentRunResult[T]:
        adapter = PydanticAIMessageAdapter()
        messages = LLMMessage.to_messages(messages)
        model_messages = [adapter.to_native(message) for message in messages]
        message_history, current_message = model_messages[:-1], model_messages[-1]
        pair = CurrentModelRequestPair.from_model_message(current_message)
        agent = self._make_agent(system_prompt=pair.system_prompt, tools=tools)
        event_sink = self._event_sink or NullEventSink()
        event_handler = EventHandler(event_sink)
        result = agent.run_sync(
            user_prompt=pair.user_prompt,
            output_type=output_type,
            message_history=message_history,
            event_stream_handler=event_handler.event_stream_handler,
        )
        return result

    def run_with_result[T: BaseModel | str](
        self, messages: LLMMessagesLike, output_type: type[T], tools: Sequence[LLMTool | Callable] = tuple()
    ) -> LLMResult[T]:
        adapter = PydanticAIMessageAdapter()
        run_result = self.run_with_native(messages, output_type=output_type, tools=tools)
        return adapter.to_llm_output(run_result)
