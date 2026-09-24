from dataclasses import dataclass
from typing import Self, Sequence, cast

from pydantic import BaseModel
from pydantic_ai import (
    Agent,
    AgentRunResult,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    UserPromptPart,
)

from pytoy_llm.connection_configuration import ConnectionConfiguration
from pytoy_llm.models import (
    LLMRequestLike,
)
from pytoy_llm.models.agent_metas import UsageLimit
from pytoy_llm.models.connections import Connection
from pytoy_llm.models.llm_events import LLMEventEmitters
from pytoy_llm.models.llm_messages import LLMRequest, LLMResult
from pytoy_llm.models.llm_metas import LLMParam
from pytoy_llm.models.llm_tools import LLMToolsLike, from_llm_tools_like
from pytoy_llm.pydantic_agent.adapter import PydanticAIMessageAdapter, UsageLimitConverter
from pytoy_llm.pydantic_agent.event_handler import EventHandler
from pytoy_llm.pydantic_agent.factory import PydanticAIModelFactory


@dataclass(frozen=True)
class LatestMessageResolver:
    user_prompt: str | None
    other_parts: Sequence[ModelRequestPart]

    @classmethod
    def from_model_message(cls, model_message: ModelMessage) -> Self:
        if not isinstance(model_message, ModelRequest):
            raise ValueError("Current ModelMessage must be `ModelRequest`")

        def _is_user_prompt_part(part: ModelRequestPart) -> bool:
            return isinstance(part, UserPromptPart) and isinstance(part.content, str)

        user_prompt_parts = [part for part in model_message.parts if _is_user_prompt_part(part)]
        other_parts = [part for part in model_message.parts if not _is_user_prompt_part(part)]
        if user_prompt_parts:
            last_user_prompt = cast(UserPromptPart, user_prompt_parts[-1])
            return cls(user_prompt=str(last_user_prompt.content), other_parts=other_parts)
        return cls(user_prompt=None, other_parts=other_parts)


class PytoyPydanticAIAgent:
    def __init__(
        self,
        connection: str | Connection,
        llm_param: LLMParam | None = None,
        event_emitters: LLMEventEmitters | None = None,
    ) -> None:
        if isinstance(connection, str):
            connection = ConnectionConfiguration().get_connection(connection)
        llm_param = llm_param or connection.llm_param or LLMParam()
        self._connection = connection
        self._llm_param = llm_param
        self._event_emitters = event_emitters or LLMEventEmitters()

    def _make_agent(self, system_prompt: str | None | tuple, tools: LLMToolsLike) -> Agent:
        system_prompt = system_prompt or tuple()
        model = PydanticAIModelFactory.create(self._connection, self._llm_param)
        tools = from_llm_tools_like(tools)
        return Agent(model=model, system_prompt=system_prompt, tools=tools)

    def run[T: BaseModel | str](
        self,
        request: LLMRequestLike,
        output_type: type[T],
        tools: LLMToolsLike = tuple(),
        usage_limit: UsageLimit | None = None,
    ) -> T:
        result = self.run_with_native(
            request=request, output_type=output_type, tools=tools, usage_limit=usage_limit
        )
        return result.output

    def run_with_native[T: BaseModel | str](
        self,
        request: LLMRequestLike,
        output_type: type[T],
        tools: LLMToolsLike = tuple(),
        usage_limit: UsageLimit | None = None,
    ) -> AgentRunResult[T]:
        usage_limits = UsageLimitConverter().to_usage_limits(usage_limit or UsageLimit())

        request = LLMRequest.from_any(request)
        adapter = PydanticAIMessageAdapter()
        if request.system_prompt:
            if request.system_prompt.as_history:
                system_prompt = request.system_prompt.content
                instructions = None
            else:
                system_prompt = None
                instructions = request.system_prompt.content
        else:
            system_prompt = None
            instructions = None
        model_messages = [adapter.to_native(message) for message in request.messages]
        if model_messages:
            message_history, latest_message = model_messages[:-1], model_messages[-1]
            resolver = LatestMessageResolver.from_model_message(latest_message)
            if resolver.user_prompt is None:
                message_history = model_messages
                user_prompt = None
            else:
                message_history = [*message_history, ModelRequest(parts=resolver.other_parts)]
                user_prompt = resolver.user_prompt
        else:
            message_history = []
            user_prompt = None
        agent = self._make_agent(system_prompt=system_prompt, tools=tools)
        event_handler = EventHandler(self._event_emitters)
        event_handler.emit_request(request)
        result = agent.run_sync(
            user_prompt=user_prompt,
            instructions=instructions,
            output_type=output_type,
            message_history=message_history,
            usage_limits=usage_limits,
            event_stream_handler=event_handler.event_stream_handler,
        )
        return result

    def run_with_result[T: BaseModel | str](
        self,
        request: LLMRequestLike,
        output_type: type[T],
        tools: LLMToolsLike = tuple(),
        usage_limit: UsageLimit | None = None,
    ) -> LLMResult[T]:
        adapter = PydanticAIMessageAdapter()
        run_result = self.run_with_native(
            request, output_type=output_type, tools=tools, usage_limit=usage_limit
        )
        return adapter.to_llm_output(run_result)
