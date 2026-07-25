from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Self, assert_never, overload

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
from pytoy_llm.models import (
    Connection,
    LLMConfig,
    LLMMessage,
    LLMOutputModel,
    LLMTool,
    ResultType,
)
from pytoy_llm.pydantic_agent.adapter import PydanticAIMessageAdapter
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
        self,
        connection: str | Connection,
        llm_config: LLMConfig | None = None,
    ) -> None:
        llm_config = llm_config or LLMConfig()

        if isinstance(connection, str):
            connection = ConnectionConfiguration().get_connection(connection)
        self._connection = connection
        self._llm_config = llm_config

    def _normalize_tool(self, tool: LLMTool | Callable) -> Callable:
        if isinstance(tool, LLMTool):
            return tool.to_pydantic_tool()
        else:
            return tool

    def _make_agent(self, system_prompt: str | None | tuple, tools: Sequence[LLMTool | Callable]):
        system_prompt = system_prompt or tuple()
        model = PydanticAIModelFactory.create(self._connection, self._llm_config)
        tools = [self._normalize_tool(tool) for tool in tools]
        return Agent(model=model, system_prompt=system_prompt, tools=tools)

    @overload
    def run_sync[T: BaseModel | str](
        self,
        messages: Sequence[LLMMessage] | str | LLMMessage | Sequence[Mapping[str, Any]],
        output_type: type[T],
        tools: Sequence[LLMTool | Callable] = tuple(),
        result_type: Literal["output"] = "output",
    ) -> T: ...

    @overload
    def run_sync[T: BaseModel | str](
        self,
        messages: Sequence[LLMMessage] | str | LLMMessage | Sequence[Mapping[str, Any]],
        output_type: type[T],
        tools: Sequence[LLMTool | Callable] = tuple(),
        result_type: Literal["native-result"] = "native-result",
    ) -> AgentRunResult[T]: ...

    @overload
    def run_sync[T: BaseModel | str](
        self,
        messages: Sequence[LLMMessage] | str | LLMMessage | Sequence[Mapping[str, Any]],
        output_type: type[T],
        tools: Sequence[LLMTool | Callable] = tuple(),
        result_type: Literal["pytoy-result"] = "pytoy-result",
    ) -> LLMOutputModel[T]: ...

    def run_sync[T: BaseModel | str](
        self,
        messages: Sequence[LLMMessage] | str | LLMMessage | Sequence[Mapping[str, Any]],
        output_type: type[T] = str,
        tools: Sequence[LLMTool | Callable] = tuple(),
        result_type: ResultType = "output",
    ) -> T | AgentRunResult[T] | LLMOutputModel[T]:
        adapter = PydanticAIMessageAdapter()
        messages = LLMMessage.to_messages(messages)
        model_messages = [adapter.to_native(message) for message in messages]
        message_history, current_message = model_messages[:-1], model_messages[-1]
        pair = CurrentModelRequestPair.from_model_message(current_message)

        agent = self._make_agent(system_prompt=pair.system_prompt, tools=tools)

        result = agent.run_sync(
            user_prompt=pair.user_prompt,
            output_type=output_type,
            message_history=message_history,
        )

        match result_type:
            case "native-result":
                return result
            case "pytoy-result":
                return adapter.to_llm_output(result)
            case "output":
                return result.output
            case _:
                assert_never(result_type)
