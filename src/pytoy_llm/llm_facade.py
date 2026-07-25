from collections.abc import Callable, Sequence
from typing import overload

from pydantic import BaseModel

from pytoy_llm.connection_configuration import DEFAULT_NAME
from pytoy_llm.litellm_client.client import PytoyLiteLLMClient
from pytoy_llm.models import Connection, LLMConfig, LLMMessagesLike, LLMTool
from pytoy_llm.pydantic_agent.agent import PytoyPydanticAIAgent


class LLMFacade:
    def __init__(self, connection: str | Connection = DEFAULT_NAME, llm_config: LLMConfig | None = None):
        self.connection = connection
        self.llm_config = llm_config

    @overload
    def completion(
        self,
        messages: LLMMessagesLike,
        output_type: type[str],
    ) -> str: ...

    @overload
    def completion[T: BaseModel](
        self,
        messages: LLMMessagesLike,
        output_type: type[T],
    ) -> T: ...

    @overload
    def completion(
        self,
        messages: LLMMessagesLike,
        output_type: type[BaseModel] | type[str],
    ) -> BaseModel | str: ...

    def completion(
        self,
        messages: LLMMessagesLike,
        output_type: type[BaseModel] | type[str],
    ) -> BaseModel | str:
        client = PytoyLiteLLMClient(self.connection, llm_config=self.llm_config)
        return client.completion(messages, output_type=output_type)

    @overload
    def run(
        self,
        messages: LLMMessagesLike,
        output_type: type[str],
        tools: Sequence[Callable | LLMTool] = (),
    ) -> str: ...

    @overload
    def run[T: BaseModel](
        self,
        messages: LLMMessagesLike,
        output_type: type[T],
        tools: Sequence[Callable | LLMTool] = (),
    ) -> T: ...

    @overload
    def run(
        self,
        messages: LLMMessagesLike,
        output_type: type[BaseModel] | type[str],
        tools: Sequence[Callable | LLMTool] = (),
    ) -> BaseModel | str: ...

    def run(
        self,
        messages: LLMMessagesLike,
        output_type: type[BaseModel] | type[str],
        tools: Sequence[Callable | LLMTool] = (),
    ) -> BaseModel | str:
        """Alias of `run_agent` for better readability."""
        agent = PytoyPydanticAIAgent(self.connection, llm_config=self.llm_config)
        return agent.run(messages, output_type=output_type, tools=tools)
