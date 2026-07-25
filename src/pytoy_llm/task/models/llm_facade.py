from collections.abc import Callable, Sequence

from pydantic import BaseModel

from pytoy_llm.connection_configuration import DEFAULT_NAME
from pytoy_llm.impl import completion, run_agent
from pytoy_llm.models import LLMConfig, LLMMessage, LLMTool
from pytoy_llm.task.models.context_protocols import LLMFacadeProtocol


class LLMFacade[T: BaseModel | str](LLMFacadeProtocol[T]):
    def __init__(self, connection_name: str = DEFAULT_NAME, llm_config: LLMConfig | None = None):
        self.connection_name = connection_name
        self.llm_config = llm_config

    def completion(
        self,
        input_messages: Sequence[LLMMessage],
        output_type: type[T],
        llm_config: LLMConfig | None,
        connection_name: str | None = None,
    ) -> T:
        connection_name = connection_name or self.connection_name
        llm_config = llm_config or self.llm_config
        return completion(
            input_messages, output_type, connection=connection_name, llm_config=llm_config
        ) 

    def run_agent(
        self,
        input_messages: Sequence[LLMMessage],
        output_format: type[T],
        tools: Sequence[Callable | LLMTool] = (),
        llm_config: LLMConfig | None = None,
        connection_name: str | None = None,
    ) -> T:
        """Alias of `run_agent` for better readability."""
        connection_name = connection_name or self.connection_name
        llm_config = llm_config or self.llm_config
        result = run_agent(
            input_messages,
            output_format,
            tools=tools,
            connection=connection_name,
            llm_config=llm_config,
        )
        return result