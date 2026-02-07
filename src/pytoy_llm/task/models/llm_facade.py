from pytoy_llm.connection_configuration import DEFAULT_NAME
from pytoy_llm.impl import completion, run_agent
from pytoy_llm.models import InputMessage, LLMConfig, LLMTool
from pytoy_llm.task.models.context_protocols import LLMFacadeProtocol

from pydantic import BaseModel


from typing import Callable, Sequence, cast


class LLMFacade[T: BaseModel | str](LLMFacadeProtocol[T]):
    def __init__(self, connection_name: str = DEFAULT_NAME, llm_config: LLMConfig | None = None):
        self.connection_name = connection_name
        self.llm_config = llm_config

    def completion(
        self,
        input_messages: Sequence[InputMessage],
        output_format: type[str] | type[T],
        llm_config: LLMConfig | None,
        connection_name: str | None = None,
    ) -> str | T:
        connection_name = connection_name or self.connection_name
        llm_config = llm_config or self.llm_config
        raw_output = completion(
            input_messages, output_format, connection=connection_name, llm_config=llm_config
        )  # type: ignore
        if isinstance(output_format, type) and issubclass(output_format, BaseModel):
            return output_format.model_validate(raw_output)
        elif output_format is str:
            return raw_output  # type: ignore
        else:
            raise TypeError(f"Unsupported output_format type: {output_format}")

    def run_agent(
        self,
        input_messages: Sequence[InputMessage],
        output_format: type[str] | type[T],
        tools: Sequence[Callable | LLMTool] = (),
        llm_config: LLMConfig | None = None,
        connection_name: str | None = None,
    ) -> str | T:
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
        return cast(str | T, result)