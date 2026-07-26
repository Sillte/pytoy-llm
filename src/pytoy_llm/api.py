from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel

from pytoy_llm.connection_configuration import DEFAULT_NAME, ConnectionConfiguration
from pytoy_llm.llm_facade import LLMFacade
from pytoy_llm.models import LLMMessagesLike
from pytoy_llm.models.connections import Connection
from pytoy_llm.models.llm_metas import LLMConfig
from pytoy_llm.models.llm_tools import LLMTool
from pytoy_llm.pydantic_agent.agent import PytoyPydanticAIAgent


def initialize_configuration(name: str = DEFAULT_NAME) -> Path:
    return ConnectionConfiguration().initialize_connection_file(name)


def get_configuration_path(name: str = DEFAULT_NAME) -> Path:
    path = ConnectionConfiguration().get_connection_path(name)
    if not path.exists():
        initialize_configuration(name)
    return path


def completion[T: BaseModel | str](
    messages: LLMMessagesLike,
    output_type: type[T] = str,  # type: ignore
    llm_config: LLMConfig | None = None,
    connection: str | Connection = DEFAULT_NAME,
) -> T:
    """Execute the `litellm.completion`."""
    facade = LLMFacade(connection, llm_config)
    return facade.completion(messages=messages, output_type=output_type)


def run[T: BaseModel | str](
    messages: LLMMessagesLike,
    output_type: type[T],
    tools: Sequence[Callable | LLMTool] = tuple(),
    llm_config: LLMConfig | None = None,
    connection: str | Connection = DEFAULT_NAME,
) -> T:
    """Execute the `pydantic_ai.Agent.run_sync`."""
    agent = PytoyPydanticAIAgent(connection, llm_config=llm_config)
    result = agent.run(messages, output_type=output_type, tools=tools)
    return result


if __name__ == "__main__":
    ...
    result = completion("hogehoge", output_type=str)
