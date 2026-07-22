from __future__ import annotations

from pathlib import Path
from typing import Sequence, Callable
from pydantic import BaseModel

from pytoy_llm.litellm_client import PytoyLiteLLMClient, Connection
from pytoy_llm.connection_configuration import ConnectionConfiguration, DEFAULT_NAME
from pytoy_llm.models import LLMTool, InputMessage, LLMConfig


def initialize_configuration(name: str = DEFAULT_NAME) -> Path:
    return ConnectionConfiguration().initialize_connection_file(name)


def get_configuration_path(name: str = DEFAULT_NAME) -> Path:
    path = ConnectionConfiguration().get_connection_path(name)
    if not path.exists():
        initialize_configuration(name)
    return path


def completion[T: BaseModel | str](
    content: str | list | Sequence[InputMessage],
    output_type: type[T] = str,
    llm_config: LLMConfig | None = None,
    connection: str | Connection = DEFAULT_NAME,
) -> T:
    """Execute the `litellm.completion`."""
    client = PytoyLiteLLMClient(connection, llm_config=llm_config)
    return client.completion(content, output_type=output_type, result_type="output")


def run_agent[T: BaseModel | str](content: str | list | Sequence[InputMessage],
              output_type: type[T] = str,
              tools: Sequence[Callable | LLMTool] = tuple(),
              llm_config: LLMConfig | None = None,
              connection: str | Connection = DEFAULT_NAME) -> T:
    """Execute the `pydantic_ai.Agent.run_sync`."""
    from pytoy_llm.pydantic_agent import PytoyAgent
    agent = PytoyAgent(connection, llm_config=llm_config)
    return agent.run_sync(content, output_type=output_type, result_type="output", tools=tools)
