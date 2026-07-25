from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from pytoy_llm.connection_configuration import DEFAULT_NAME, ConnectionConfiguration
from pytoy_llm.litellm_client.client import PytoyLiteLLMClient
from pytoy_llm.models import Connection, LLMConfig, LLMMessage, LLMTool
from pytoy_llm.pydantic_agent.agent import PytoyPydanticAIAgent


def initialize_configuration(name: str = DEFAULT_NAME) -> Path:
    return ConnectionConfiguration().initialize_connection_file(name)


def get_configuration_path(name: str = DEFAULT_NAME) -> Path:
    path = ConnectionConfiguration().get_connection_path(name)
    if not path.exists():
        initialize_configuration(name)
    return path


def completion[T: BaseModel | str](
    messages: Sequence[LLMMessage] | str | Sequence[Mapping[str, Any]] | LLMMessage,
    output_type: type[T] = str,
    llm_config: LLMConfig | None = None,
    connection: str | Connection = DEFAULT_NAME,
) -> T:
    """Execute the `litellm.completion`."""
    client = PytoyLiteLLMClient(connection, llm_config=llm_config)
    return client.completion(messages, output_type=output_type, result_type="output")


def run_agent[T: BaseModel | str](
    messages: Sequence[LLMMessage] | str | Sequence[Mapping[str, Any]] | LLMMessage,
    output_type: type[T] = str,
    tools: Sequence[Callable | LLMTool] = tuple(),
    llm_config: LLMConfig | None = None,
    connection: str | Connection = DEFAULT_NAME,
) -> T:
    """Execute the `pydantic_ai.Agent.run_sync`."""

    agent = PytoyPydanticAIAgent(connection, llm_config=llm_config)
    return agent.run_sync(messages, output_type=output_type, result_type="output", tools=tools)
