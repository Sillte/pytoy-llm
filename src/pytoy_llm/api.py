from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import overload

from pydantic import BaseModel

from pytoy_llm.connection_configuration import DEFAULT_NAME, ConnectionConfiguration
from pytoy_llm.llm_facade import LLMFacade
from pytoy_llm.models import Connection, LLMConfig, LLMMessagesLike, LLMTool
from pytoy_llm.pydantic_agent.agent import PytoyPydanticAIAgent


def initialize_configuration(name: str = DEFAULT_NAME) -> Path:
    return ConnectionConfiguration().initialize_connection_file(name)


def get_configuration_path(name: str = DEFAULT_NAME) -> Path:
    path = ConnectionConfiguration().get_connection_path(name)
    if not path.exists():
        initialize_configuration(name)
    return path


@overload
def completion[T: BaseModel](
    messages: LLMMessagesLike,
    output_type: type[T],
    llm_config: LLMConfig | None = None,
    connection: str | Connection = DEFAULT_NAME,
) -> T: ...


@overload
def completion(
    messages: LLMMessagesLike,
    output_type: type[str],
    llm_config: LLMConfig | None = None,
    connection: str | Connection = DEFAULT_NAME,
) -> str: ...


def completion(
    messages: LLMMessagesLike,
    output_type: type[BaseModel] | type[str] = str,
    llm_config: LLMConfig | None = None,
    connection: str | Connection = DEFAULT_NAME,
) -> BaseModel | str:
    """Execute the `litellm.completion`."""
    facade = LLMFacade(connection, llm_config)
    return facade.completion(messages=messages, output_type=output_type)


@overload
def run(
    messages: LLMMessagesLike,
    output_type: type[str],
    tools: Sequence[Callable | LLMTool],
    llm_config: LLMConfig | None = None,
    connection: str | Connection = DEFAULT_NAME,
) -> str: ...


@overload
def run[T: BaseModel](
    messages: LLMMessagesLike,
    output_type: type[T],
    tools: Sequence[Callable | LLMTool],
    llm_config: LLMConfig | None = None,
    connection: str | Connection = DEFAULT_NAME,
) -> T: ...


def run(
    messages: LLMMessagesLike,
    output_type: type[BaseModel] | type[str],
    tools: Sequence[Callable | LLMTool] = tuple(),
    llm_config: LLMConfig | None = None,
    connection: str | Connection = DEFAULT_NAME,
) -> BaseModel | str:
    """Execute the `pydantic_ai.Agent.run_sync`."""
    agent = PytoyPydanticAIAgent(connection, llm_config=llm_config)
    return agent.run(messages, output_type=output_type, tools=tools)
