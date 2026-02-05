from __future__ import annotations

from pathlib import Path
from typing import Sequence, Callable

from pytoy_llm.litellm_client import PytoyLiteLLMClient, Connection
from pytoy_llm.connection_configuration import ConnectionConfiguration, DEFAULT_NAME
from pytoy_llm.models import SyncOutputType, LLMTool, InputMessage, LLMConfig


def initialize_configuration(name: str = DEFAULT_NAME) -> Path:
    return ConnectionConfiguration().initialize_connection_file(name)


def get_configuration_path(name: str = DEFAULT_NAME) -> Path:
    path = ConnectionConfiguration().get_connection_path(name)
    if not path.exists():
        initialize_configuration(name)
    return path


def completion(
    content: str | list | Sequence[InputMessage],
    output_format: SyncOutputType = str,
    llm_config: LLMConfig | None = None,
    connection: str | Connection = DEFAULT_NAME,
):
    """Execute the `litellm.completion`."""
    client = PytoyLiteLLMClient(connection, llm_config=llm_config)
    return client.completion(content, llm_response_format=output_format)


def run_agent(content: str | list | Sequence[InputMessage],
              output_format: SyncOutputType = str,
              tools: Sequence[Callable | LLMTool] = tuple(),
              llm_config: LLMConfig | None = None,
              connection: str | Connection= DEFAULT_NAME):
    """Execute the `pydantic_ai.Agent.run_sync`."""
    from pytoy_llm.pydantic_agent import PytoyAgent
    agent = PytoyAgent(connection, tools=tools, llm_config=llm_config)
    return agent.run_sync(content, llm_response_format=output_format)
