from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from pytoy_llm.connection_configuration import DEFAULT_NAME, ConnectionConfiguration
from pytoy_llm.event_sinks.protocol import EventSinkProtocol
from pytoy_llm.llm_facade import LLMFacade
from pytoy_llm.models import LLMMessagesLike
from pytoy_llm.models.connections import Connection
from pytoy_llm.models.llm_metas import LLMParam
from pytoy_llm.models.llm_tools import LLMToolsLike


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
    llm_param: LLMParam | None = None,
    connection: str | Connection = DEFAULT_NAME,
    event_sink: EventSinkProtocol | None = None,
) -> T:
    """Execute the `litellm.completion`."""
    facade = LLMFacade(connection=connection, llm_param=llm_param, event_sink=event_sink)
    return facade.completion(messages=messages, output_type=output_type)


def run[T: BaseModel | str](
    messages: LLMMessagesLike,
    output_type: type[T],
    tools: LLMToolsLike = tuple(),
    llm_param: LLMParam | None = None,
    connection: str | Connection = DEFAULT_NAME,
    event_sink: EventSinkProtocol | None = None,
) -> T:
    """Execute the `pydantic_ai.Agent.run_sync`."""
    facade = LLMFacade(connection=connection, llm_param=llm_param, event_sink=event_sink)
    result = facade.run(messages, output_type=output_type, tools=tools)
    return result


if __name__ == "__main__":
    ...
    result = completion("hogehoge", output_type=str)
