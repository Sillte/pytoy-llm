from __future__ import annotations

from pathlib import Path

from pytoy_llm.litellm_client import PytoyLiteLLMClient, Connection
from pytoy_llm.connection_configuration import ConnectionConfiguration, DEFAULT_NAME
from pytoy_llm.models import  SyncOutputType
from pydantic import BaseModel


def initialize_configuration(name: str = DEFAULT_NAME) -> Path:
    return ConnectionConfiguration().initialize_connection_file(name)


def get_configuration_path(name: str = DEFAULT_NAME) -> Path:
    path = ConnectionConfiguration().get_connection_path(name)
    if not path.exists():
        initialize_configuration(name)
    return path


def completion(
    content: str | list,
    output_format: SyncOutputType = str,
    connection: str | Connection = DEFAULT_NAME,
):
    """Execute the `litellm.completion`."""
    client = PytoyLiteLLMClient(connection)
    return client.completion(content, llm_response_format=output_format)
