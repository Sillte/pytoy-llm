from __future__ import annotations

from pathlib import Path

from pytoy_llm.client import PytoyLLMClient, Connection
from pytoy_llm.configurations import ConfigurationClient, DEFAULT_NAME
from pytoy_llm.models import SyncOutputFormat, SyncOutputFormatStr
from pydantic import BaseModel


def initialize_configuration(name: str = DEFAULT_NAME) -> Path:
    return ConfigurationClient().initialize_connection_file(name)


def completion(
    content: str | list,
    output_format: SyncOutputFormat | SyncOutputFormatStr | type[BaseModel] = "str",
    connection: str | Connection = DEFAULT_NAME,
):
    """Execute the `litellm.completion`."""
    client = PytoyLLMClient(connection)
    return client.completion(content, output_format=output_format)
