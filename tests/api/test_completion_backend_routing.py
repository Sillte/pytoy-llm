import builtins
import sys
from types import ModuleType
from typing import Literal
from unittest.mock import Mock

import pytest

from pytoy_llm.completion_clients import DefaultCompletionClientFactory
from pytoy_llm.completion_clients.openai_client import PytoyOpenAIClient
from pytoy_llm.llm_facade import LLMFacade
from pytoy_llm.models import CancelToken
from pytoy_llm.models.connections import Connection


def make_connection(backend: Literal["openai", "litellm"] = "openai") -> Connection:
    return Connection(
        model="openai/test-model",
        base_url="https://example.com/v1",
        api_key="test-key",
        completion_backend=backend,
    )


def test_facade_accepts_injected_completion_client_factory() -> None:
    connection = make_connection()
    cancel_token = CancelToken()
    client = Mock()
    client.completion.return_value = "from-injected-client"
    factory = Mock()
    factory.create.return_value = client
    facade = LLMFacade(connection=connection, completion_client_factory=factory)

    result = facade.completion("hello", output_type=str, cancel_token=cancel_token)

    assert result == "from-injected-client"
    factory.create.assert_called_once_with(
        connection=connection,
        llm_param=None,
        event_emitters=facade.event_emitters,
        cancel_token=cancel_token,
    )


def test_default_factory_selects_openai_for_legacy_connections() -> None:
    connection = make_connection()

    client = DefaultCompletionClientFactory().create(connection)

    assert isinstance(client, PytoyOpenAIClient)
    assert client.connection is connection


def test_default_factory_selects_litellm_only_when_requested(monkeypatch) -> None:
    class FakeLiteLLMClient:
        def __init__(self, connection, **kwargs):
            self.connection = connection
            self.kwargs = kwargs

    package = ModuleType("pytoy_llm.completion_clients.litellm_client")
    package.PytoyLiteLLMClient = FakeLiteLLMClient  # type: ignore
    monkeypatch.setitem(sys.modules, package.__name__, package)
    connection = make_connection("litellm")

    client = DefaultCompletionClientFactory().create(connection)

    assert isinstance(client, FakeLiteLLMClient)
    assert client.connection is connection


def test_default_factory_explains_missing_litellm_extra(monkeypatch) -> None:
    original_import = builtins.__import__

    def import_without_litellm(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pytoy_llm.completion_clients.litellm_client":
            raise ModuleNotFoundError("No module named 'litellm'", name="litellm")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_litellm)

    with pytest.raises(ImportError, match=r"pytoy-llm\[litellm\]"):
        DefaultCompletionClientFactory().create(make_connection("litellm"))
