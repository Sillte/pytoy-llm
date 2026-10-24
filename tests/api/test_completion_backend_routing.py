import builtins
import sys
from types import ModuleType

import pytest

from pytoy_llm.llm_facade import LLMFacade
from pytoy_llm.models.connections import Connection


def make_litellm_connection() -> Connection:
    return Connection(
        model="openai/test-model",
        base_url="https://example.com/v1",
        api_key="test-key",
        completion_backend="litellm",
    )


def test_facade_uses_litellm_only_when_selected(monkeypatch) -> None:
    calls = {}

    class FakeLiteLLMClient:
        def __init__(self, connection, **kwargs):
            calls["connection"] = connection
            calls["kwargs"] = kwargs

        def completion(self, request, output_type):
            return "from-litellm"

    package = ModuleType("pytoy_llm.litellm_client")
    package.PytoyLiteLLMClient = FakeLiteLLMClient
    monkeypatch.setitem(sys.modules, "pytoy_llm.litellm_client", package)
    connection = make_litellm_connection()

    result = LLMFacade(connection=connection).completion("hello", output_type=str)

    assert result == "from-litellm"
    assert calls["connection"] is connection


def test_facade_explains_missing_litellm_extra(monkeypatch) -> None:
    original_import = builtins.__import__

    def import_without_litellm(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pytoy_llm.litellm_client":
            raise ModuleNotFoundError("No module named 'litellm'", name="litellm")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_litellm)

    with pytest.raises(ImportError, match=r"pytoy-llm\[litellm\]"):
        LLMFacade(connection=make_litellm_connection()).completion("hello", output_type=str)
