import importlib

import pytest

from pytoy_llm.models.connections import Connection


def test_litellm_client_completes_with_its_optional_dependency(monkeypatch) -> None:
    litellm = pytest.importorskip("litellm")
    client_module = importlib.import_module("pytoy_llm.litellm_client")
    response = litellm.ModelResponse(
        choices=[
            {
                "message": {"role": "assistant", "content": "hello from LiteLLM"},
                "finish_reason": "stop",
            }
        ],
        model="openai/test-model",
        usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    )
    monkeypatch.setattr(litellm, "completion", lambda **_kwargs: response)
    client = client_module.PytoyLiteLLMClient(
        Connection(
            model="openai/test-model",
            base_url="https://example.com/v1",
            api_key="test-key",
            completion_backend="litellm",
        )
    )

    result = client.completion_with_result("hello", output_type=str)

    assert result.output == "hello from LiteLLM"
    assert result.meta.tokens.prompt == 1
