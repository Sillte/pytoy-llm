import importlib

import pytest

from pytoy_llm.models import LLMTokens
from pytoy_llm.models.connections import Connection


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        None,
    ],
)
def test_litellm_client_completes_with_its_optional_dependency(monkeypatch, usage) -> None:
    litellm = pytest.importorskip("litellm")
    client_module = importlib.import_module("pytoy_llm.completion_clients.litellm_client")
    response = litellm.ModelResponse(
        choices=[
            {
                "message": {"role": "assistant", "content": "hello from LiteLLM"},
                "finish_reason": "stop",
            }
        ],
        model="openai/test-model",
        usage=usage,
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
    if usage is None:
        assert result.meta.tokens == LLMTokens(prompt=0, completion=0, total=0, usage_missing=True)
    else:
        assert result.meta.tokens == LLMTokens(prompt=1, completion=2, total=3)
