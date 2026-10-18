from unittest.mock import Mock

import pytest
from litellm import ModelResponse

from pytoy_llm import api
from pytoy_llm.litellm_client.client import PytoyLiteLLMClient
from pytoy_llm.models import CancelToken, LLMCancelException
from pytoy_llm.models.connections import Connection
from pytoy_llm.models.parts import TextPart


def test_completion_forwards_cancel_token_to_llm_client(monkeypatch) -> None:
    cancel_token = CancelToken()
    client = Mock()
    client.return_value = client
    client.completion.return_value = "result"
    monkeypatch.setattr("pytoy_llm.llm_facade.PytoyLiteLLMClient", client)

    result = api.completion("hello", cancel_token=cancel_token)

    assert result == "result"
    assert client.call_args.kwargs["cancel_token"] is cancel_token


def test_litellm_client_raises_cancellation_after_completion(monkeypatch) -> None:
    cancel_token = CancelToken()
    client = PytoyLiteLLMClient(
        Connection(
            model="openai/test-model",
            base_url="https://example.com",
            api_key="test-key",
        ),
        cancel_token=cancel_token,
    )

    def complete(**_kwargs):
        cancel_token.set()
        return ModelResponse(
            choices=[
                {
                    "message": {"role": "assistant", "content": "completed response"},
                    "finish_reason": "stop",
                }
            ],
            model="openai/test-model",
            usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        )

    monkeypatch.setattr("litellm.completion", complete)

    with pytest.raises(LLMCancelException) as exc_info:
        client.completion_with_result("hello", output_type=str)

    messages = exc_info.value.messages
    assert isinstance(messages, tuple)
    assert len(messages) == 2
    assert messages[0].kind == "request"
    assert messages[1].kind == "response"
    assert isinstance(messages[1].parts[0], TextPart)
    assert messages[1].parts[0].content == "completed response"
