from unittest.mock import MagicMock, Mock

import pytest
from openai.types.chat import ChatCompletion

from pytoy_llm import api
from pytoy_llm.completion_clients.openai_client import PytoyOpenAIClient
from pytoy_llm.models import CancelToken, LLMCancelException
from pytoy_llm.models.connections import Connection
from pytoy_llm.models.parts import TextPart


def test_completion_forwards_cancel_token_to_facade(monkeypatch) -> None:
    cancel_token = CancelToken()
    facade = Mock()
    facade.completion.return_value = "result"
    monkeypatch.setattr(api, "LLMFacade", Mock(return_value=facade))

    result = api.completion("hello", cancel_token=cancel_token)

    assert result == "result"
    assert facade.completion.call_args.kwargs["cancel_token"] is cancel_token


def test_openai_client_raises_cancellation_after_completion(monkeypatch) -> None:
    cancel_token = CancelToken()
    client = PytoyOpenAIClient(
        Connection(
            model="openai/test-model",
            base_url="https://example.com/v1",
            api_key="test-key",
        ),
        cancel_token=cancel_token,
    )
    response = ChatCompletion.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "completed response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }
    )
    fake_client = MagicMock()
    fake_client.__enter__.return_value = fake_client
    fake_client.chat.completions.create.return_value = response
    monkeypatch.setattr(
        "pytoy_llm.completion_clients.openai_client.client.OpenAI", Mock(return_value=fake_client)
    )
    fake_client.chat.completions.parse.side_effect = AssertionError("unexpected structured parse")

    def complete(**_kwargs):
        cancel_token.set()
        return response

    fake_client.chat.completions.create.side_effect = complete

    with pytest.raises(LLMCancelException) as exc_info:
        client.completion_with_result("hello", output_type=str)

    messages = exc_info.value.messages
    assert isinstance(messages, tuple)
    assert len(messages) == 2
    assert messages[0].kind == "request"
    assert messages[1].kind == "response"
    assert isinstance(messages[1].parts[0], TextPart)
    assert messages[1].parts[0].content == "completed response"
