from unittest.mock import MagicMock, Mock

from openai.types.chat import ChatCompletion, ParsedChatCompletion
from pydantic import BaseModel

from pytoy_llm.models.connections import Connection
from pytoy_llm.models.llm_events import LLMEventEmitters
from pytoy_llm.models.llm_messages import LLMRequest
from pytoy_llm.models.llm_metas import LLMParam
from pytoy_llm.openai_client import PytoyOpenAIClient


class StructuredOutput(BaseModel):
    answer: str


def make_response(content: str) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }
    )


def test_openai_client_sends_openai_compatible_chat_completion(monkeypatch) -> None:
    response = make_response("hello")
    completion_endpoint = Mock()
    completion_endpoint.create.return_value = response
    fake_client = MagicMock()
    fake_client.__enter__.return_value = fake_client
    fake_client.chat.completions = completion_endpoint
    openai_constructor = Mock(return_value=fake_client)
    monkeypatch.setattr("pytoy_llm.openai_client.client.OpenAI", openai_constructor)
    client = PytoyOpenAIClient(
        Connection(
            model="openai/test-model",
            base_url="https://example.com/v1",
            api_key="test-key",
        ),
        llm_param=LLMParam(
            temperature=0.2,
            max_tokens=7,
            reasoning_effort="low",
            allowed_openai_params=["reasoning_effort"],
        ),
    )

    result = client.completion_with_result(
        LLMRequest.from_prompt(system="Be concise.", user="hello"),
        output_type=str,
    )

    assert result.output == "hello"
    assert result.meta.tokens.prompt == 1
    assert result.meta.finish_reason == "stop"
    openai_constructor.assert_called_once_with(
        api_key="test-key",
        base_url="https://example.com/v1",
    )
    completion_endpoint.create.assert_called_once()
    call = completion_endpoint.create.call_args.kwargs
    assert call["model"] == "test-model"
    assert call["messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "hello"},
    ]
    assert call["temperature"] == 0.2
    assert call["max_tokens"] == 7
    assert call["reasoning_effort"] == "low"
    assert "allowed_openai_params" not in call


def test_openai_client_uses_sdk_parsing_for_structured_output(monkeypatch) -> None:
    response = ParsedChatCompletion[StructuredOutput].model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": '{"answer":"fine"}',
                        "parsed": {"answer": "fine"},
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }
    )
    fake_client = MagicMock()
    fake_client.__enter__.return_value = fake_client
    fake_client.chat.completions.parse.return_value = response
    monkeypatch.setattr("pytoy_llm.openai_client.client.OpenAI", Mock(return_value=fake_client))
    client = PytoyOpenAIClient(
        Connection(model="test-model", base_url="https://example.com/v1", api_key="test-key")
    )

    result = client.completion("hello", output_type=StructuredOutput)

    assert result == StructuredOutput(answer="fine")
    fake_client.chat.completions.parse.assert_called_once()


def test_openai_client_emits_request_and_response_activities(monkeypatch) -> None:
    response = make_response("hello")
    completion_endpoint = Mock()
    completion_endpoint.create.return_value = response
    fake_client = MagicMock()
    fake_client.__enter__.return_value = fake_client
    fake_client.chat.completions = completion_endpoint
    monkeypatch.setattr("pytoy_llm.openai_client.client.OpenAI", Mock(return_value=fake_client))
    emitters = LLMEventEmitters()
    activities = []
    emitters.on_activity.subscribe(activities.append)
    client = PytoyOpenAIClient(
        Connection(model="test-model", base_url="https://example.com/v1", api_key="test-key"),
        event_emitters=emitters,
    )

    client.completion("hello", output_type=str)

    assert [activity.activity_type for activity in activities] == [
        "pre_api_call",
        "post_api_call",
        "response_activity",
    ]
