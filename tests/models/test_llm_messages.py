import pytest
from litellm import ModelResponse
from pydantic import BaseModel

from pytoy_llm.models.llm_messages import LLMMessage, LLMRequest
from pytoy_llm.models.parts import TextPart


# --- Mock Data ---
@pytest.fixture
def mock_response():
    # LiteLLMのModelResponseを模したデータ
    response_data = {
        "choices": [
            {
                "message": {"role": "assistant", "content": '{"answer": "fine"}'},
                "finish_reason": "stop",
            }
        ],
        "model": "gemini/gemini-2.0-flash",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    return ModelResponse(**response_data)


class DummyStructuredModel(BaseModel):
    answer: str


# --- InputConverter Tests ---


def test_input_converter_str():
    res = LLMMessage.from_prompt(user="hello")
    assert len(res.parts) == 1
    assert isinstance(res.parts[0], TextPart)
    assert res.parts[0].role == "user"
    assert res.parts[0].content == "hello"


def test_merge_messages():
    first = LLMMessage.from_parts(
        [
            TextPart(
                role="assistant",
                content="You are helpful.",
            )
        ],
        kind="request",
    )

    second = LLMMessage.from_parts(
        [
            TextPart(
                role="user",
                content="Hello",
            )
        ],
        kind="request",
    )

    merged = LLMMessage.merge([first, second])

    assert merged.kind == "request"
    assert len(merged.parts) == 2
    assert isinstance(merged.parts[0], TextPart)
    assert isinstance(merged.parts[1], TextPart)
    assert merged.parts[0].content == "You are helpful."
    assert merged.parts[1].content == "Hello"


def test_request_from_mapping_wraps_single_message() -> None:
    request = LLMRequest.from_any(
        {"kind": "request", "parts": [{"role": "user", "content": "Hello"}]}
    )

    assert len(request.messages) == 1
    part = request.messages[0].parts[0]
    assert isinstance(part, TextPart)
    assert part.content == "Hello"


def test_request_rejects_any_system_prompt_override() -> None:
    request = LLMRequest.from_prompt(user="Hello")

    with pytest.raises(ValueError, match="already LLMRequest"):
        LLMRequest.from_any(request, system_prompt="")
