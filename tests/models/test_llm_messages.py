import pytest
from litellm import ModelResponse
from pydantic import BaseModel

from pytoy_llm.models.llm_messages import LLMMessage
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
    res = LLMMessage.from_prompt(user="hello", system="evening")
    assert len(res.parts) == 2
    assert isinstance(res.parts[0], TextPart)
    assert res.parts[0].role == "system"
    assert res.parts[0].content == "evening"
    assert isinstance(res.parts[1], TextPart)
    assert res.parts[1].role == "user"
    assert res.parts[1].content == "hello"


def test_merge_messages():
    first = LLMMessage.from_parts(
        [
            TextPart(
                role="system",
                content="You are helpful.",
            )
        ]
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
