import pytest
from pydantic_ai import ModelRequest, ModelResponse, ToolReturnPart
from pydantic_ai.usage import RequestUsage, RunUsage

from pytoy_llm.models import LLMMessage, LLMTokens
from pytoy_llm.models.parts import ContentPart, ToolResultPart
from pytoy_llm.pydantic_agent.adapter import (
    PydanticAIMessageAdapter,
    _run_usage_to_llm_tokens,
)


def test_request_content_part_is_not_converted() -> None:
    message = LLMMessage.from_parts(
        [
            ContentPart(
                role="user",
                content=[{"type": "text", "text": "Describe this image."}],
            )
        ],
        kind="request",
    )

    with pytest.raises(ValueError, match="cannot be converted"):
        PydanticAIMessageAdapter().to_native(message)


def test_request_tool_result_part_converts_to_tool_return() -> None:
    message = LLMMessage.from_parts(
        [ToolResultPart(call_id="call-1", tool_name="weather", result='{"temperature": 20}')],
        kind="request",
    )

    native = PydanticAIMessageAdapter().to_native(message)

    assert isinstance(native, ModelRequest)
    assert len(native.parts) == 1
    part = native.parts[0]
    assert isinstance(part, ToolReturnPart)
    assert part.tool_call_id == "call-1"
    assert part.tool_name == "weather"
    assert part.content == '{"temperature": 20}'


def test_tool_return_converts_to_tool_result_part() -> None:
    native = ModelRequest(
        parts=[
            ToolReturnPart(
                tool_name="weather",
                tool_call_id="call-1",
                content='{"temperature": 20}',
            )
        ]
    )

    message = PydanticAIMessageAdapter().from_native(native)

    assert len(message.parts) == 1
    part = message.parts[0]
    assert isinstance(part, ToolResultPart)
    assert part.call_id == "call-1"
    assert part.tool_name == "weather"
    assert part.result == '{"temperature": 20}'


def test_run_usage_marks_missing_if_any_response_has_no_token_counts() -> None:
    usage = RunUsage(requests=2, input_tokens=10, output_tokens=4)
    messages = [
        ModelResponse(parts=[], usage=RequestUsage(input_tokens=10, output_tokens=4)),
        ModelResponse(parts=[]),
    ]

    result = _run_usage_to_llm_tokens(usage, messages)

    assert result == LLMTokens(
        prompt=10,
        completion=4,
        total=14,
        cache_read=0,
        cache_write=0,
        usage_missing=False,
    )
