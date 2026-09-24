import pytest

from pytoy_llm.json_codecs import CompletionMessagesCodec
from pytoy_llm.models import LLMMessage
from pytoy_llm.models.parts import (
    AnyContentPart,
    OpaquePart,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)


@pytest.fixture
def codec():
    return CompletionMessagesCodec()


def test_text_message_round_trip(codec):
    message = LLMMessage.from_parts([TextPart(role="user", content="Hello")])

    native = codec.to_native(message)

    assert native == [{"role": "user", "content": "Hello"}]
    assert codec.from_native(native[0], kind="request") == message


def test_messages_array_flattens_parts_and_decodes(codec):
    messages = [
        LLMMessage.from_parts([TextPart(role="user", content="Be concise")]),
        LLMMessage.from_parts([TextPart(role="user", content="Hello")]),
    ]

    native = codec.to_native_messages(messages)

    assert native == [
        {"role": "user", "content": "Be concise"},
        {"role": "user", "content": "Hello"},
    ]
    assert codec.from_native_messages(native, kind="request") == messages


def test_tool_call_round_trip(codec):
    message = LLMMessage.from_parts(
        [ToolCallPart(tool_name="weather", call_id="call-1", args={"city": "Tokyo"})],
        kind="response",
    )

    native = codec.to_native(message)

    assert native == [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "weather", "arguments": '{"city": "Tokyo"}'},
                }
            ],
        }
    ]
    decoded = codec.from_native(native[0], kind="response")
    assert decoded.kind == "response"
    assert decoded.parts == [
        ToolCallPart(tool_name="weather", call_id="call-1", args='{"city": "Tokyo"}')
    ]


def test_assistant_text_and_tool_calls_are_combined(codec):
    message = LLMMessage.from_parts(
        [
            TextPart(role="assistant", content="I will check."),
            ToolCallPart(tool_name="weather", call_id="call-1", args={"city": "Tokyo"}),
            ToolCallPart(tool_name="time", call_id="call-2", args=None),
        ],
        kind="response",
    )

    assert codec.to_native(message) == [
        {
            "role": "assistant",
            "content": "I will check.",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "weather", "arguments": '{"city": "Tokyo"}'},
                },
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {"name": "time", "arguments": None},
                },
            ],
        }
    ]


def test_invalid_tool_calls_are_rejected(codec):
    decoded = codec.from_native({"role": "assistant", "tool_calls": "invalid"}, kind="response")

    assert decoded.parts == [OpaquePart(value={"role": "assistant", "tool_calls": "invalid"})]


def test_content_part_array_round_trip(codec):
    message = LLMMessage.from_parts(
        [
            AnyContentPart(
                role="user",
                content=[{"type": "text", "text": "Describe this image."}],
            )
        ],
        kind="request",
    )

    native = codec.to_native(message)

    assert native == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Describe this image."}],
        }
    ]
    assert codec.from_native(native[0], kind="request") == message


def test_tool_result_round_trip(codec):
    message = LLMMessage.from_parts(
        [ToolResultPart(call_id="call-1", content='{"temperature": 20}')],
        kind="response",
    )

    native = codec.to_native(message)

    assert native == [
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": '{"temperature": 20}',
        }
    ]
    assert codec.from_native(native[0], kind="response") == message


def test_unsupported_role_is_preserved_as_opaque(codec):
    native = {"role": "developer", "content": "Use concise answers."}

    decoded = codec.from_native(native, kind="request")

    assert decoded.parts == [OpaquePart(value=native)]
