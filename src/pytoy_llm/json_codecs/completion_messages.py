from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel

from pytoy_llm.models import LLMMessage
from pytoy_llm.models.parts import (
    AnyContentPart,
    OpaquePart,
    Part,
    TextPart,
    ToolCallRequestPart,
    ToolResultPart,
)

CompletionMessage = Mapping[str, Any]


class _AssistantMessageComposer:
    def __init__(self) -> None:
        self._content: str | None = None
        self._tool_calls: list[dict[str, Any]] = []

    def add(self, part: TextPart | ToolCallRequestPart) -> None:
        match part:
            case TextPart():
                self._content = (self._content or "") + part.content
            case ToolCallRequestPart():
                self._tool_calls.append(
                    {
                        "id": part.call_id,
                        "type": "function",
                        "function": {
                            "name": part.tool_name,
                            "arguments": _dump_arguments(part.args),
                        },
                    }
                )

    def build(self) -> dict[str, Any]:
        result: dict[str, Any] = {"role": "assistant"}
        if self._content is not None:
            result["content"] = self._content
        if self._tool_calls:
            result["tool_calls"] = self._tool_calls
        return result


class CompletionMessagesCodec:
    """Convert LLM messages to and from OpenAI Chat Completions messages."""

    def to_native(self, message: LLMMessage) -> list[dict[str, Any]]:
        """Convert one LLM message into its OpenAI message objects."""
        result: list[dict[str, Any]] = []
        assistant_composer: _AssistantMessageComposer | None = None

        for part in message.parts:
            if isinstance(part, TextPart) and part.role == "assistant":
                if assistant_composer is None:
                    assistant_composer = _AssistantMessageComposer()
                assistant_composer.add(part)
            elif isinstance(part, ToolCallRequestPart):
                if assistant_composer is None:
                    assistant_composer = _AssistantMessageComposer()
                assistant_composer.add(part)
            else:
                self._flush_assistant(result, assistant_composer)
                assistant_composer = None
                result.append(self._part_to_native(part))

        self._flush_assistant(result, assistant_composer)
        return result

    @staticmethod
    def _flush_assistant(
        result: list[dict[str, Any]], composer: _AssistantMessageComposer | None
    ) -> None:
        if composer is not None:
            result.append(composer.build())

    def from_native(
        self,
        message: CompletionMessage,
        *,
        kind: Literal["request", "response"],
    ) -> LLMMessage:
        """Convert one OpenAI message object into an LLM message."""
        role = message.get("role")
        parts: list[Part] = []

        if role in {"system", "user", "assistant"}:
            content = message.get("content")
            if isinstance(content, str):
                parts.append(TextPart(role=role, content=content))
            elif isinstance(content, Sequence) and not isinstance(content, str | bytes):
                parts.append(AnyContentPart(role=role, content=list(content)))
            elif content is not None:
                return LLMMessage(kind=kind, parts=[OpaquePart(value=dict(message))])

        tool_calls = message.get("tool_calls")
        if tool_calls is not None:
            if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, str | bytes):
                return LLMMessage(kind=kind, parts=[OpaquePart(value=dict(message))])
            for tool_call in tool_calls:
                try:
                    parts.append(self._tool_call_from_native(tool_call))
                except ValueError:
                    return LLMMessage(kind=kind, parts=[OpaquePart(value=dict(message))])

        if role == "tool":
            call_id = message.get("tool_call_id")
            if isinstance(call_id, str) and "content" in message:
                return LLMMessage(
                    kind=kind,
                    parts=[ToolResultPart(call_id=call_id, content=message["content"])],
                )
            return LLMMessage(kind=kind, parts=[OpaquePart(value=dict(message))])

        if not parts:
            parts.append(OpaquePart(value=dict(message)))

        return LLMMessage(kind=kind, parts=parts)

    def to_native_messages(self, messages: Sequence[LLMMessage]) -> list[dict[str, Any]]:
        """Convert LLM messages into an OpenAI messages array."""
        return sum((self.to_native(message) for message in messages), [])

    def from_native_messages(
        self,
        messages: Sequence[CompletionMessage],
        *,
        kind: Literal["request", "response"],
    ) -> list[LLMMessage]:
        """Convert an OpenAI messages array into LLM messages."""
        return [self.from_native(message, kind=kind) for message in messages]

    @staticmethod
    def _part_to_native(part: Part) -> dict[str, Any]:
        match part:
            case TextPart():
                return {"role": part.role, "content": part.content}
            case AnyContentPart():
                return {"role": part.role, "content": part.content}
            case ToolCallRequestPart():
                return {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": part.call_id,
                            "type": "function",
                            "function": {
                                "name": part.tool_name,
                                "arguments": _dump_arguments(part.args),
                            },
                        }
                    ],
                }
            case ToolResultPart():
                return {
                    "role": "tool",
                    "tool_call_id": part.call_id,
                    "content": part.content,
                }
            case OpaquePart():
                if isinstance(part.value, Mapping):
                    return dict(part.value)
                if isinstance(part.value, BaseModel):
                    return part.value.model_dump()
                raise ValueError(f"{part} cannot be converted to a completion message.")
        raise TypeError(f"Unsupported part: {part!r}")

    @staticmethod
    def _tool_call_from_native(tool_call: Any) -> ToolCallRequestPart:
        if not isinstance(tool_call, Mapping):
            raise ValueError("Completion message tool_calls must contain objects.")
        function = tool_call.get("function")
        if not isinstance(function, Mapping):
            raise ValueError("Completion tool call function must be an object.")
        try:
            call_id = tool_call["id"]
            tool_name = function["name"]
            arguments = function.get("arguments")
        except KeyError as error:
            raise ValueError("Completion tool call is missing a required field.") from error
        return ToolCallRequestPart(
            tool_name=str(tool_name),
            call_id=str(call_id),
            args=arguments,
        )


def _dump_arguments(arguments: str | dict[str, Any] | None) -> str | None:
    if isinstance(arguments, dict):
        return json.dumps(arguments)
    return arguments
