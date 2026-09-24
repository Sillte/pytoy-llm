from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import chain
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, Field

from pytoy_llm.models.llm_metas import LLMOutputMeta
from pytoy_llm.models.parts import Part, PartAdapter, TextPart
from pytoy_llm.models.system_prompt import SystemPrompt


class LLMMessage(BaseModel, frozen=True):
    """
    Represents a single interaction unit exchanged with an LLM.
    Unlike OpenAI ChatMessage, this object is not a single role/content pair.
    """

    kind: Literal["request", "response"] = "request"
    parts: Sequence[Part]

    @classmethod
    def from_prompt(
        cls,
        user: str | None = None,
        parts: Sequence[Part] | None = None,
    ) -> Self:
        parts = parts or []
        parts = list(parts)
        if user is not None:
            parts.append(TextPart(role="user", content=user))
        return cls(kind="request", parts=parts)

    @classmethod
    def from_records(
        cls,
        records: Sequence[Mapping[str, Any]],
        kind: Literal["request", "response"] | None = None,
    ) -> Self:
        parts = [PartAdapter.validate_python(record) for record in records]
        return cls.from_parts(parts, kind)

    @classmethod
    def from_parts(
        cls, parts: Sequence[Part], kind: Literal["request", "response"] | None = None
    ) -> Self:
        def _infer_kind(parts: Sequence[Part]) -> Literal["request", "response"]:
            for part in parts:
                if isinstance(part, TextPart) and part.role in {"user", "system"}:
                    return "request"
            return "response"

        kind = kind or _infer_kind(parts)
        return cls(kind=kind, parts=parts)

    @classmethod
    def from_any(
        cls,
        arg: LLMMessageLike,
    ) -> Self:
        if isinstance(arg, str):
            try:
                result = cls.model_validate_json(arg)
            except ValueError:
                return cls.from_prompt(user=arg)
            else:
                return result
        return cls.model_validate(arg)

    @classmethod
    def chat(cls, content: str) -> Self:
        return cls.from_prompt(user=content)

    @classmethod
    def merge(
        cls,
        messages: Sequence["LLMMessage"],
    ) -> "LLMMessage":
        if not messages:
            raise ValueError("Cannot merge empty messages.")

        kind = messages[0].kind

        if any(message.kind != kind for message in messages):
            raise ValueError("Cannot merge messages with different kinds.")

        return cls(
            kind=kind,
            parts=list(chain.from_iterable(message.parts for message in messages)),
        )


type LLMMessageLike = LLMMessage | str | Sequence[Mapping[str, Any]] | Mapping[str, Any]


class LLMRequest(BaseModel, frozen=True):
    """A complete LLM input, including an optional system prompt and history."""

    system_prompt: SystemPrompt | None = None
    messages: Sequence[LLMMessage] = Field(default_factory=list)

    @classmethod
    def from_prompt(
        cls,
        system: SystemPrompt | str | None = None,
        user: str | None = None,
        system_as_history: bool = False,
    ) -> Self:
        system_prompt = (
            SystemPrompt.from_any(content=system, as_history=system_as_history)
            if system is not None
            else None
        )
        if user is not None:
            llm_messages = [LLMMessage(kind="request", parts=[TextPart(role="user", content=user)])]
        else:
            llm_messages = []
        return cls(system_prompt=system_prompt, messages=llm_messages)

    @classmethod
    def from_any(
        cls, arg: LLMRequestLike, *, system_prompt: SystemPrompt | str | None = None
    ) -> Self:
        system_prompt = SystemPrompt.from_any(system_prompt) if system_prompt is not None else None
        if isinstance(arg, cls):
            if system_prompt is not None:
                raise ValueError("Cannot provide `system_prompt` when `arg` is already LLMRequest.")
            return arg

        elif isinstance(arg, LLMMessage):
            return cls(
                system_prompt=system_prompt,
                messages=[arg],
            )
        elif isinstance(arg, str):
            return cls.from_prompt(system=system_prompt, user=arg)
        elif isinstance(arg, Mapping):
            return cls(
                system_prompt=system_prompt,
                messages=[LLMMessage.from_any(arg)],
            )
        elif isinstance(arg, Sequence):
            return cls(
                system_prompt=system_prompt,
                messages=[LLMMessage.from_any(elem) for elem in arg],
            )
        else:
            raise ValueError(f"Cannot convert {arg} to LLMRequest.")


# TODO: Consider SystemPromptPart is acceptable when the multiple messages define them.
type LLMRequestLike = LLMRequest | LLMMessageLike


class LLMResult[T: BaseModel | str](BaseModel, frozen=True):
    meta: Annotated[LLMOutputMeta, Field(description="Meta data of the response of LLM")]
    output: Annotated[T, Field(description="The main text content from LLM")]
    messages: Annotated[Sequence[LLMMessage], Field(description="Message sequence of LLM")]
