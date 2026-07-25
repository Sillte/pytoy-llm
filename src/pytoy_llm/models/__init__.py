from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    Self,
    Union,
    get_args,
    get_origin,
)

if TYPE_CHECKING:
    from pydantic_ai import ModelSettings

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter, field_validator

StrictStr = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]


class Connection(BaseModel, frozen=True):
    model: Annotated[
        StrictStr,
        Field(
            description="Model Name of LLM",
            examples=["gemini/gemini-2.0-flash", "gpt-4o"],
        ),
    ]
    base_url: Annotated[
        StrictStr,
        Field(
            description="Endpoint for LLM.",
            examples=["https://"],
        ),
    ]
    api_key: Annotated[
        StrictStr,
        Field(description="Credential Information for using LLM.", examples=["SECRET-KEY"]),
    ]

    @field_validator("base_url", mode="before")
    @classmethod
    def normalize_base_url(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip("/")
        else:
            return value


type Role = Literal["system", "user", "assistant"]


class BasePart(BaseModel, frozen=True): ...


class TextPart(BasePart, frozen=True):
    role: Role
    content: str


class OpaquePart(BasePart, frozen=True):
    value: Any


Part = TextPart | OpaquePart
PartAdapter = TypeAdapter(Part)


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
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        parts: Sequence[Part] | None = None,
    ) -> Self:
        parts = parts or []
        parts = list(parts)
        if system_prompt:
            parts.append(TextPart(role="system", content=system_prompt))
        if user_prompt:
            parts.append(TextPart(role="user", content=user_prompt))
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
    def from_parts(cls, parts: Sequence[Part], kind: Literal["request", "response"] | None = None) -> Self:
        def _infer_kind(parts: Sequence[Part]) -> Literal["request", "response"]:
            for part in parts:
                if isinstance(part, TextPart) and part.role in {"user", "system"}:
                    return "request"
            return "response"

        kind = kind or _infer_kind(parts)
        return cls(kind=kind, parts=parts)

    @classmethod
    def from_any(cls, arg: str | Sequence[Part] | Mapping[str, Any] | Sequence[Mapping[str, Any]] | Self) -> Self:
        if isinstance(arg, str):
            try:
                result = cls.model_validate_json(arg)
            except ValueError:
                return cls.from_prompt(user_prompt=arg)
            else:
                return result
        return cls.model_validate(arg)

    @classmethod
    def to_messages(cls, arg: str | Self | Sequence[Self] | Sequence[Mapping[str, Any]]) -> Sequence[Self]:
        if isinstance(arg, str):
            return [cls.from_prompt(user_prompt=arg)]
        elif isinstance(arg, LLMMessage):
            return [arg]
        if not arg:
            raise ValueError("Emypy messages are not acceptable.")
        return [cls.from_any(elem) for elem in arg]

    @classmethod
    def chat(cls, content: str) -> Self:
        return cls.from_prompt(user_prompt=content)


class LLMMessageHistory(BaseModel, frozen=True):
    items: Annotated[Sequence[LLMMessage], Field(description="")]


class LLMTokens(BaseModel, frozen=True):
    prompt: int
    completion: int
    total: int
    model_config = ConfigDict(extra="allow")


class LLMOutputMeta(BaseModel, frozen=True):
    tokens: LLMTokens
    llm_calls: int = 1
    finish_reason: str | None = None


class LLMOutputModel[T: BaseModel | str](BaseModel, frozen=True):
    meta: Annotated[LLMOutputMeta, Field(description="Meta data of the response of LLM")]
    output: Annotated[T, Field(description="The main text content from LLM")]
    messages: Annotated[Sequence[LLMMessage], Field(description="Message sequence of LLM")]


type ResultType = Literal["pytoy-result", "native-result", "output"]


class LLMConfig(BaseModel, frozen=True):
    temperature: float | None = None
    max_tokens: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None

    def to_litellm_kwargs(self) -> dict:
        return self.model_dump(exclude_none=True)

    def to_pydantic_model_settings(self) -> ModelSettings:
        from pydantic_ai import ModelSettings

        return ModelSettings(**self.model_dump(exclude_none=True))


class LLMTool(BaseModel):
    impl: Annotated[Callable, Field(description="Implementation of the tool.")]
    doc: str | None = None

    def to_pydantic_tool(self) -> Callable:
        if self.doc is not None:
            try:
                self.impl.__doc__ = self.doc
            except Exception:
                pass
        return self.impl

    @field_validator("impl")
    def _check_callable(cls, value: Callable) -> Callable:
        def _is_allowed_type(tp: Any) -> bool:
            if isinstance(tp, type) and issubclass(tp, BaseModel):
                return True
            if tp in (str, int, float, bool):
                return True

            origin = get_origin(tp)
            if origin in (list, tuple, Sequence, Literal, Union):
                return all(_is_allowed_type(arg) for arg in get_args(tp))
            elif origin in (dict, Mapping):
                k, v = get_args(tp)
                return _is_allowed_type(k) and _is_allowed_type(v)
            elif origin is not None:
                return _is_allowed_type(origin)
            return False

        if not callable(value):
            raise TypeError(f"`{value=}` is not callable.")
        sig = inspect.signature(value)
        for param in sig.parameters.values():
            if param.annotation is inspect._empty:
                raise TypeError("Tool parameters must be type-annotated")
            if not _is_allowed_type(param.annotation):
                raise TypeError(f"Unsupported parameter type: {param.annotation}")

        if sig.return_annotation is inspect._empty:
            raise TypeError("Tool must have return type annotation")
        return value
