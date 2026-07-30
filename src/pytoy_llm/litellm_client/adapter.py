from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, assert_never, cast

import litellm
from litellm import ModelResponse
from pydantic import BaseModel

from pytoy_llm.models import (
    Part,
    PartAdapter,
)
from pytoy_llm.models.llm_messages import LLMMessage, LLMResult
from pytoy_llm.models.llm_metas import LLMOutputMeta, LLMParam, LLMTokens
from pytoy_llm.models.parts import OpaquePart, TextPart


class LiteLLMPartConverter:
    def __init__(self) -> None: ...

    def to_native(self, part: Part) -> Mapping[str, Any]:
        match part:
            case TextPart():
                return part.model_dump()
            case OpaquePart():
                if isinstance(part.value, Mapping):
                    return part.value
                elif isinstance(part.value, BaseModel):
                    return part.value.model_dump()
                raise ValueError(f"{part} cannot be recognized.")
            case _:
                assert_never(part)

    def from_native(self, native_part: Mapping[str, Any]) -> Part:
        try:
            return PartAdapter.validate_python(native_part)
        except ValueError:
            return OpaquePart(value=native_part)


class LiteLLMMessageAdapter:
    def __init__(self) -> None:
        self._part_converter = LiteLLMPartConverter()

    def to_native(
        self,
        message: LLMMessage,
    ) -> Sequence[Mapping[str, Any]]:
        return [self._part_converter.to_native(part) for part in message.parts]

    def from_native(self, native_records: Sequence[Mapping[str, Any]], kind: Literal["response", "request"]) -> LLMMessage:
        parts = [self._part_converter.from_native(elem) for elem in native_records]
        return LLMMessage(kind=kind, parts=parts)

    def to_llm_model[T: BaseModel | str](
        self, input_messages: Sequence[LLMMessage], llm_response: ModelResponse, output_type: type[T]
    ) -> LLMResult[T]:

        response = cast(litellm.TextCompletionResponse, llm_response)
        usage = response.usage
        if usage is None:
            raise ValueError("Response is strange.")
        tokens = LLMTokens(prompt=usage.prompt_tokens, completion=usage.completion_tokens, total=usage.total_tokens)
        finish_reason = response.choices[0].finish_reason
        meta = LLMOutputMeta(tokens=tokens, finish_reason=finish_reason, llm_calls=1)

        choices = cast(litellm.Choices, response.choices)
        choice = choices[0]
        content = choice.message.content
        content = cast(str, content)

        if output_type is str:
            content = str(content)
        else:
            t_output_type = cast(T, output_type)
            content = cast(T, t_output_type.model_validate_json(content))  # type:ignore
        output_message = self.from_native([llm_response.json()], kind="response")
        messages = [*input_messages, output_message]
        return cast(LLMResult[T], LLMResult(output=content, meta=meta, messages=messages))


class LLMParamConverter:
    def __init__(self) -> None: ...

    def to_litellm_kwargs(self, llm_param: LLMParam) -> dict:
        return llm_param.model_dump(exclude_none=True)
