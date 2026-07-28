from typing import assert_never

from pydantic import BaseModel
from pydantic_ai import (
    AgentRunResult,
    FilePart,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    ModelSettings,
    SystemPromptPart,
    ThinkingPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai import TextPart as PydanticTextPart

from pytoy_llm.models import Part as LLMPart
from pytoy_llm.models.llm_messages import LLMMessage, LLMResult
from pytoy_llm.models.llm_metas import LLMOutputMeta, LLMParam, LLMTokens
from pytoy_llm.models.parts import OpaquePart
from pytoy_llm.models.parts import TextPart as LLMTextPart


class RequestPartConverter:
    def __init__(self) -> None: ...

    def to_native(self, part: LLMPart) -> ModelRequestPart:
        match part:
            case LLMTextPart():
                if part.role == "user":
                    return UserPromptPart(content=part.content)
                elif part.role == "system":
                    return SystemPromptPart(content=part.content)
            case OpaquePart():
                return part.value
            case _:
                assert_never(part)
        raise ValueError(f"`{part}` cannot be converted to a ModelRequestPart")

    def from_native(self, part: ModelRequestPart) -> LLMPart:
        match part:
            case UserPromptPart():
                return LLMTextPart(role="user", content=str(part.content))
            case SystemPromptPart():
                return LLMTextPart(role="system", content=part.content)
            case _:
                return OpaquePart(value=part)


class ResponsePartConverter:
    def __init__(self) -> None: ...

    def to_native(self, part: LLMPart) -> ModelResponsePart:
        match part:
            case LLMTextPart():
                return PydanticTextPart(content=part.content)
            case OpaquePart():
                return part.value
            case _:
                assert_never(part)
        raise ValueError(f"`{part}` cannot be converted to a ModelResponsePart")

    def from_native(self, part: ModelResponsePart) -> LLMPart:
        match part:
            case PydanticTextPart():
                return LLMTextPart(content=part.content, role="assistant")
            case ToolCallPart():
                # TODO: Should be revised
                return LLMTextPart(content=str(part.args), role="assistant")
            case ThinkingPart():
                # TODO: Should be revised
                return LLMTextPart(content=part.content, role="assistant")
            case FilePart():
                # TODO: Should be revised
                return LLMTextPart(content=str(part.content), role="assistant")
            case _:
                return OpaquePart(value=part)


class PydanticAIMessageAdapter:
    def __init__(self):
        self._request_part_converter = RequestPartConverter()
        self._response_part_converter = ResponsePartConverter()

    def to_native(
        self,
        message: LLMMessage,
    ) -> ModelMessage:
        match message.kind:
            case "request":
                parts = [self._request_part_converter.to_native(part) for part in message.parts]
                return ModelRequest(parts=parts)

            case "response":
                parts = [self._response_part_converter.to_native(part) for part in message.parts]
                return ModelResponse(parts=parts)
            case _:
                assert_never(message.kind)

    def from_native(self, model_message: ModelMessage) -> LLMMessage:
        match model_message.kind:
            case "request":
                parts = [self._request_part_converter.from_native(part) for part in model_message.parts]
                return LLMMessage(kind="request", parts=parts)
            case "response":
                parts = [self._response_part_converter.from_native(part) for part in model_message.parts]
                return LLMMessage(kind="response", parts=parts)
            case _:
                assert_never(model_message.kind)

    def to_llm_output[T: str | BaseModel](self, run_result: AgentRunResult[T]) -> LLMResult[T]:
        messages = [self.from_native(elem) for elem in run_result.all_messages()]

        usage = run_result.usage
        prompt = usage.input_tokens
        completion = usage.output_tokens
        tokens = LLMTokens(prompt=prompt, completion=completion, total=prompt + completion)  # NOTE: ....? really?
        meta = LLMOutputMeta(tokens=tokens, finish_reason=None, llm_calls=usage.requests)
        return LLMResult(output=run_result.output, meta=meta, messages=messages)


class LLMParamConverter:
    def __init__(self) -> None: ...

    def to_model_settings(self, llm_param: LLMParam) -> ModelSettings:
        return ModelSettings(**llm_param.model_dump(exclude_none=True))
