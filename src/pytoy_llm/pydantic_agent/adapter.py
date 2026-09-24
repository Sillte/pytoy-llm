from typing import Mapping, assert_never

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
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai import TextPart as PydanticTextPart
from pydantic_ai.settings import ThinkingLevel
from pydantic_ai.usage import UsageLimits

from pytoy_llm.models import Part as LLMPart
from pytoy_llm.models.agent_metas import UsageLimit as PytoyUsageLimit
from pytoy_llm.models.llm_messages import LLMMessage, LLMResult
from pytoy_llm.models.llm_metas import LLMOutputMeta, LLMParam, LLMTokens, ReasoningEffort
from pytoy_llm.models.parts import (
    AnyContentPart,
    OpaquePart,
    SystemPromptHistoryPart,
    ToolResultPart,
)
from pytoy_llm.models.parts import TextPart as LLMTextPart
from pytoy_llm.models.parts import ToolCallPart as LLMToolCallPart


class RequestPartConverter:
    def __init__(self) -> None: ...

    def to_native(self, part: LLMPart) -> ModelRequestPart:
        match part:
            case LLMTextPart():
                if part.role == "user":
                    return UserPromptPart(content=part.content)
            case AnyContentPart():
                raise ValueError(f"`{part}` cannot be converted to a ModelRequestPart")
            case SystemPromptHistoryPart():
                return SystemPromptPart(content=part.content)
            case OpaquePart():
                return part.value
            case LLMToolCallPart():
                raise TypeError(f"{part=}")
            case ToolResultPart():
                if part.tool_name is None:
                    raise ValueError(f"ToolResultPart must have `tool_name`.{part=}")
                return ToolReturnPart(
                    tool_name=part.tool_name,
                    tool_call_id=part.call_id,
                    content=part.content,
                )
            case _:
                assert_never(part)
        raise ValueError(f"`{part}` cannot be converted to a ModelRequestPart")

    def from_native(self, part: ModelRequestPart) -> LLMPart:
        match part:
            case UserPromptPart():
                return LLMTextPart(role="user", content=str(part.content))
            case SystemPromptPart():
                return SystemPromptHistoryPart(content=part.content)
            case ToolReturnPart():
                return ToolResultPart(
                    call_id=part.tool_call_id,
                    content=part.content,
                    tool_name=part.tool_name,
                )
            case _:
                return OpaquePart(value=part)


class ResponsePartConverter:
    def __init__(self) -> None: ...

    def to_native(self, part: LLMPart) -> ModelResponsePart:
        match part:
            case LLMTextPart():
                return PydanticTextPart(content=part.content)
            case LLMToolCallPart():
                return ToolCallPart(
                    tool_name=part.tool_name, tool_call_id=part.call_id, args=part.args
                )
            case ToolResultPart():
                raise TypeError(f"{part=}")

            case AnyContentPart():
                raise ValueError(f"`{part}` cannot be converted to a ModelResponsePart")
            case SystemPromptHistoryPart():
                raise ValueError(f"`{part}` cannot be converted to a ModelResponsePart")
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
                if isinstance(part.args, Mapping):
                    args = dict(part.args)
                else:
                    args = part.args
                return LLMToolCallPart(
                    args=args, call_id=part.tool_call_id, tool_name=part.tool_name
                )
            case ThinkingPart():
                return OpaquePart(value=part)
            case FilePart():
                return OpaquePart(value=part)
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
                parts = [
                    self._request_part_converter.from_native(part) for part in model_message.parts
                ]
                return LLMMessage(kind="request", parts=parts)
            case "response":
                parts = [
                    self._response_part_converter.from_native(part) for part in model_message.parts
                ]
                return LLMMessage(kind="response", parts=parts)
            case _:
                assert_never(model_message.kind)

    def to_llm_output[T: str | BaseModel](self, run_result: AgentRunResult[T]) -> LLMResult[T]:
        messages = [self.from_native(elem) for elem in run_result.all_messages()]

        usage = run_result.usage
        prompt = usage.input_tokens
        completion = usage.output_tokens
        tokens = LLMTokens(prompt=prompt, completion=completion, total=prompt + completion)
        meta = LLMOutputMeta(tokens=tokens, finish_reason=None, llm_calls=usage.requests)
        return LLMResult(output=run_result.output, meta=meta, messages=messages)


class LLMParamConverter:
    def __init__(self) -> None: ...

    def to_model_settings(self, llm_param: LLMParam) -> ModelSettings:
        raw = llm_param.model_dump(exclude_none=True)

        # Meaning conversions.
        if "reasoning_effort" in raw:
            raw["thinking"] = self._to_thinking(raw.pop("reasoning_effort"))

        result = ModelSettings(**raw)
        return result

    def _to_thinking(
        self,
        value: ReasoningEffort,
    ) -> ThinkingLevel:
        match value:
            case "none":
                return False
            case _:
                return value


class UsageLimitConverter:
    def __init__(self) -> None: ...

    def to_usage_limits(self, usage_limit: PytoyUsageLimit) -> UsageLimits:
        return UsageLimits(
            total_tokens_limit=usage_limit.max_total_tokens, request_limit=usage_limit.max_requests
        )
