from __future__ import annotations

from typing import Any, cast

from openai.types.chat import ChatCompletion, ParsedChatCompletion
from pydantic import BaseModel

from pytoy_llm.json_codecs import CompletionMessagesCodec
from pytoy_llm.models.llm_messages import LLMRequest, LLMResult
from pytoy_llm.models.llm_metas import LLMOutputMeta, LLMParam, LLMTokens
from pytoy_llm.models.parts import SystemPromptHistoryPart

OpenAICompletion = ChatCompletion | ParsedChatCompletion[Any]


class OpenAITransactionAdapter:
    def __init__(self) -> None:
        self._codec = CompletionMessagesCodec()

    def to_native(self, request: LLMRequest) -> list[dict[str, Any]]:
        messages = sum((self._codec.to_native(message) for message in request.messages), [])
        if request.system_prompt:
            messages.insert(0, {"role": "system", "content": request.system_prompt.content})
        return messages

    def to_llm_model[T: BaseModel | str](
        self,
        request: LLMRequest,
        llm_response: OpenAICompletion,
        output_type: type[T],
    ) -> LLMResult[T]:
        choice = llm_response.choices[0]
        message = choice.message
        content = message.content
        if output_type is str:
            output: T = cast(T, str(content))
        else:
            model_type = cast(type[BaseModel], output_type)
            parsed = getattr(message, "parsed", None)
            if isinstance(parsed, model_type):
                output = cast(T, parsed)
            elif content is not None:
                output = cast(T, model_type.model_validate_json(content))
            else:
                raise ValueError("Response did not contain structured output.")

        usage = llm_response.usage
        if usage is None:
            raise ValueError("Response is strange: token usage is missing.")
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        cache_read = getattr(prompt_details, "cached_tokens", None)
        if cache_read is None:
            cache_read = getattr(usage, "cache_read_input_tokens", None)
        tokens = LLMTokens(
            prompt=usage.prompt_tokens,
            completion=usage.completion_tokens,
            total=usage.total_tokens,
            cache_read=cache_read,
            cache_write=getattr(prompt_details, "cache_write_tokens", None)
            or getattr(usage, "cache_creation_input_tokens", None),
        )
        meta = LLMOutputMeta(
            tokens=tokens,
            finish_reason=choice.finish_reason,
            llm_calls=1,
        )
        output_message = self._codec.from_native(
            message.model_dump(exclude_none=True), kind="response"
        )

        if request.system_prompt and request.system_prompt.as_history and request.messages:
            last_message = list(request.messages)[-1]
            parts = [
                SystemPromptHistoryPart(content=request.system_prompt.content),
                *last_message.parts,
            ]
            messages = [
                *request.messages[:-1],
                last_message.model_copy(update={"parts": parts}),
                output_message,
            ]
        else:
            messages = [*request.messages, output_message]

        return cast(LLMResult[T], LLMResult(output=output, meta=meta, messages=messages))


class LLMParamConverter:
    def to_openai_kwargs(self, llm_param: LLMParam) -> dict[str, Any]:
        result = llm_param.model_dump(exclude_none=True)
        result.pop("allowed_openai_params", None)
        return result
