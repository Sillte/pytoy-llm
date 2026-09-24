from __future__ import annotations

from typing import Any, Literal, cast

import litellm
from litellm import ModelResponse
from pydantic import BaseModel

from pytoy_llm.json_codecs import CompletionMessagesCodec
from pytoy_llm.models.llm_messages import LLMMessage, LLMRequest, LLMResult
from pytoy_llm.models.llm_metas import LLMOutputMeta, LLMParam, LLMTokens
from pytoy_llm.models.parts import SystemPromptHistoryPart


class LiteLLMMessageAdapter:
    def __init__(self) -> None:
        self._codec = CompletionMessagesCodec()

    def to_native(
        self,
        request: LLMRequest,
    ) -> list[dict[str, Any]]:
        dict_arrays = sum((self._codec.to_native(elem) for elem in request.messages), [])
        if request.system_prompt:
            row = {"role": "system", "content": request.system_prompt.content}
            dict_arrays.insert(0, row)
        return dict_arrays

    def from_native(
        self, litellm_message: litellm.Message, kind: Literal["response", "request"]
    ) -> LLMMessage:
        return self._codec.from_native(litellm_message.model_dump(), kind=kind)

    def to_llm_model[T: BaseModel | str](
        self,
        request: LLMRequest,
        llm_response: ModelResponse,
        output_type: type[T],
    ) -> LLMResult[T]:

        response = cast(litellm.TextCompletionResponse, llm_response)
        usage = response.usage
        if usage is None:
            raise ValueError("Response is strange.")
        tokens = LLMTokens(
            prompt=usage.prompt_tokens, completion=usage.completion_tokens, total=usage.total_tokens
        )
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
        output_message = self.from_native(choice.message, kind="response")
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
        return cast(LLMResult[T], LLMResult(output=content, meta=meta, messages=messages))


class LLMParamConverter:
    def __init__(self) -> None: ...

    def to_litellm_kwargs(self, llm_param: LLMParam) -> dict:
        result = llm_param.model_dump(exclude_none=True)
        candidates = ["reasoning_effort", "verbosity"]
        allowed_openai_params = list(result.get("allowed_openai_params", []))
        for cand in candidates:
            if cand not in allowed_openai_params and cand in result:
                allowed_openai_params.append(cand)
        if allowed_openai_params:
            result["allowed_openai_params"] = allowed_openai_params
        return result
