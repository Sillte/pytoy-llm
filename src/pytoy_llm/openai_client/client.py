from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from pytoy_llm.connection_configuration import ConnectionConfiguration
from pytoy_llm.models import CancelToken, LLMCancelException
from pytoy_llm.models.connections import Connection
from pytoy_llm.models.llm_activities import (
    LLMMinimumActivity,
    LLMRequestActivity,
    LLMResponseActivity,
)
from pytoy_llm.models.llm_events import LLMEventEmitters
from pytoy_llm.models.llm_messages import LLMRequest, LLMRequestLike, LLMResult
from pytoy_llm.models.llm_metas import LLMParam, LLMTokens
from pytoy_llm.openai_client.adapter import (
    LLMParamConverter,
    OpenAICompletion,
    OpenAITransactionAdapter,
)


class PytoyOpenAIClient:
    """OpenAI-compatible Chat Completions client for pytoy-llm."""

    def __init__(
        self,
        connection: str | Connection,
        llm_param: LLMParam | None = None,
        event_emitters: LLMEventEmitters | None = None,
        cancel_token: CancelToken | None = None,
    ) -> None:
        if isinstance(connection, str):
            connection = ConnectionConfiguration().get_connection(connection)
        self._connection = connection
        self._llm_param = llm_param or connection.llm_param or LLMParam()
        self._event_emitters = event_emitters or LLMEventEmitters()
        self._cancel_token = cancel_token

    @property
    def connection(self) -> Connection:
        return self._connection

    def completion[T: BaseModel | str](
        self,
        request: LLMRequestLike,
        output_type: type[T],
    ) -> T:
        return self.completion_with_result(request, output_type).output

    def completion_with_result[T: BaseModel | str](
        self,
        request: LLMRequestLike,
        output_type: type[T],
    ) -> LLMResult[T]:
        normalized_request = LLMRequest.from_any(request)
        response = self.completion_with_native(normalized_request, output_type)
        result = OpenAITransactionAdapter().to_llm_model(
            request=normalized_request,
            llm_response=response,
            output_type=output_type,
        )
        if self._cancel_token is not None and self._cancel_token.is_set():
            raise LLMCancelException(messages=result.messages)
        return result

    def completion_with_native[T: BaseModel | str](
        self,
        request: LLMRequestLike,
        output_type: type[T],
    ) -> OpenAICompletion:
        if self._connection.api_protocol == "responses":
            raise ValueError(
                "The connection is configured to use the 'responses' API protocol. "
                "However, the 'completion_with_native' method is designed for the "
                "'completions' API protocol."
            )

        normalized_request = LLMRequest.from_any(request)
        adapter = OpenAITransactionAdapter()
        messages = adapter.to_native(normalized_request)
        params = LLMParamConverter().to_openai_kwargs(self._llm_param)
        model = self._model_name(self._connection.model)
        trace_id = str(uuid4())
        self._emit_request(trace_id, messages, params)

        try:
            with OpenAI(
                api_key=self._connection.api_key,
                base_url=self._connection.base_url,
            ) as client:
                if output_type is str:
                    response = client.chat.completions.create(
                        model=model,
                        messages=cast(Any, messages),
                        **params,
                    )
                else:
                    response = client.chat.completions.parse(
                        model=model,
                        messages=cast(Any, messages),
                        response_format=cast(type[BaseModel], output_type),
                        **params,
                    )
        except Exception as error:
            self._emit_failure(error)
            raise
        finally:
            self._event_emitters.emit_activity(LLMMinimumActivity(activity_type="post_api_call"))

        self._emit_response(response, trace_id)
        return response

    @staticmethod
    def _model_name(model: str) -> str:
        prefix, separator, model_name = model.partition("/")
        if separator and prefix in {"openai", "gemini", "ollama"}:
            return model_name
        return model

    def _emit_request(
        self,
        trace_id: str,
        messages: list[dict[str, Any]],
        params: dict[str, Any],
    ) -> None:
        try:
            activity = LLMRequestActivity(
                messages=messages,
                trace_id=trace_id,
                model=self._connection.model,
                timeout=params.get("timeout"),
                activity_type="pre_api_call",
            )
        except ValidationError as error:
            activity = LLMMinimumActivity(
                activity_type="pre_api_call",
                message=f"Failed to create LLMRequestActivity: {error}",
            )
        self._event_emitters.emit_activity(activity)

    def _emit_response(self, response: OpenAICompletion, trace_id: str) -> None:
        usage = response.usage
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        if usage is None:
            tokens = None
        else:
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
        try:
            content = response.choices[0].message.content
            if not isinstance(content, str):
                raise ValueError("Response content is not text.")
            activity = LLMResponseActivity(
                response=content,
                tokens=tokens,
                trace_id=trace_id,
                activity_type="response_activity",
            )
        except Exception as error:
            activity = LLMMinimumActivity(
                activity_type="response_activity",
                message=f"Failed to create LLMResponseActivity: {error}",
            )
        self._event_emitters.emit_activity(activity)

    def _emit_failure(self, error: Exception) -> None:
        self._event_emitters.emit_activity(
            LLMMinimumActivity(
                activity_type="response_activity",
                message=f"Completion failed: {error}",
            )
        )
