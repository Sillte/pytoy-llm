from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, ClassVar, Generator
from uuid import uuid4

import litellm
from litellm.integrations.custom_logger import CustomLogger
from pydantic import ValidationError

from pytoy_llm.activity_sinks import NullActivitySink
from pytoy_llm.activity_sinks.protocol import ActivitySinkProtocol
from pytoy_llm.models.activities.llm_activities import LLMMinimumActivity, LLMRequestActivity, LLMResponseActivity
from pytoy_llm.models.llm_metas import LLMTokens


class ActivitySinkRepository:
    def __init__(self) -> None:
        self._activity_sinks: dict[str, ActivitySinkProtocol] = {}
        self._lock = threading.RLock()
        self._ttl_seconds: float = 0.2

    @contextmanager
    def register(
        self,
        activity_sink: ActivitySinkProtocol,
    ) -> Generator[dict[str, str]]:
        sink_id = str(uuid4())

        with self._lock:
            self._activity_sinks[sink_id] = activity_sink

        def _remove(sink_id: str) -> None:
            with self._lock:
                self._activity_sinks.pop(sink_id, None)

        # NOTE: Since `litellm.callbacks` are global, this delay is important.
        try:
            yield {"event_sink_id": sink_id}
        finally:
            # In case of `NullEventSink`, it is not necessary to wait.
            if isinstance(activity_sink, NullActivitySink):
                _remove(sink_id)
            else:
                timer = threading.Timer(
                    self._ttl_seconds,
                    _remove,
                    args=(sink_id,),
                )
                timer.daemon = True
                timer.start()

    def get(
        self,
        metadata: dict[str, Any],
    ) -> ActivitySinkProtocol | None:
        sink_id = metadata.get("activity_sink_id", "")
        with self._lock:
            return self._activity_sinks.get(sink_id)


class LiteLLMEventHandler(CustomLogger):
    _used: bool = False

    _instance: ClassVar[LiteLLMEventHandler | None] = None
    _instance_lock = threading.RLock()

    def __new__(cls) -> LiteLLMEventHandler:
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "_initialized"):
            return
        super().__init__()
        self.sink_repository = ActivitySinkRepository()

        # NOTE: Add this handle to `litellm.callbacks`. This is a global callback, so it will be called for every request.

        callbacks = litellm.callbacks or []
        if self not in callbacks:
            litellm.callbacks = [*callbacks, self]
        callbacks = litellm.success_callback or []
        if self not in callbacks:
            litellm.success_callback = [*callbacks, self]
        callbacks = litellm.failure_callback or []
        if self not in callbacks:
            litellm.failure_callback = [*callbacks, self]

        self._initialized = True

    def _activity_sink(self, **kwargs) -> ActivitySinkProtocol | None:
        metadata = kwargs.get("litellm_params", {}).get("metadata", {})
        return self.sink_repository.get(metadata)

    @contextmanager
    def register(self, activity_sink: ActivitySinkProtocol) -> Generator[dict[str, str]]:
        with self.sink_repository.register(activity_sink) as metadata:
            yield metadata

    def log_pre_api_call(self, model, messages, kwargs, **_):
        sink = self._activity_sink(**kwargs)
        if sink is None:
            return
        try:
            trace_id, call_id = kwargs.get("litellm_trace_id"), kwargs.get("litellm_call_id")
            timeout = kwargs.get("timeout")
            activity = LLMRequestActivity(
                messages=messages,
                trace_id=trace_id,
                call_id=call_id,
                model=model,
                timeout=timeout,
                activity_type="pre_api_call",
            )
        except ValidationError as e:
            activity = LLMMinimumActivity(activity_type="pre_api_call", message=f"Failed to create LLMRequestActivity: {e}")
        sink.emit(activity)

    def log_post_api_call(self, kwargs, response_obj, start_time, end_time, **_):
        sink = self._activity_sink(**kwargs)
        if sink is None:
            return
        sink.emit(LLMMinimumActivity(activity_type="post_api_call"))

    def log_success_event(self, kwargs, response_obj, start_time, end_time, **_):
        sink = self._activity_sink(**kwargs)
        if sink is None:
            return
        activity = self._to_response_activity(response_obj)
        sink.emit(activity)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time, **_):
        sink = self._activity_sink(**kwargs)
        if sink is None:
            return
        activity = self._to_response_activity(response_obj)
        sink.emit(activity)

    def _to_response_activity(self, response_obj) -> LLMResponseActivity | LLMMinimumActivity:
        try:
            usage = response_obj.usage
            if usage:
                tokens = LLMTokens(prompt=usage.prompt_tokens, completion=usage.completion_tokens, total=usage.total_tokens)
            else:
                tokens = None
            choice = response_obj.choices[0]
            content = choice.message.content
            activity = LLMResponseActivity(response=content, tokens=tokens, activity_type="response_activity")
        except Exception as e:
            activity = LLMMinimumActivity(
                activity_type="response_activity", message=f"Failed to create LLMResponseActivity: {e}"
            )
        return activity
