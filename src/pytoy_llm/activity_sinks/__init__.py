import logging
from pathlib import Path
from queue import Queue
from threading import RLock
from typing import Any, Literal, Sequence

from pydantic import BaseModel

from pytoy_llm.models.llm_activities import LLMActivity, LLMActivityLog, LLMActivitySink


def to_json_serializable(activity: LLMActivity) -> Any:
    if isinstance(activity, BaseModel):
        return activity.model_dump_json()
    return str(activity)


class LoggerActivitySink(LLMActivitySink):
    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)

    def emit(self, activity: LLMActivity) -> None:
        self.logger.info(activity.model_dump_json())


class QueueActivitySink(LLMActivitySink):
    def __init__(self, queue: Queue) -> None:
        self._queue = queue

    def emit(self, activity: LLMActivity) -> None:
        try:
            self._queue.put(activity, timeout=0.1)
        except Exception:
            ...


class NullActivitySink(LLMActivitySink):
    def emit(self, activity: LLMActivity) -> None:
        pass


class PrintActivitySink(LLMActivitySink):
    def emit(self, activity: LLMActivity) -> None:
        print(str(activity), str(activity.__class__.__name__), flush=True)


class FileActivitySink(LLMActivitySink):
    def __init__(
        self,
        path: Path | str,
        mode: Literal["append", "overwrite", "a", "w"] = "append",
        encoding: str = "utf-8",
    ) -> None:
        path = Path(path)
        if mode == "w":
            mode = "overwrite"
        elif mode == "a":
            mode = "append"

        self.path = path
        self.encoding = encoding
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if mode == "overwrite":
            self.path.write_text("", encoding=self.encoding)

    def emit(self, activity: LLMActivity) -> None:
        with open(self.path, mode="a", encoding=self.encoding) as f:
            f.write(f"{str(activity)}\n")


class ActivityLogSink(LLMActivitySink):
    def __init__(self) -> None:
        self._activities: list[LLMActivity] = []
        self._lock = RLock()  # If `async` calls the same `LogSink`...

    def emit(self, activity: LLMActivity) -> None:
        with self._lock:
            self._activities.append(activity)

    @property
    def activities(self) -> Sequence[LLMActivity]:
        with self._lock:
            return tuple(self._activities)

    @property
    def log(self) -> LLMActivityLog:
        return LLMActivityLog.from_activities(self.activities)


class CompositeActivitySink(LLMActivitySink):
    def __init__(self, sinks: Sequence[LLMActivitySink]) -> None:
        self._sinks = tuple(sinks)

    def emit(self, activity: LLMActivity) -> None:
        for sink in self._sinks:
            sink.emit(activity)
