import logging
from pathlib import Path
from queue import Queue
from typing import Any, Literal

from pydantic import BaseModel

from pytoy_llm.models.activities.llm_activities import LLMActivity

from .protocol import ActivitySinkProtocol


def to_json_serializable(activity: LLMActivity) -> Any:
    if isinstance(activity, BaseModel):
        return activity.model_dump_json()
    return str(activity)


class LoggerActivitySink(ActivitySinkProtocol):
    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)

    def emit(self, activity: LLMActivity) -> None:
        self.logger.info(activity.model_dump_json())


class QueueActivitySink(ActivitySinkProtocol):
    def __init__(self, queue: Queue) -> None:
        self._queue = queue

    def emit(self, activity: LLMActivity) -> None:
        try:
            self._queue.put(activity, timeout=0.1)
        except Exception:
            ...


class NullActivitySink(ActivitySinkProtocol):
    def emit(self, activity: LLMActivity) -> None:
        pass


class PrintActivitySink(ActivitySinkProtocol):
    def emit(self, activity: LLMActivity) -> None:
        print(str(activity), flush=True)


class FileActivitySink(ActivitySinkProtocol):
    def __init__(
        self, path: Path | str, mode: Literal["append", "overwrite", "a", "w"] = "append", encoding: str = "utf-8"
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
