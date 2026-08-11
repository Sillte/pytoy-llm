import logging
from queue import Queue
from typing import Any

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from pytoy_llm.models.events.llm_events import LLMEvent

from .protocol import EventSinkProtocol


def to_json_serializable(event: LLMEvent) -> Any:
    if isinstance(event, BaseModel):
        return event.model_dump_json()
    return str(event)


class LoggerEventSink(EventSinkProtocol):
    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)

    def emit(self, event: LLMEvent) -> None:
        self.logger.info(event.model_dump_json())


class QueueEventSink(EventSinkProtocol):
    def __init__(self, queue: Queue) -> None:
        self._queue = queue

    def emit(self, event: LLMEvent) -> None:
        try:
            self._queue.put(event, timeout=0.1)
        except Exception:
            ...


class NullEventSink(EventSinkProtocol):
    def emit(self, event: LLMEvent) -> None:
        pass


class PrintEventSink(EventSinkProtocol):
    def emit(self, event: LLMEvent) -> None:
        print(str(event), flush=True)



class FileEventSink(EventSinkProtocol):
    def __init__(
        self,
        path: Path | str,
        mode: Literal["append", "overwrite", "a", "w"] = "append",
        encoding: str = "utf-8"
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

    def emit(self, event: LLMEvent) -> None:
        with open(self.path, mode="a", encoding=self.encoding) as f:
            f.write(f"{str(event)}\n")
