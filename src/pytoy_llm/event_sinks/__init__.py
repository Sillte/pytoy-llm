import logging
from queue import Queue
from typing import Any

from pydantic import BaseModel

from .protocol import EventSinkProtocol


def to_json_serializable(event: Any) -> Any:
    if isinstance(event, BaseModel):
        return event.model_dump_json()
    return str(event)


class LoggerEventSink(EventSinkProtocol):
    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)

    def emit(self, event: Any) -> None:
        event = to_json_serializable(event)
        self.logger.info(event)


class QueueEventSink(EventSinkProtocol):
    def __init__(self, queue: Queue) -> None:
        self._queue = queue

    def emit(self, event: Any) -> None:
        try:
            self._queue.put(event, timeout=0.1)
        except Exception:
            ...


class NullEventSink(EventSinkProtocol):
    def emit(self, event: Any) -> None:
        pass


class PrintEventSink(EventSinkProtocol):
    def emit(self, event: Any) -> None:
        print(str(event), flush=True)
