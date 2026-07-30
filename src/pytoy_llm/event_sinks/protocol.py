from typing import Any, Protocol


class EventSinkProtocol(Protocol):
    def emit(self, event: Any) -> None: ...
