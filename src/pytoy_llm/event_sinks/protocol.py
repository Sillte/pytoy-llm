from typing import Protocol

from pytoy_llm.models.events.llm_events import LLMEvent


class EventSinkProtocol(Protocol):
    def emit(self, event: LLMEvent) -> None: ...
