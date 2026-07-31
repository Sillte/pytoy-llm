from typing import Protocol

from pytoy_llm.models.llm_events import LLMEvent


class EventSinkProtocol(Protocol):
    def emit(self, event: LLMEvent) -> None: ...
