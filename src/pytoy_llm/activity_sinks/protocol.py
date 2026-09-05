from typing import Protocol

from pytoy_llm.models.activities.llm_activities import LLMActivity


class ActivitySinkProtocol(Protocol):
    def emit(self, activity: LLMActivity) -> None: ...
