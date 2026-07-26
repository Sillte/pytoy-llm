from typing import Protocol

from pytoy_llm.llm_facade import LLMFacade
from pytoy_llm.task.models.repository import LLMTaskStateRepository


class LLMTaskContextProtocol(Protocol):
    @property
    def llm_facade(self) -> LLMFacade: ...

    @property
    def repository(self) -> LLMTaskStateRepository: ...
