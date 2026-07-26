from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from pytoy_llm.llm_facade import LLMFacade
from pytoy_llm.models.llm_messages import LLMMessage
from pytoy_llm.task.models.repository import LLMTaskStateRepository


@dataclass(frozen=True)
class LLMTaskContext:
    llm_facade: LLMFacade
    llm_messages: Sequence[LLMMessage]

    repository: LLMTaskStateRepository = field(default_factory=LLMTaskStateRepository)
