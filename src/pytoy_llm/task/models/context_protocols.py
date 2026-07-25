from collections.abc import Callable, Sequence
from typing import Protocol

from pydantic import BaseModel

from pytoy_llm.models import LLMConfig, LLMMessage, LLMTool
from pytoy_llm.task.models.repository import LLMTaskStateRepository


class LLMFacadeProtocol[T: BaseModel | str](Protocol):
    def completion(
        self,
        messages: Sequence[LLMMessage] | LLMMessage,
        output_type: type[T],
        llm_config: LLMConfig | None,
        connection_name: str | None = None,
    ) -> T:
        """Invoke one LLM call."""
        ...

    def run_agent(
        self,
        messages: Sequence[LLMMessage] | LLMMessage,
        output_type: type[T],
        tools: Sequence[Callable | LLMTool] = (),
        llm_config: LLMConfig | None = None,
        connection_name: str | None = None,
    ) -> T:
        """Use Agent with `tools`."""
        ...


class LLMTaskContextProtocol[T: BaseModel | str](Protocol):
    @property
    def llm_facade(self) -> LLMFacadeProtocol[T]: ...

    @property
    def repository(self) -> LLMTaskStateRepository: ...
