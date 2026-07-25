from collections.abc import Callable, Sequence
from typing import Protocol, overload

from pydantic import BaseModel

from pytoy_llm.llm_facade import LLMFacade
from pytoy_llm.models import Connection, LLMConfig, LLMMessage, LLMMessagesLike, LLMTool
from pytoy_llm.task.models.repository import LLMTaskStateRepository


class LLMFacadeProtocol(Protocol):
    @overload
    def completion(
        self,
        messages: LLMMessagesLike,
        output_type: type[str],
        llm_config: LLMConfig | None,
        connection: Connection | str | None,
    ) -> str: ...

    @overload
    def completion[T: BaseModel](
        self,
        messages: LLMMessagesLike,
        output_type: type[T],
        llm_config: LLMConfig | None,
        connection: Connection | str | None,
    ) -> T: ...

    def completion(
        self,
        messages: LLMMessagesLike,
        output_type: type[BaseModel] | type[str],
        llm_config: LLMConfig | None = None,
        connection: Connection | str | None = None,
    ) -> BaseModel | str: ...

    def run_agent[T: BaseModel](
        self,
        messages: Sequence[LLMMessage] | LLMMessage,
        output_type: type[T],
        tools: Sequence[Callable | LLMTool] = (),
        llm_config: LLMConfig | None = None,
        connection: Connection | str | None = None,
    ) -> T | str:
        """Use Agent with `tools`."""
        ...


class LLMTaskContextProtocol(Protocol):
    @property
    def llm_facade(self) -> LLMFacade: ...

    @property
    def repository(self) -> LLMTaskStateRepository: ...
