from pytoy_llm.models import InputMessage, LLMConfig, LLMTool


from pydantic import BaseModel


from typing import Callable, Protocol, Sequence

from pytoy_llm.task.models.repository import LLMTaskStateRepository


class LLMFacadeProtocol[T: BaseModel | str](Protocol):
    def completion(
        self,
        input_messages: Sequence[InputMessage],
        output_format: type[T],
        llm_config: LLMConfig | None,
        connection_name: str | None = None,
    ) -> T:
        """Invoke one LLM call.
        """
        ...

    def run_agent(
        self,
        input_messages: Sequence[InputMessage],
        output_format: type[str] | type[T],
        tools: Sequence[Callable | LLMTool] = (),
        llm_config: LLMConfig | None = None,
        connection_name: str | None = None,
    ) -> T:
        """Use Agent with `tools`.
        """
        ...


class LLMTaskContextProtocol[T: BaseModel | str](Protocol):
    @property
    def llm_facade(self) -> LLMFacadeProtocol[T]: ...

    @property
    def repository(self) -> LLMTaskStateRepository: ...