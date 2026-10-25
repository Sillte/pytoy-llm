from typing import Protocol

from pydantic import BaseModel

from pytoy_llm.models import CancelToken, LLMRequestLike, LLMResult
from pytoy_llm.models.connections import Connection
from pytoy_llm.models.llm_events import LLMEventEmitters
from pytoy_llm.models.llm_metas import LLMParam


class CompletionClient(Protocol):
    def completion[T: BaseModel | str](
        self,
        request: LLMRequestLike,
        output_type: type[T],
    ) -> T: ...

    def completion_with_result[T: BaseModel | str](
        self,
        request: LLMRequestLike,
        output_type: type[T],
    ) -> LLMResult[T]: ...


class CompletionClientFactory(Protocol):
    def create(
        self,
        connection: str | Connection,
        llm_param: LLMParam | None = None,
        event_emitters: LLMEventEmitters | None = None,
        cancel_token: CancelToken | None = None,
    ) -> CompletionClient: ...
