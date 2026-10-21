from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, MutableMapping, Self

from pydantic import BaseModel, Field

from pytoy_llm.models import CancelToken, UsageLimit
from pytoy_llm.models.connections import Connection
from pytoy_llm.models.llm_events import LLMEventEmitters
from pytoy_llm.models.llm_messages import LLMMessage
from pytoy_llm.models.llm_metas import LLMParam

type TaskRunState = MutableMapping[str, Any]


ExecutionEvents = LLMEventEmitters


@dataclass(frozen=True)
class ExecutionContext:
    llm_param: LLMParam | None
    connection: Connection | str | None
    llm_messages: list[LLMMessage]
    state: TaskRunState = field(default_factory=dict)
    emitters: LLMEventEmitters = field(default_factory=LLMEventEmitters)
    usage_limit: UsageLimit | None = None
    cancel_token: CancelToken = field(default_factory=CancelToken)


@dataclass(frozen=True)
class RuntimeContextPatch:
    """Contains message history automatically produced by the runtime.

    Runtime messages are copied into the shared context list before ``on_result`` runs.
    ``None`` leaves the history unchanged; an empty list clears it.
    """

    llm_messages: list[LLMMessage] | None = None
    """The latest LLM message history for this context."""

    def apply(self, context: ExecutionContext) -> ExecutionContext:
        if self.llm_messages is not None:
            context.llm_messages[:] = self.llm_messages
        return context


class TaskContextState(BaseModel, frozen=True):
    """
    This object represents the state that can be passed between task executions.
    """

    llm_messages: Annotated[
        list[LLMMessage],
        Field(description="Mutable LLM messages given as the history of interactions"),
    ] = Field(default_factory=list)

    state: Annotated[
        TaskRunState,
        Field(description="Persistent task state."),
    ] = Field(default_factory=dict)

    @classmethod
    def from_execution_context(cls, context: ExecutionContext) -> Self:
        return cls(llm_messages=context.llm_messages, state=context.state)
