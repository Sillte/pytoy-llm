from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, Self

from pydantic import BaseModel, Field, field_validator

from pytoy_llm.event_sinks import EventSinkProtocol
from pytoy_llm.models.connections import Connection
from pytoy_llm.models.llm_messages import LLMMessage, LLMMessagesLike
from pytoy_llm.models.llm_metas import LLMParam
from pytoy_llm.task.models.task_state import TaskRunState


@dataclass(frozen=True)
class ExecutionContext:
    llm_param: LLMParam | None
    connection: Connection | str | None
    llm_messages: Sequence[LLMMessage]
    state: TaskRunState = field(default_factory=dict)
    event_sink: EventSinkProtocol | None = None


class ContextPatch(BaseModel, frozen=True):
    state_updates: Annotated[
        Mapping[str, Any],
        Field(description="Updates to the task state requested by this invocation."),
    ] = {}

    llm_messages: Annotated[
        Sequence[LLMMessage],
        Field(description="LLM message history updates produced during this invocation."),
    ] = ()

    def patch(self, context: ExecutionContext) -> ExecutionContext:
        state = context.state
        state.update(self.state_updates)

        llm_messages = self.llm_messages

        return replace(context, state=state, llm_messages=llm_messages)


class TaskContextState(BaseModel, frozen=True):
    """
    This object represents the state that can be passed between task executions.
    """

    llm_messages: Annotated[
        Sequence[LLMMessage],
        Field(description="LLM messages given as the history of interactions"),
    ] = ()

    state: Annotated[
        TaskRunState,
        Field(description="Persistent task state."),
    ] = Field(default_factory=dict)

    @field_validator("llm_messages", mode="before")
    @classmethod
    def normalize_messages(cls, value: LLMMessagesLike) -> Sequence[LLMMessage]:
        return LLMMessage.to_messages(value)

    @classmethod
    def from_execution_context(cls, context: ExecutionContext) -> Self:
        return cls(llm_messages=context.llm_messages, state=context.state)
