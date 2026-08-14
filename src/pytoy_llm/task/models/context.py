from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, MutableMapping, Self

from pydantic import BaseModel, Field, field_validator

from pytoy_llm.event_sinks import EventSinkProtocol
from pytoy_llm.models.connections import Connection
from pytoy_llm.models.llm_messages import LLMMessage, LLMMessagesLike
from pytoy_llm.models.llm_metas import LLMParam

type TaskRunState = MutableMapping[str, Any]


@dataclass(frozen=True)
class ExecutionContext:
    llm_param: LLMParam | None
    connection: Connection | str | None
    llm_messages: Sequence[LLMMessage]
    state: TaskRunState = field(default_factory=dict)
    event_sink: EventSinkProtocol | None = None


@dataclass(frozen=True)
class RuntimeContextPatch:
    """Represents changes to an `ExecutionContext` automatically produced by the runtime.

    This patch is applied before `ContextPatch`.
    """

    llm_messages: Sequence[LLMMessage]
    """The latest LLM message history for this context."""

    def apply(self, context: ExecutionContext) -> ExecutionContext:
        return replace(context, llm_messages=self.llm_messages)


class ContextPatch(BaseModel, frozen=True):
    """Represents changes to an `ExecutionContext` produced by an invocation.

    The patch describes changes to be applied when constructing the context
    for subsequent invocations. This is applied after `RuntimeContextPatch`.
    """

    state: Annotated[
        Mapping[str, Any] | None,
        Field(description="New states requested by the invocation."),
    ] = None

    llm_messages: Annotated[
        Sequence[LLMMessage] | None,
        Field(
            description=("Overrides the LLM message history for this context. Applied after `RuntimeContextPatch`."),
        ),
    ] = None

    def apply(self, context: ExecutionContext) -> ExecutionContext:
        if self.state is not None:
            context = replace(context, state=self.state)
        if self.llm_messages is not None:
            context = replace(context, llm_messages=self.llm_messages)
        return context


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
