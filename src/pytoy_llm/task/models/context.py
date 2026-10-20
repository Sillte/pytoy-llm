from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
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
    llm_messages: Sequence[LLMMessage]
    state: TaskRunState = field(default_factory=dict)
    emitters: LLMEventEmitters = field(default_factory=LLMEventEmitters)
    usage_limit: UsageLimit | None = None
    cancel_token: CancelToken = field(default_factory=CancelToken)


@dataclass(frozen=True)
class RuntimeContextPatch:
    """Represents changes to an `ExecutionContext` automatically produced by the runtime.

    This patch is applied before `ContextPatch`. `None` leaves the history unchanged;
    an empty sequence clears it.
    """

    llm_messages: Sequence[LLMMessage] | None = None
    """The latest LLM message history for this context."""

    def apply(self, context: ExecutionContext) -> ExecutionContext:
        if self.llm_messages is not None:
            return replace(context, llm_messages=self.llm_messages)
        return context


class ContextPatch(BaseModel):
    """Represents mutable changes to an `ExecutionContext` produced by an invocation.

    The patch is applied after `RuntimeContextPatch`. State is cleared first,
    then removed, then updated; updates win when a key appears in both sets.
    """

    state_updates: Annotated[
        TaskRunState,
        Field(description="State entries to update in the existing context state."),
    ] = Field(default_factory=dict)
    state_removals: Annotated[
        set[str], Field(description="State keys to remove from the existing context state.")
    ] = Field(default_factory=set)
    clear_state: bool = False

    llm_messages: Annotated[
        Sequence[LLMMessage] | None,
        Field(
            description=(
                "Overrides the LLM message history when not None. Applied after `RuntimeContextPatch`."
            ),
        ),
    ] = None

    def apply(self, context: ExecutionContext) -> ExecutionContext:
        if self.clear_state or self.state_removals or self.state_updates:
            state = {} if self.clear_state else dict(context.state)
            for key in self.state_removals:
                state.pop(key, None)
            state.update(self.state_updates)
            context = replace(context, state=state)
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

    @classmethod
    def from_execution_context(cls, context: ExecutionContext) -> Self:
        return cls(llm_messages=context.llm_messages, state=context.state)
