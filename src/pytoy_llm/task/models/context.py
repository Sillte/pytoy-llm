from __future__ import annotations

import uuid
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


def _resolve_task_id(task_id: str | None = None) -> str:
    return task_id if task_id is not None else str(uuid.uuid4())


@dataclass(frozen=True)
class ExecutionContext:
    llm_param: LLMParam | None
    connection: Connection | str | None
    llm_messages: list[LLMMessage]
    state: TaskRunState = field(default_factory=dict)
    emitters: LLMEventEmitters = field(default_factory=LLMEventEmitters)
    usage_limit: UsageLimit | None = None
    cancel_token: CancelToken = field(default_factory=CancelToken)
    task_id: str = field(default_factory=_resolve_task_id)

    @classmethod
    def from_any(
        cls,
        llm_param: LLMParam | None,
        connection: Connection | str | None,
        llm_messages: list[LLMMessage],
        *,
        state: TaskRunState | None = None,
        emitters: LLMEventEmitters | None = None,
        usage_limit: UsageLimit | None = None,
        cancel_token: CancelToken | None = None,
        task_id: str | None = None,
    ) -> Self:
        """Create a context, generating a UUID4 task ID when one is not provided."""
        return cls(
            llm_param=llm_param,
            connection=connection,
            llm_messages=llm_messages,
            state=state if state is not None else {},
            emitters=emitters if emitters is not None else LLMEventEmitters(),
            usage_limit=usage_limit,
            cancel_token=cancel_token if cancel_token is not None else CancelToken(),
            task_id=_resolve_task_id(task_id),
        )


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
