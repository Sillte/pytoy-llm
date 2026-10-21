from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from pytoy_llm.models.llm_messages import LLMMessage
from pytoy_llm.task.models.context import ExecutionContext, TaskRunState
from pytoy_llm.task.models.expenditures import Expenditure, NoExpenditure
from pytoy_llm.task.models.metas import InvocationSpecMeta


class InvocationInfo(BaseModel, frozen=True):
    kind: Annotated[str, Field(description="Type of invocation")]
    started_at: Annotated[float, Field(description="Start time of this invocation")]
    ended_at: Annotated[float, Field(description="End time of this invocation")]
    meta: Annotated[
        InvocationSpecMeta, Field(description="Metadata about this invocation spec")
    ] = InvocationSpecMeta()

    @property
    def spec_name(self) -> str:
        return self.meta.name

    @property
    def intent(self) -> str:
        return self.meta.intent


@dataclass(frozen=True)
class InvocationResult[T]:
    """Invocation output with a reference to the live execution context.

    Context mutations are immediately visible and are not rolled back if a result hook fails.
    The frozen dataclass prevents replacing the context reference, not mutating its contents.
    """

    __pydantic_config__ = ConfigDict(arbitrary_types_allowed=True)

    output: T
    context: ExecutionContext


@dataclass(frozen=True)
class SnapshotInvocationResult[T]:
    """Deep-copied output and context values captured after a successful result hook."""

    __pydantic_config__ = ConfigDict(arbitrary_types_allowed=True)

    output: T
    state: TaskRunState
    llm_messages: list[LLMMessage]

    @classmethod
    def from_invocation_result(cls, result: InvocationResult[T]) -> SnapshotInvocationResult[T]:
        return deepcopy(
            cls(
                output=result.output,
                state=result.context.state,
                llm_messages=result.context.llm_messages,
            )
        )


class FailedSnapshot(BaseModel, frozen=True):
    """Describes a failed snapshot copy without failing the invocation."""

    error_type: str
    message: str


class InvocationTrace[T](BaseModel, frozen=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    input: Annotated[Any, Field(description="Input")]
    result: Annotated[InvocationResult[T], Field(description="Result of the invocation")]
    snapshot_result: Annotated[
        SnapshotInvocationResult[T] | FailedSnapshot | None,
        Field(description="Snapshot data, snapshot failure details, or None before on_result."),
    ] = None
    info: Annotated[InvocationInfo, Field(description="Metatada Information about the invocation.")]
    expenditure: Annotated[Expenditure, Field(description="Expenditure of the invocation")] = (
        NoExpenditure()
    )
    details: Annotated[
        Mapping[str, JsonValue], Field(description="detailed information for debugging")
    ] = {}
    children: Annotated[Sequence[InvocationTrace], Field(description="Children of execution")] = ()
