import uuid
from dataclasses import dataclass, field
from typing import Any, Self

from pytoy_llm.event_sinks import EventSinkProtocol
from pytoy_llm.task.models.context import TaskContextState
from pytoy_llm.task.models.invocation_specs import (
    InvocationSpec,
)
from pytoy_llm.task.models.task_specs import TaskSpec


@dataclass(frozen=True)
class TaskRequest[T]:
    spec: TaskSpec[T]
    input: Any  # Input of the task.,
    context_state: TaskContextState | None = None  # The state of the task when it is executed.
    event_sink: EventSinkProtocol | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @classmethod
    def from_invocation_spec(
        cls,
        spec: InvocationSpec,
        input: Any,
        *,
        context_state: TaskContextState | None = None,
        event_sink: EventSinkProtocol | None = None,
    ) -> Self:
        task_spec = TaskSpec.from_single_spec(
            spec,
            meta="SingleInvocationTask",
        )
        return cls(spec=task_spec, input=input, context_state=context_state, event_sink=event_sink)
