import uuid
from collections.abc import Sequence
from typing import Annotated

from pydantic import BaseModel, Field

from pytoy_llm.task.models.context import TaskContextState
from pytoy_llm.task.models.invocation_results import InvocationTrace


class TaskResult[T](BaseModel, frozen=True):
    id: Annotated[str, Field(description="Unique identifier for this Task result")] = Field(
        default_factory=lambda: str(uuid.uuid1())
    )
    output: Annotated[T, Field(description="Final output produced by the task")]
    traces: Annotated[
        Sequence[InvocationTrace] | None,
        Field(description="Trace of all invocations executed in this task"),
    ] = None
    task_name: Annotated[str, Field(description="Name of the executed task")] = "No Task Name"
    context_state: Annotated[TaskContextState, Field(description="Final TaskContext")]
