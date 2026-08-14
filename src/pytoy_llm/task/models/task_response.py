from dataclasses import dataclass

from pytoy_llm.task.models.context import TaskContextState
from pytoy_llm.task.models.task_results import TaskResult


@dataclass(frozen=True)
class TaskResponse[T]:
    result: TaskResult[T]
    request_id: str  # ID of request.

    @property
    def output(self) -> T:
        return self.result.output

    @property
    def context_state(self) -> TaskContextState:
        return self.result.context_state
