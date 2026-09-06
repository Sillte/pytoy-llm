from dataclasses import dataclass

from pytoy_llm.task.models.context import TaskContextState
from pytoy_llm.task.models.exceptions import InvocationException
from pytoy_llm.task.models.task_results import TaskResult
from pytoy_llm.task.shared.outcome import Outcome, is_error


@dataclass(frozen=True)
class TaskExit[T]:
    outcome: Outcome[TaskResult[T], InvocationException]
    request_id: str  # ID of request.

    @property
    def result(self) -> TaskResult[T]:
        if is_error(self.outcome):
            raise self.outcome.exception
        return self.outcome.value  # ty: ignore[unresolved-attribute]

    @property
    def output(self) -> T:
        return self.result.output

    @property
    def context_state(self) -> TaskContextState:
        if is_error(self.outcome):
            raise self.outcome.exception
        return self.outcome.value.context_state  # ty: ignore[unresolved-attribute]
