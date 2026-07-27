from typing import Annotated

from pydantic import BaseModel, Field

from pytoy_llm.task.models.context import TaskContextState
from pytoy_llm.task.models.task_results import TaskResult


class TaskResponse[T: BaseModel | str](BaseModel):
    result: TaskResult[T]
    request_id: Annotated[str, Field(description="ID of TaskRequest")]

    @property
    def output(self) -> T:
        return self.result.output

    @property
    def context_state(self) -> TaskContextState:
        return self.result.context_state
