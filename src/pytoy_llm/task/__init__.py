import uuid
from typing import Annotated, Any

from pydantic import BaseModel, Field

from pytoy_llm.task.models import LLMTaskSpec
from pytoy_llm.task.models.schemas import (
    LLMTaskResult,
    TaskContextState,
)


class LLMTaskRequest[T: BaseModel | str](BaseModel):
    spec: Annotated[LLMTaskSpec[T], Field(description="Specification of LLMTask")]
    input: Annotated[Any, Field(description="Input for the task.")]
    context_state: Annotated[TaskContextState | None, Field(description="The state of context where Task is executed.")] = None
    id: Annotated[str, Field(description="ID of TaskRequest")] = Field(default_factory=lambda: str(uuid.uuid4()))


class LLMTaskResponse[T: BaseModel | str](BaseModel):
    result: LLMTaskResult[T]
    context_state: Annotated[TaskContextState | None, Field(description="The state of context where Task is finished.")] = None
    id: Annotated[str, Field(description="ID of TaskRequest")]

    @property
    def output(self) -> T:
        return self.result.output


class LLMTaskExecutor:
    def execute[T: BaseModel | str](self, request: LLMTaskRequest[T]) -> LLMTaskResponse[T]:
        request_id = request.id
        task_input = request.input
        context_state = request.context_state or TaskContextState()
        record = request.spec.run(task_input=task_input, context_state=context_state)
        return LLMTaskResponse(result=record, id=request_id)
