import uuid
from typing import Annotated, Any

from pydantic import BaseModel, Field

from pytoy_llm.models import LLMMessageHistory
from pytoy_llm.task.models import LLMTaskSpec
from pytoy_llm.task.models.schemas import (  # NOQA
    LLMTaskRecord,
)


class LLMTaskRequest[T: BaseModel | str](BaseModel):
    task_spec: Annotated[LLMTaskSpec[T], Field(description="Specification of LLMTask")]
    task_input: Annotated[Any, Field(description="Input for the task.")]
    history: Annotated[
        LLMMessageHistory | None,
        Field(description="History of messages. `task_input` is not included."),
    ] = None
    id: Annotated[str, Field(description="ID of TaskRequest")] = Field(
        default_factory=lambda: str(uuid.uuid4())
    )


class LLMTaskResponse[T: BaseModel | str](BaseModel):
    record: LLMTaskRecord[T]
    id: Annotated[str, Field(description="ID of TaskRequest")]

    @property
    def output(self) -> T:
        return self.record.output


class LLMTaskExecutor:
    def execute[T: BaseModel | str](self, request: LLMTaskRequest[T]) -> LLMTaskResponse[T]:
        request_id = request.id
        task_input = request.task_input
        history = request.history
        record = request.task_spec.run(task_input=task_input, history=history)
        return LLMTaskResponse(record=record, id=request_id)
