from pytoy_llm.models import LLMMessageHistory
from pytoy_llm.task.models import LLMTaskSpec  # NOQA
from pytoy_llm.task.models.schemas import LLMTaskRecord  # NOQA

from pydantic import BaseModel, Field

import uuid
from typing import Annotated, Any

from pytoy_llm.task.models.invocations import (
    AgentInvocationSpec,
    FunctionInvocationSpec,
    LLMInvocationSpec,
    SelectedInvocationSpec,
)  # NOQA
from pytoy_llm.task.models.schemas import InvocationSpecMeta, LLMTaskSpecMeta  # NOQA
from pytoy_llm.task.models.context import LLMTaskContext  # NOQA


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
    def output(self) -> T | str:
        return self.record.output


class LLMTaskExecutor:
    def execute[T: BaseModel | str](self, request: LLMTaskRequest) -> LLMTaskResponse[T]:
        request_id = request.id
        task_input = request.task_input
        history = request.history
        record = request.task_spec.run(task_input=task_input, history=history)
        return LLMTaskResponse(record=record, id=request_id)