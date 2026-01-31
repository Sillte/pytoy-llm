from pytoy_llm.models import LLMMessageHistory
from pytoy_llm.task.models import LLMTaskSpec, LLMInvocationSpec, InvocationSpec  # NOQA
from pytoy_llm.task.models import LLMInvocationSpec, InvocationSpec, LLMTaskMeta  # NOQA

from pydantic import BaseModel, Field

import uuid
from typing import Annotated, Any


class LLMTaskRequest(BaseModel):
    task_spec: Annotated[LLMTaskSpec, Field(description="Specification of LLMTask")]
    task_input: Annotated[Any, Field(description="Input for the task.")]
    history : Annotated[LLMMessageHistory | None , Field(description="History")] = None
    id: Annotated[str, Field(description="ID of TaskRequest")] = str(uuid.uuid1())


class LLMTaskResponse[S: BaseModel](BaseModel):
    output: S | str
    id: Annotated[str, Field(description="ID of TaskRequest")]
    # metadata: dict[str, Any] = {} # トークン数や実行ログなど


class LLMTaskExecutor:
    def execute(self, request: LLMTaskRequest) -> LLMTaskResponse:
        request_id = request.id
        task_input = request.task_input
        history = request.history
        output = request.task_spec.run(task_input=task_input, history=history)
        return LLMTaskResponse(output=output, id=request_id)