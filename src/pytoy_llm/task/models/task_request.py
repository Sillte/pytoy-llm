import uuid
from typing import Annotated, Any

from pydantic import BaseModel, Field

from pytoy_llm.task.models.context import TaskContextState
from pytoy_llm.task.models.task_specs import TaskSpec


class TaskRequest[T: BaseModel | str](BaseModel):
    spec: Annotated[TaskSpec[T], Field(description="Specification of LLMTask")]
    input: Annotated[Any, Field(description="Input for the task.")]
    context_state: Annotated[TaskContextState | None, Field(description="The state of context where Task is executed.")] = None
    id: Annotated[str, Field(description="ID of TaskRequest")] = Field(default_factory=lambda: str(uuid.uuid4()))
