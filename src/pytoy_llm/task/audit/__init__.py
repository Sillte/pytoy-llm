from pydantic import BaseModel

from pytoy_llm.task.models.task_specs import TaskSpec


class TaskInputAuditLog(BaseModel, frozen=True):
    pass
