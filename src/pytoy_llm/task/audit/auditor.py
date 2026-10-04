from pathlib import Path
from typing import Annotated, Self, Sequence

from pydantic import BaseModel, Field

from pytoy_llm.composers.materials import MaterialSection
from pytoy_llm.materials.models import JsonMaterialData
from pytoy_llm.models import LLMMessage, LLMTokens
from pytoy_llm.task.audit.models import TaskAuditLog
from pytoy_llm.task.models import TaskResult


class TaskLLMMessagesAuditLog(BaseModel, frozen=True):
    """Audit log for `LLMMessages` of the task."""

    llm_messages: Annotated[
        Sequence[LLMMessage],
        Field(description="LLM messages of context at the end."),
    ]
    llm_tokens: Annotated[LLMTokens, Field(description="LLM Tokens of the task.")]


class TaskAuditor:
    def __init__(self, audit_log: TaskAuditLog) -> None:
        self._audit_log = audit_log

    @classmethod
    def from_task_result(cls, task_result: TaskResult) -> Self:
        return cls(audit_log=TaskAuditLog.from_task_result(task_result))

    def dump(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text(self._audit_log.model_dump_json(indent=2), encoding="utf8")

    def make_llm_messages_audit_section(self) -> MaterialSection:
        llm_messages = self._audit_log.end.llm_messages
        llm_tokens = self._audit_log.end.llm_tokens
        audit_log = TaskLLMMessagesAuditLog(llm_messages=llm_messages, llm_tokens=llm_tokens)
        material_data = JsonMaterialData(
            description="LLM context of the task.",
            json_schema=audit_log.model_json_schema(),
            data=audit_log.model_dump(mode="json"),
        )
        usage = "LLMMessages used for analysis"
        return MaterialSection.from_any("LLM Messages", usage=usage, data=material_data)
