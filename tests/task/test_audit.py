import json

from pydantic import BaseModel

from pytoy_llm.task.audit import TaskAuditor
from pytoy_llm.task.models import TaskContextState, TaskResult


class AuditPayload(BaseModel):
    value: int


def test_auditor_dump_serializes_jsonable_models(tmp_path) -> None:
    result = TaskResult(
        input=AuditPayload(value=1),
        output=AuditPayload(value=2),
        context_state=TaskContextState(state={"ready": True}),
    )

    path = tmp_path / "audit.json"
    TaskAuditor.from_task_result(result).dump(path)
    audit_data = json.loads(path.read_text(encoding="utf8"))

    assert audit_data["start"]["input"] == {"value": 1}
    assert audit_data["end"]["output"] == {"value": 2}


def test_auditor_dump_replaces_non_jsonable_values(tmp_path) -> None:
    result = TaskResult(
        input=object(),
        output=object(),
        context_state=TaskContextState(state={"unserializable": object()}),
    )

    path = tmp_path / "audit.json"
    TaskAuditor.from_task_result(result).dump(path)
    audit_data = json.loads(path.read_text(encoding="utf8"))

    assert audit_data["start"]["input"] == "NonJsonable"
    assert audit_data["end"]["output"] == "NonJsonable"
    assert audit_data["end"]["state"] == "NonJsonable"
    assert audit_data["links"][0]["value"] == "NonJsonable"
