import json
import re

from pydantic import BaseModel

from pytoy_llm.composers.materials import MaterialSection
from pytoy_llm.models import LLMMessage
from pytoy_llm.task.audit import TaskAuditor
from pytoy_llm.task.models import TaskContextState, TaskResult


class AuditPayload(BaseModel):
    value: int


def test_auditor_make_llm_messages_audit_section_includes_schema_and_data() -> None:
    message_text = "Question included in the audit context"
    result = TaskResult(
        input=AuditPayload(value=1),
        output=AuditPayload(value=2),
        context_state=TaskContextState(
            llm_messages=[LLMMessage.chat(message_text)],
            state={"ready": True},
        ),
    )

    section = TaskAuditor.from_task_result(result).make_llm_messages_audit_section()
    assert isinstance(section, MaterialSection)
    assert section.name == "LLM Messages"
    assert section.usage.usage == "LLMMessages used for analysis"

    audit_text = section.compose(header_depth=2)

    json_blocks = re.findall(r"```json\n(.*?)\n```", audit_text, flags=re.DOTALL)
    assert "## Material LLM Messages" in audit_text
    assert "#### JSON Schema" in audit_text
    assert "#### JSON Instance" in audit_text
    assert len(json_blocks) == 2
    audit_schema, audit_data = (json.loads(block) for block in json_blocks)
    assert set(audit_schema["properties"]) == {"llm_messages", "llm_tokens"}
    assert audit_data["llm_tokens"]["total"] == 0
    assert message_text in json.dumps(audit_data["llm_messages"])
    assert message_text not in json_blocks[0]


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
