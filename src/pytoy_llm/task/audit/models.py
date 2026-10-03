import uuid
from typing import Annotated, Any, Literal, Mapping, Self, Sequence

from pydantic import BaseModel, Field, JsonValue, TypeAdapter, ValidationError
from pydantic_core import PydanticSerializationError, to_jsonable_python

from pytoy_llm.models import LLMMessage, LLMTokens
from pytoy_llm.task.models import (
    Expenditure,
    InvocationInfo,
    InvocationTrace,
    TaskResult,
)

_JSON_VALUE_ADAPTER = TypeAdapter(JsonValue)


def _to_json_value(value: Any) -> JsonValue:
    try:
        return _JSON_VALUE_ADAPTER.validate_python(to_jsonable_python(value))
    except (PydanticSerializationError, ValidationError):
        return "NonJsonable"


type AuditLogType = Literal["task-start", "invocation", "task-end"]
type AuditLogID = str


class AuditLogRef(BaseModel):
    """Specify the reference a AuditLog."""

    type: AuditLogType = Field(description="Type of Audit log.")
    id: AuditLogID = Field(description="")


class InvocationLinkAuditLog(BaseModel, frozen=True):
    """Audit record representing a value transferred between audit records."""

    source: AuditLogRef
    target: AuditLogRef
    value: JsonValue = Field(
        description=("Value transferred from the source audit record to the target audit record.")
    )


class AuditLogLinkGenerator:
    def __init__(self):
        self._task_start_ref: AuditLogRef = AuditLogRef(type="task-start", id=str(uuid.uuid4()))
        self._task_end_ref: AuditLogRef = AuditLogRef(type="task-end", id=str(uuid.uuid4()))
        self._references = {
            self._task_start_ref.id: self._task_start_ref,
            self._task_end_ref.id: self._task_end_ref,
        }
        self._links = list()

    @property
    def task_start_ref(self) -> AuditLogRef:
        return self._task_start_ref

    @property
    def task_end_ref(self) -> AuditLogRef:
        return self._task_end_ref

    def get_ref(self, log_id: AuditLogID) -> AuditLogRef | None:
        return self._references.get(log_id)

    def generate_ref(self, log_type: AuditLogType) -> AuditLogRef:
        log_id = str(uuid.uuid4())
        ref = AuditLogRef(type=log_type, id=log_id)
        self._references[log_id] = ref
        return ref

    def generate_link(
        self, source_ref: AuditLogRef, target_ref: AuditLogRef, value: Any
    ) -> InvocationLinkAuditLog:
        link = InvocationLinkAuditLog(
            source=source_ref, target=target_ref, value=_to_json_value(value)
        )
        self._links.append(link)
        return link

    @property
    def links(self) -> Sequence[InvocationLinkAuditLog]:
        return self._links


class TaskStartAuditLog(BaseModel, frozen=True):
    """Audit record for starting of task."""

    type: Literal["task-start"] = "task-start"
    input: JsonValue = Field(description="Input of Task")
    task_name: str = Field(description="Name of Task")
    task_id: str = Field(description="ID of Task")
    log_id: AuditLogID = Field(
        description="AuditLog ID.", default_factory=lambda: str(uuid.uuid4())
    )

    @classmethod
    def from_task_result(
        cls, task_result: TaskResult, link_generator: AuditLogLinkGenerator
    ) -> Self:
        ref = link_generator.task_start_ref
        return cls(
            input=_to_json_value(task_result.input),
            task_name=task_result.task_name,
            task_id=task_result.id,
            log_id=ref.id,
        )


class InvocationAuditLog(BaseModel, frozen=True):
    """Audit record for invocation."""

    type: Literal["invocation"] = "invocation"
    info: Annotated[InvocationInfo, Field(description="Metatada Information about the invocation.")]
    expenditure: Annotated[Expenditure, Field(description="Expenditure of the invocation")]
    details: Annotated[
        Mapping[str, JsonValue], Field(description="detailed information for debugging")
    ] = {}

    log_id: AuditLogID = Field(
        description="AuditLog ID.", default_factory=lambda: str(uuid.uuid4())
    )

    @classmethod
    def from_invocation_traces(
        cls,
        invocation_traces: Sequence[InvocationTrace],
        current_ref: AuditLogRef,
        link_generator: AuditLogLinkGenerator,
    ) -> Sequence[Self]:
        invocation_logs = []
        for trace in invocation_traces:
            elems = cls._from_invocation_trace(trace, link_generator, current_ref)
            current_ref = AuditLogRef(id=elems[-1].log_id, type="invocation")
            invocation_logs += elems
        return invocation_logs

    @classmethod
    def _from_invocation_trace(
        cls,
        invocation_trace: InvocationTrace,
        link_generator: AuditLogLinkGenerator,
        current_ref: AuditLogRef,
    ) -> Sequence[Self]:
        ref = link_generator.generate_ref("invocation")
        link_generator.generate_link(current_ref, ref, value=invocation_trace.input)

        children = []
        for child in invocation_trace.children:
            children += cls._from_invocation_trace(child, link_generator, ref)

        return [
            *children,
            cls(
                info=invocation_trace.info,
                expenditure=invocation_trace.expenditure,
                details=invocation_trace.details,
                log_id=ref.id,
            ),
        ]


class TaskEndAuditLog(BaseModel, frozen=True):
    """Audit record for ending of task."""

    type: Literal["task-end"] = "task-end"

    output: JsonValue
    llm_messages: Annotated[
        Sequence[LLMMessage],
        Field(description="LLM messages of context at the end."),
    ]
    llm_tokens: Annotated[LLMTokens, Field(description="LLM Tokens of the task.")]
    state: Annotated[JsonValue, Field(description="State of the task at the end")]
    log_id: AuditLogID = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="ID of Task End inside Audit.",
    )

    @classmethod
    def from_task_result(
        cls, result: TaskResult, current_ref: AuditLogRef, link_generator: AuditLogLinkGenerator
    ) -> Self:
        output = result.output
        link_generator.generate_link(current_ref, link_generator.task_end_ref, value=output)

        return cls(
            output=_to_json_value(output),
            llm_messages=result.context_state.llm_messages,
            llm_tokens=result.llm_tokens,
            state=_to_json_value(result.context_state.state),
            log_id=link_generator.task_end_ref.id,
        )


class TaskAuditLog(BaseModel, frozen=True):
    """Audit record for a task."""

    start: Annotated[TaskStartAuditLog, Field(description="AuditLog of task start.")]
    invocations: Annotated[
        Sequence[InvocationAuditLog], Field(description="AuditLog of invocations.")
    ]
    links: Annotated[Sequence[InvocationLinkAuditLog], Field(description="Links for AuditLog")]
    end: Annotated[TaskEndAuditLog, Field(description="AuditLog for task end.")]

    @classmethod
    def from_task_result(cls, task_result: TaskResult) -> Self:
        link_generator = AuditLogLinkGenerator()
        start = TaskStartAuditLog.from_task_result(task_result, link_generator)
        invocations = InvocationAuditLog.from_invocation_traces(
            task_result.traces,
            current_ref=link_generator.task_start_ref,
            link_generator=link_generator,
        )
        if invocations:
            current_ref = link_generator.get_ref(log_id=invocations[-1].log_id)
            if current_ref is None:
                raise RuntimeError("Implementation Error.")
        else:
            current_ref = link_generator.task_start_ref
        end = TaskEndAuditLog.from_task_result(task_result, current_ref, link_generator)
        links = link_generator.links

        return cls(start=start, invocations=invocations, end=end, links=links)
