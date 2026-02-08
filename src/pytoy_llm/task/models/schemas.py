from __future__ import annotations
import uuid
from pytoy_llm.models import LLMMessageHistory


from pydantic import BaseModel, Field


from typing import Annotated, Any, Literal, Mapping, Sequence


class LLMTaskArgument[T: BaseModel | str](BaseModel):
    initial_input: Annotated[
        str | T, Field(description="The initial input to the task / first step")
    ]
    initial_history: Annotated[
        LLMMessageHistory | None, Field(description="Message given_history for context")
    ] = None


class LLMTaskSpecMeta(BaseModel):
    name: Annotated[str, Field(description="Human-readable task name")]
    intent: Annotated[str | None, Field(description="What the overall task is intended to do")] = (
        None
    )
    rules: Annotated[
        Sequence[str] | None, Field(description="Guiding rules or constraints for this task")
    ] = None
    description: Annotated[
        str | None, Field(description="Optional longer explanation of the task purpose")
    ] = None


class InvocationSpecMeta(BaseModel):
    name: Annotated[str, Field(description="Name of the invocation step.")]
    intent: Annotated[str, Field(description="Intent of this invocation step.")]


class InvocationMeta(BaseModel, frozen=True):
    spec_meta: Annotated[
        InvocationSpecMeta, Field(description="Metadata about this invocation spec")
    ]
    kind: Annotated[Literal["llm", "agent", "function", "selector"], Field(description="Type of invocation")]
    started_at: Annotated[float, Field(description="Start time of this invocation")]
    ended_at: Annotated[float, Field(description="End time of this invocation")]


    @property
    def spec_name(self) -> str:
        return self.spec_meta.name

    @property
    def intent(self) -> str:
        return self.spec_meta.intent


class InvocationRecord[T: Any](BaseModel, frozen=True):
    id: Annotated[str, Field(description="Unique identifier for this invocation record")] = Field(
        default_factory=lambda: str(uuid.uuid1())
    )
    meta: Annotated[InvocationMeta, Field(description="Metadata about this invocation")]
    input: Annotated[Any, Field(description="Input value given to this invocation")]
    output: Annotated[Any, Field(description="Output value produced by this invocation")]


class InvocationRecords(BaseModel):
    entries: Sequence[InvocationRecord] = Field(default_factory=list)
    repository_updates: Mapping[str, Any] = Field(default_factory=dict)

    def updated(self, other: "InvocationRecords") -> "InvocationRecords":
        return InvocationRecords(entries=list(self.entries) + list(other.entries),
                                 repository_updates={**self.repository_updates, **other.repository_updates})

    @property
    def output(self) -> Any:
        return self.entries[-1].output if self.entries else None


class InvocationEffect(BaseModel, frozen=True):
    output: Annotated[Any, Field(description="Output value produced by this invocation effect")]
    repository_updates: Annotated[
        Mapping[str, Any], Field(description="Updates to be applied to the task repository")
    ]

    @classmethod
    def from_any(cls, arg: Any) -> "InvocationEffect":
        if isinstance(arg, InvocationEffect):
            return arg
        else:
            return InvocationEffect(output=arg, repository_updates={})


class LLMTaskRecord[T: BaseModel | str](BaseModel, frozen=True):
    id: Annotated[str, Field(description="Unique identifier for this Task record")] = Field(
        default_factory=lambda: str(uuid.uuid1())
    )
    task_name: Annotated[str, Field(description="Name of the executed task")]
    output: Annotated[T, Field(description="Final output produced by the task")]

    invocation_records: Annotated[
        Sequence[InvocationRecord],
        Field(description="Ordered history of all invocations executed in this task"),
    ]

    repository_snapshot: Annotated[
        dict[str, Any], Field(description="Final snapshot of the shared state repository")
    ]