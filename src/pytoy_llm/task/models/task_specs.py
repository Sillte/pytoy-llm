from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Self

from pytoy_llm.activity_sinks import ActivitySinkProtocol
from pytoy_llm.task.models import AgentInvocationSpec, LLMInvocationSpec
from pytoy_llm.task.models.context import ExecutionContext, TaskContextState
from pytoy_llm.task.models.invocation_specs import (
    InvocationSpec,
)
from pytoy_llm.task.models.metas import TaskSpecMeta
from pytoy_llm.task.models.task_results import TaskResult


@dataclass(frozen=True)
class TaskSpec[T]:
    """
    Represents a higher-level Task composed of multiple InvocationSpecs.
    """

    invocation_specs: Sequence[InvocationSpec]
    output_type: type[T]
    meta: TaskSpecMeta = field(default_factory=lambda: TaskSpecMeta(name="NoTaskName"))

    def __post_init__(self):
        if not self.invocation_specs:
            raise ValueError("Empty invocation specs is not allowed.")

    def run(
        self, task_input: Any, context_state: TaskContextState, activity_sink: ActivitySinkProtocol | None = None
    ) -> TaskResult[T]:
        llm_param = None
        connection = None
        execution_context = ExecutionContext(
            llm_param=llm_param,
            connection=connection,
            llm_messages=context_state.llm_messages,
            state=context_state.state,
            activity_sink=activity_sink,
        )

        traces = []

        invocation_input = task_input
        for invocation_spec in self.invocation_specs:
            invocation_result = invocation_spec.invoke(invocation_input, execution_context)
            if invocation_result.runtime_patch:
                execution_context = invocation_result.runtime_patch.apply(execution_context)
            if invocation_result.context_patch:
                execution_context = invocation_result.context_patch.apply(execution_context)
            if invocation_result.trace:
                traces.append(invocation_result.trace)
            invocation_input = invocation_result.output

        return TaskResult(
            task_name=self.meta.name,
            output=invocation_result.output,
            traces=traces,
            context_state=TaskContextState.from_execution_context(execution_context),
        )

    @property
    def name(self) -> str:
        return self.meta.name

    @classmethod
    def from_single_spec(
        cls,
        invocation_spec: InvocationSpec,
        meta: TaskSpecMeta | str | None = None,
        output_type: type | None = None,
    ) -> Self:
        """Utility function for construction the task with 1 LLMInvocation."""

        meta = cls._to_task_spec_meta(meta, invocation_specs=[invocation_spec])
        if output_type is None:
            output_type = cls._infer_output_type([invocation_spec])

        return cls(
            meta=meta,
            output_type=output_type,
            invocation_specs=[invocation_spec],
        )

    @classmethod
    def from_specs(
        cls,
        invocation_specs: Sequence[InvocationSpec],
        meta: TaskSpecMeta | str | None = None,
        output_type: type | None = None,
    ) -> Self:
        """Utility function for construction the task with 1 LLMInvocation."""
        if not invocation_specs:
            raise ValueError("Empty invocation specs is not allowed.")

        meta = cls._to_task_spec_meta(meta, invocation_specs=invocation_specs)
        if output_type is None:
            output_type = cls._infer_output_type(invocation_specs)

        return cls(
            meta=meta,
            output_type=output_type,
            invocation_specs=invocation_specs,
        )

    @classmethod
    def _to_task_spec_meta(cls, meta: TaskSpecMeta | str | None, invocation_specs: Sequence[InvocationSpec]) -> TaskSpecMeta:
        if isinstance(meta, str):
            meta = TaskSpecMeta(name=meta, intent=invocation_specs[-1].meta.intent)
        elif meta is None:
            meta = TaskSpecMeta(name="NoTaskName")
        return meta

    @classmethod
    def _infer_output_type(cls, specs: Sequence[InvocationSpec]) -> type:
        if isinstance(specs[-1], (LLMInvocationSpec, AgentInvocationSpec)):
            return specs[-1].output_type
        return object
