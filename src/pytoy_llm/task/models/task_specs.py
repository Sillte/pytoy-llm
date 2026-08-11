from collections.abc import Sequence
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pytoy_llm.connection_configuration import DEFAULT_NAME
from pytoy_llm.event_sinks import EventSinkProtocol
from pytoy_llm.models.llm_metas import LLMParam
from pytoy_llm.task.models.context import ExecutionContext, TaskContextState
from pytoy_llm.task.models.invocation_specs import (
    AgentInvocationSpec,
    FunctionInvocationSpec,
    LLMInvocationSpec,
    SelectedInvocationSpec,
)
from pytoy_llm.task.models.metas import TaskSpecMeta
from pytoy_llm.task.models.task_results import TaskResult


class TaskSpec[T](BaseModel, frozen=True):
    """
    Represents a higher-level Task composed of multiple InvocationSpecs.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    invocation_specs: Annotated[
        Sequence[FunctionInvocationSpec | LLMInvocationSpec | AgentInvocationSpec | SelectedInvocationSpec],
        Field(description="Ordered list of steps or conditional branches"),
    ]
    meta: Annotated[TaskSpecMeta, Field(description="Meta data for the task.")]
    output_type: Annotated[type[T] | None, Field(description="output type of the task.", exclude=False)] = None

    @model_validator(mode="before")
    @classmethod
    def infer_output_type(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values

        specs = values.get("invocation_specs")
        if not specs:
            raise ValueError("invocation_specs must contain at least one InvocationSpec.")

        inferred = specs[-1].output_type

        output_type = values.get("output_type")
        if output_type is None:
            values["output_type"] = inferred
        elif output_type is not inferred:
            msg = f"output_type ({output_type}) does not match the last InvocationSpec output_type ({inferred})."
            raise ValueError(msg)
        return values

    def run(
        self, task_input: Any, context_state: TaskContextState, event_sink: EventSinkProtocol | None = None
    ) -> TaskResult[T]:
        llm_param = LLMParam()
        connection = DEFAULT_NAME
        execution_context = ExecutionContext(
            llm_param=llm_param,
            connection=connection,
            llm_messages=context_state.llm_messages,
            state=context_state.state,
            event_sink=event_sink,
        )

        traces = []

        invocation_input = task_input
        for invocation_spec in self.invocation_specs:
            invocation_result = invocation_spec.invoke(invocation_input, execution_context)
            if invocation_result.runtime_patch:
                execution_context = invocation_result.runtime_patch.patch(execution_context)
            if invocation_result.context_patch:
                execution_context = invocation_result.context_patch.patch(execution_context)
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
        meta: str | TaskSpecMeta,
        invocation_spec: FunctionInvocationSpec | LLMInvocationSpec | AgentInvocationSpec | SelectedInvocationSpec,
    ) -> Self:
        """Utility function for construction the task with 1 LLMInvocation."""
        if isinstance(meta, str):
            meta = TaskSpecMeta(name=meta, intent=invocation_spec.meta.intent)
        return cls(
            meta=meta,
            invocation_specs=[invocation_spec],
        )
