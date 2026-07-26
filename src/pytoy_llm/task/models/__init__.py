from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, Self

from pydantic import BaseModel, Field

from pytoy_llm.llm_facade import LLMFacade
from pytoy_llm.task.models.context import LLMTaskContext
from pytoy_llm.task.models.invocations import (
    AgentInvocationSpec,
    FunctionInvocationSpec,
    LLMInvocationSpec,
    SelectedInvocationSpec,
)
from pytoy_llm.task.models.schemas import (
    InvocationSpecMeta,  # NOQA
    LLMTaskResult,
    LLMTaskSpecMeta,
    TaskContextState,
)


class LLMTaskSpec[T](BaseModel):
    """
    Represents a higher-level Task composed of multiple InvocationSpecs.
    """

    invocation_specs: Annotated[
        Sequence[FunctionInvocationSpec | LLMInvocationSpec | AgentInvocationSpec | SelectedInvocationSpec],
        Field(description="Ordered list of steps or conditional branches"),
    ]
    meta: Annotated[LLMTaskSpecMeta, Field(description="Meta data for the task.")]

    @property
    def output_type(self) -> type[BaseModel] | None:
        return self.invocation_specs[-1].output_type if self.invocation_specs else None  # type: ignore

    def run(self, task_input: Any, context_state: TaskContextState) -> LLMTaskResult[T]:
        llm_facade = LLMFacade()
        task_context = LLMTaskContext(
            llm_facade=llm_facade,
            llm_messages=context_state.llm_messages,
            repository=context_state.repository,
        )

        traces = []

        input = task_input
        for invocation_spec in self.invocation_specs:
            invocation_result = invocation_spec.invoke(input, task_context)
            if invocation_result.runtime_patch:
                task_context = invocation_result.runtime_patch.patch(task_context)
            if invocation_result.context_patch:
                task_context = invocation_result.context_patch.patch(task_context)
            if invocation_result.trace:
                traces.append(invocation_result.trace)
            input = invocation_result.output

        return LLMTaskResult(
            task_name=self.meta.name,
            output=invocation_result.output,
            traces=traces,
            task_context=task_context,
        )

    @property
    def name(self) -> str:
        return self.meta.name

    @classmethod
    def from_single_spec(
        cls,
        meta: str | LLMTaskSpecMeta,
        invocation_spec: LLMInvocationSpec,
    ) -> Self:
        """Utility function for construction the task with 1 LLMInvocation."""
        if isinstance(meta, str):
            meta = LLMTaskSpecMeta(name=meta, intent=invocation_spec.meta.intent)
        return cls(
            meta=meta,
            invocation_specs=[invocation_spec],
        )
