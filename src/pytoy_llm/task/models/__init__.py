from __future__ import annotations

from pytoy_llm.models import LLMMessageHistory

from pydantic import BaseModel, Field

from typing import Annotated, Sequence, Self, Any

from pytoy_llm.task.models.schemas import LLMTaskArgument
from pytoy_llm.task.models.schemas import LLMTaskSpecMeta
from pytoy_llm.task.models.schemas import InvocationSpecMeta # NOQA
from pytoy_llm.task.models.schemas import InvocationRecords
from pytoy_llm.task.models.schemas import LLMTaskRecord

from pytoy_llm.task.models.invocations import FunctionInvocationSpec
from pytoy_llm.task.models.invocations import LLMInvocationSpec
from pytoy_llm.task.models.invocations import AgentInvocationSpec
from pytoy_llm.task.models.invocations import SelectedInvocationSpec  

from pytoy_llm.task.models.llm_facade import LLMFacade
from pytoy_llm.task.models.context import LLMTaskContext


class LLMTaskSpec[S: BaseModel | str](BaseModel):
    """
    Represents a higher-level Task composed of multiple InvocationSpecs.
    """

    invocation_specs: Annotated[
        Sequence[FunctionInvocationSpec | LLMInvocationSpec | AgentInvocationSpec | SelectedInvocationSpec],
        Field(description="Ordered list of steps or conditional branches"),
    ]
    meta: Annotated[LLMTaskSpecMeta, Field(description="Meta data for the task.")]

    @property
    def output_spec(self) -> S | None:
        return self.invocation_specs[-1].output_spec if self.invocation_specs else None  # type: ignore

    def run(
        self, task_input: Any, history: LLMMessageHistory | None = None
    ) -> LLMTaskRecord[S]:
        task_agument = LLMTaskArgument(initial_history=history, initial_input=task_input)
        llm_facade = LLMFacade()
        task_context = LLMTaskContext(task_meta=self.meta, task_argument=task_agument, llm_facade=llm_facade)

        invocation_records: InvocationRecords = InvocationRecords()

        input = task_input
        for invocation_spec in self.invocation_specs:
            records = invocation_spec.invoke(input, task_context)
            # Update of the `repository` based on the `InvocationRecords`
            task_context.repository.update(records.repository_updates)
            invocation_records = invocation_records.updated(records)
            input = invocation_records.output
        return LLMTaskRecord(
            task_name=self.meta.name,
            output=invocation_records.output,
            invocation_records=invocation_records.entries,
            repository_snapshot=task_context.repository.snapshot,
        )  # type: ignore

    @property
    def name(self) -> str:
        return self.meta.name

    @classmethod
    def from_single_spec(
        cls, meta: str | LLMTaskSpecMeta, invocation_spec: LLMInvocationSpec[S], 
    ) -> Self:
        """Utility function for construction the task with 1 LLMInvocation."""
        if isinstance(meta, str):
            meta = LLMTaskSpecMeta(name=meta, intent=invocation_spec.meta.intent)
        return cls(
            meta=meta,
            invocation_specs=[invocation_spec],
        )  # type: ignore
