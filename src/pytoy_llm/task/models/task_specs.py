from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Self

from pytoy_llm.models import LLMEventEmitters, LLMParam, UsageLimit
from pytoy_llm.task.models import AgentInvocationSpec, LLMInvocationSpec
from pytoy_llm.task.models.context import CancelToken, ExecutionContext, TaskContextState
from pytoy_llm.task.models.exceptions import InvocationException, TaskCanceledException
from pytoy_llm.task.models.invocation_results import InvocationTrace
from pytoy_llm.task.models.invocation_specs import InvocationSpec
from pytoy_llm.task.models.metas import TaskSpecMeta
from pytoy_llm.task.models.task_results import TaskResult
from pytoy_llm.task.shared.outcome import Error, Outcome, Success


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
        self,
        task_input: Any,
        context_state: TaskContextState,
        emitters: LLMEventEmitters | None = None,
        llm_param: LLMParam | None = None,
        usage_limit: UsageLimit | None = None,
        cancel_token: CancelToken | None = None,
    ) -> Outcome[TaskResult[T], InvocationException]:
        """Run with base configuration and observe cancellation between invocations.

        Cancellation does not interrupt an invocation already in progress.

        Raises:
            TaskCanceledException: If cancellation is observed at an invocation boundary. The
                exception contains traces completed before that boundary; the invocation that
                observed cancellation may not be included.
        """
        connection = None
        execution_context = ExecutionContext(
            llm_param=llm_param,
            connection=connection,
            llm_messages=context_state.llm_messages,
            usage_limit=usage_limit,
            state=context_state.state,
            emitters=emitters or LLMEventEmitters(),
            cancel_token=cancel_token if cancel_token is not None else CancelToken(),
        )

        traces: list[InvocationTrace[Any]] = []

        def raise_if_cancelled() -> None:
            token = execution_context.cancel_token
            if token is not None and token.is_set():
                raise TaskCanceledException(completed_invocation_traces=traces)

        invocation_input = task_input
        for invocation_spec in self.invocation_specs:
            raise_if_cancelled()
            try:
                invocation_trace = invocation_spec.invoke(invocation_input, execution_context)
            except TaskCanceledException as exc:
                raise TaskCanceledException(
                    completed_invocation_traces=(
                        *traces,
                        *exc.completed_invocation_traces,
                    ),
                    exception_trace=exc.exception_trace,
                ) from exc
            except Exception as exc:
                return Error(
                    exception=InvocationException(
                        context=execution_context, invocation_exception=exc
                    )
                )
            raise_if_cancelled()

            invocation_result = invocation_trace.result
            if invocation_result.runtime_patch:
                execution_context = invocation_result.runtime_patch.apply(execution_context)
            if invocation_result.context_patch:
                execution_context = invocation_result.context_patch.apply(execution_context)
            traces.append(invocation_trace)
            invocation_input = invocation_result.output

        result = TaskResult(
            input=task_input,
            task_name=self.meta.name,
            output=invocation_result.output,
            traces=traces,
            context_state=TaskContextState.from_execution_context(execution_context),
        )
        return Success(result)

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
    def _to_task_spec_meta(
        cls, meta: TaskSpecMeta | str | None, invocation_specs: Sequence[InvocationSpec]
    ) -> TaskSpecMeta:
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
