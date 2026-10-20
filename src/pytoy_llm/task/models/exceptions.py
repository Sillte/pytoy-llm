import traceback
from collections.abc import Sequence
from typing import Any, Literal

from pytoy_llm.task.models.context import ExecutionContext
from pytoy_llm.task.models.invocation_results import InvocationResult, InvocationTrace


class TaskCanceledException(Exception):
    """Raised when a task observes cancellation, with its completed invocation history."""

    def __init__(
        self,
        completed_invocation_traces: Sequence[InvocationTrace[Any]] = (),
        *,
        exception_trace: str | None = None,
    ) -> None:
        self.completed_invocation_traces = tuple(completed_invocation_traces)
        self._exception_trace = (
            exception_trace
            if exception_trace is not None
            else "".join(traceback.format_stack()[:-1])
        )
        super().__init__("Task execution was canceled.")

    @property
    def exception_trace(self) -> str:
        return self._exception_trace


class InvocationException(Exception):
    def __init__(self, context: ExecutionContext, invocation_exception: Exception):
        self.context = context
        self.invocation_exception = invocation_exception
        self.stack_trace = "".join(traceback.format_exception(invocation_exception))

    def __str__(self):
        return f"{self.context=}\n{self.invocation_exception=}\n{self.stack_trace}"


class InvocationHookException(Exception):
    def __init__(
        self,
        hook_name: Literal["on_start", "on_result", "on_exception"],
        hook_exception: Exception,
        *,
        result: InvocationResult[Any] | None = None,
        exception: Exception | None = None,
    ) -> None:
        self.hook_name = hook_name
        self.hook_exception = hook_exception
        self.result = result
        self.exception = exception
        self.stack_trace = "".join(traceback.format_exception(hook_exception))
        super().__init__(f"{hook_name} hook failed: {hook_exception}")


class TaskUnknownException(Exception):
    def __init__(self, exception: Exception):
        self.exception = exception
        self.stack_trace = "".join(traceback.format_exception(exception))

    def __str__(self):
        return f"{self.exception=}\n{self.stack_trace}"


type TaskExecutionException = InvocationException | TaskUnknownException | TaskCanceledException
