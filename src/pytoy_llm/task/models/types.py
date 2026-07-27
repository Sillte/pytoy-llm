from typing import Any, Callable

from pytoy_llm.task.models.context import ExecutionContext
from pytoy_llm.task.models.invocation_results import InvocationResult

type InvocationCallable[T] = Callable[[Any, ExecutionContext], T | InvocationResult[T]]
