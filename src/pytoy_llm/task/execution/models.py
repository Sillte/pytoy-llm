from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Self, Sequence

from pytoy_llm.task.models import TaskRequest, TaskResult
from pytoy_llm.task.models.exceptions import TaskExecutionException
from pytoy_llm.task.shared.outcome import Outcome

type TaskExecutionID = str
type TaskExecutionStatus = Literal["created", "running", "finished", "error"]


@dataclass(frozen=True)
class TaskExecutionContext[T]:
    request: TaskRequest[T]
    hooks: TaskExecutionHooks[T]


@dataclass(frozen=True)
class TaskExecutionExit[T]:
    id: TaskExecutionID
    outcome: Outcome[TaskResult[T], TaskExecutionException]


@dataclass(frozen=True)
class TaskExecutionHooks[T]:
    """Recommendation policy... Use `on_finish` rather than on_success / on_failure."""

    on_result: Callable[[TaskResult[T]], None]
    on_exception: Callable[[TaskExecutionException], None]

    @staticmethod
    def merge(hook1: TaskExecutionHooks[T], hook2: TaskExecutionHooks[T]) -> TaskExecutionHooks[T]:
        from dataclasses import fields

        merged_kwargs = {}
        for item in fields(TaskExecutionHooks):
            f1 = getattr(hook1, item.name)
            f2 = getattr(hook2, item.name)

            if not f1:  # (f1= None, f2=None), (f1=None, f2=Callable)
                merged_kwargs[item.name] = f2
            elif not f2:  # (f1=Callable, f2=None)
                merged_kwargs[item.name] = f1
            else:

                def _merged(f1=f1, f2=f2):  # デフォルト引数でクロージャの参照を固定
                    return lambda *a, **k: (f1(*a, **k), f2(*a, **k))

                merged_kwargs[item.name] = _merged()
        return TaskExecutionHooks(**merged_kwargs)

    @classmethod
    def from_any(
        cls,
        on_result: Callable[[TaskResult[T]], None] | None = None,
        on_exception: Callable[[TaskExecutionException], None] | None = None,
        on_output: Callable[[T], None] | None = None,
    ) -> Self:
        def _resolve_handle_output(
            target: Callable[[TaskResult[T]], None],
            on_output: Callable[[T], None],
        ) -> Callable[[TaskResult[T]], None]:

            def handler(result: TaskResult[T]) -> None:
                target(result)
                on_output(result.output)

            return handler

        on_result = on_result or (lambda _: None)
        on_exception = on_exception or (lambda _: None)
        on_result = _resolve_handle_output(on_result, on_output) if on_output else on_result
        return cls(on_result=on_result, on_exception=on_exception)


@dataclass(frozen=True)
class TaskExecutionQuery:
    status: Sequence[TaskExecutionStatus] | None = None

    @classmethod
    def from_any(cls, status: Sequence[TaskExecutionStatus] | None = None):
        return cls(status=status)
