from typing import Self, Sequence

from pytoy_llm.models.llm_activities.llm_activities import LLMActivity
from pytoy_llm.shared.event import Event
from pytoy_llm.task.global_context import GlobalContext

from .factory import TaskExecutionFactory
from .manager import TaskExecutionManager
from .models import (
    TaskExecutionCancel,
    TaskExecutionExit,
    TaskExecutionHooks,
    TaskExecutionID,
    TaskExecutionQuery,
    TaskExecutionStart,
    TaskExecutionStatus,
    TaskRequest,
)
from .task_execution import TaskExecution


class TaskExecutionHandler[T]:
    def __init__(
        self,
        id: TaskExecutionID,
        execution: TaskExecution,
        *,
        manager: TaskExecutionManager,
    ) -> None:
        self._id = id
        self._execution = execution
        self._manager = manager

    @classmethod
    def create(cls, request: TaskRequest, *, manager: TaskExecutionManager | None = None) -> Self:
        if manager is None:
            manager = GlobalContext.get().execution_manager
        factory = TaskExecutionFactory()
        execution = factory.create(request)
        manager.register(execution)
        return cls(id=execution.id, execution=execution, manager=manager)

    @classmethod
    def query(
        cls, query: TaskExecutionQuery | None = None, *, manager: TaskExecutionManager | None = None
    ) -> Sequence[Self]:
        query = query or TaskExecutionQuery()
        if manager is None:
            manager = GlobalContext.get().execution_manager
        executions = manager.select(query)
        return [
            cls(id=execution.id, execution=execution, manager=manager) for execution in executions
        ]

    @property
    def status(self) -> TaskExecutionStatus | None:
        return self._execution.status

    @property
    def execution_exit(self) -> TaskExecutionExit[T] | None:
        return self._execution.execution_exit

    def start(self, hooks: TaskExecutionHooks | None = None) -> None:
        hooks = hooks or TaskExecutionHooks.from_any()
        self._execution.start(hooks=hooks)

    def cancel(self) -> None:
        self._execution.cancel()

    @property
    def id(self) -> TaskExecutionID:
        return self._id

    @property
    def on_exit(self) -> Event[TaskExecutionExit[T]]:
        return self._execution.on_exit

    @property
    def on_start(self) -> Event[TaskExecutionStart]:
        return self._execution.on_start

    @property
    def on_cancel(self) -> Event[TaskExecutionCancel]:
        return self._execution.on_cancel

    @property
    def on_activity(self) -> Event[LLMActivity]:
        return self._execution.on_activity
