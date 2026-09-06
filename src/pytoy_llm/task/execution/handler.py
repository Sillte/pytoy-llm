from typing import Self, Sequence

from pytoy_llm.models.activities.llm_activities import LLMActivity
from pytoy_llm.shared.event import Event
from pytoy_llm.task.global_context import GlobalContext

from .factory import TaskExecutionFactory
from .manager import TaskExecutionManager
from .models import (
    TaskExecutionExit,
    TaskExecutionHooks,
    TaskExecutionID,
    TaskExecutionQuery,
    TaskExecutionStatus,
    TaskRequest,
)


class TaskExecutionHandler[T]:
    def __init__(self, id: TaskExecutionID, *, manager: TaskExecutionManager) -> None:
        self._id = id
        self._manager = manager

    @classmethod
    def create(cls, request: TaskRequest, *, manager: TaskExecutionManager | None = None) -> Self:
        if manager is None:
            manager = GlobalContext.get().execution_manager
        factory = TaskExecutionFactory()
        execution = factory.create(request)
        manager.register(execution)
        return cls(id=execution.id, manager=manager)

    @classmethod
    def query(cls, query: TaskExecutionQuery | None = None, *, manager: TaskExecutionManager | None = None) -> Sequence[Self]:
        query = query or TaskExecutionQuery()
        if manager is None:
            manager = GlobalContext.get().execution_manager
        executions = manager.select(query)
        return [cls(id=execution.id, manager=manager) for execution in executions]

    @property
    def status(self) -> TaskExecutionStatus | None:
        execution = self._manager.get(self._id)
        if execution is None:
            return None
        return execution.status

    def start(self, hooks: TaskExecutionHooks | None = None) -> None:
        hooks = hooks or TaskExecutionHooks.from_any()
        execution = self._manager.get(self._id)
        if execution is None:
            raise ValueError(f"`execution` does not exist; {self._id=}")

        execution.start(hooks=hooks)

    @property
    def id(self) -> TaskExecutionID:
        return self._id

    @property
    def on_exit(self) -> Event[TaskExecutionExit[T]]:
        execution = self._manager.get(self._id)
        if execution is None:
            raise ValueError(f"`execution` does not exist; {self._id=}")
        return execution.on_exit

    @property
    def on_activity(self) -> Event[LLMActivity]:
        execution = self._manager.get(self._id)
        if execution is None:
            raise ValueError(f"`execution` does not exist; {self._id=}")
        return execution.on_activity
