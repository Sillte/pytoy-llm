from threading import RLock
from typing import Sequence

from pytoy_llm.task.execution.models import (
    TaskExecutionID,
    TaskExecutionQuery,
)
from pytoy_llm.task.execution.task_execution import TaskExecution


class TaskExecutionManager:
    def __init__(self):
        self._lock = RLock()
        self._executions: dict[TaskExecutionID, TaskExecution] = {}

    def register(self, execution: TaskExecution) -> None:
        with self._lock:
            self._executions[execution.id] = execution

        def _deregister(_):
            with self._lock:
                self._executions.pop(execution.id, None)

        execution.on_exit.subscribe(_deregister)

    def select(self, query: TaskExecutionQuery | None = None) -> Sequence[TaskExecution]:
        query = query or TaskExecutionQuery()
        with self._lock:
            executions = list(self._executions.values())
        if query.status:
            executions = [execution for execution in executions if execution.status in query.status]
        return executions

    def get(self, execution_id: TaskExecutionID) -> TaskExecution | None:
        with self._lock:
            return self._executions.get(execution_id)

    def get_running(self) -> Sequence[TaskExecution]:
        query = TaskExecutionQuery(status=["running"])

        return self.select(query)
