from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, replace
from threading import RLock
from typing import Self

from pytoy_llm.models.llm_activities import LLMActivity
from pytoy_llm.shared.event import Disposable
from pytoy_llm.task.execution import TaskExecutionHandler
from pytoy_llm.task.execution.manager import TaskExecutionManager
from pytoy_llm.task.execution.models import TaskExecutionHooks, TaskExecutionID
from pytoy_llm.task.models import TaskContextState, TaskRequest
from pytoy_llm.task.shared.outcome import is_success

from .models import TaskRecord, TaskSessionEvent, TaskSessionID, TaskSessionStatus


@dataclass
class TaskSession:
    context_state: TaskContextState = field(default_factory=TaskContextState)
    records: dict[TaskExecutionID, TaskRecord] = field(default_factory=dict)
    events: TaskSessionEvent = field(default_factory=TaskSessionEvent)

    status: TaskSessionStatus = "idle"
    timestamp: float = field(default_factory=time.time)

    id: TaskSessionID = field(default_factory=lambda: str(uuid.uuid4()))
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)
    _subscriptions: list[Disposable] = field(default_factory=list, init=False, repr=False)
    _terminated: bool = field(default=False, init=False, repr=False)

    @classmethod
    def from_any(
        cls,
        context_state: TaskContextState | None = None,
        id: TaskSessionID | None = None,
    ) -> Self:
        return cls(
            context_state=context_state or TaskContextState(),
            id=id or str(uuid.uuid4()),
        )

    def submit(self, request: TaskRequest, *, execution_manager: TaskExecutionManager) -> TaskExecutionHandler:
        with self._lock:
            if self._terminated:
                raise RuntimeError("Cannot submit to a terminated session.")
            if self.status != "idle":
                raise RuntimeError(f"Task submission is accepted only when `idle`, but {self.status=}")
            if request.context_state is None:
                request = TaskRequest(
                    spec=request.spec,
                    input=request.input,
                    context_state=self.context_state,
                    activity_sink=request.activity_sink,
                    id=request.id,
                )

            handler = TaskExecutionHandler.create(request=request, manager=execution_manager)
            execution_id = handler.id
            self.records[execution_id] = TaskRecord(id=execution_id, request=request, status="running")
            self._subscriptions.append(handler.on_exit.subscribe(self._on_exit))
            self._subscriptions.append(
                handler.on_activity.subscribe(lambda activity: self._on_activity(execution_id, activity))
            )
            self.status = "running"
            handler.start(TaskExecutionHooks.from_any())
            return handler

    def _on_exit(self, exit_entity) -> None:
        with self._lock:
            if self._terminated:
                return
            record = self.records[exit_entity.id]
            if is_success(exit_entity.outcome):
                record = replace(
                    record,
                    status="finished",
                    output=exit_entity.outcome.value.output,
                )
                self.context_state = exit_entity.outcome.value.context_state
            else:
                record = replace(record, status="error", exception=exit_entity.outcome.exception)
            self.records[exit_entity.id] = record
            self.status = "idle"
            self.events.task_exit_emitter.fire(exit_entity)

    def _on_activity(self, task_id: TaskExecutionID, activity: LLMActivity) -> None:
        with self._lock:
            if self._terminated:
                return
            record = self.records[task_id]
            self.records[task_id] = replace(record, activities=(*record.activities, activity))
            self.events.emit_activity(activity)

    def complete(self) -> None:
        with self._lock:
            if self._terminated:
                raise RuntimeError("Cannot complete a terminated session.")
            if self.status != "idle":
                raise RuntimeError(f"Session completion is accepted only when `idle`, but {self.status=}")
            self.status = "completed"
            self.events.session_exit_emitter.fire(self.id)

    def terminate(self) -> None:
        with self._lock:
            if self._terminated:
                return
            self._terminated = True
            for subscription in self._subscriptions:
                subscription.dispose()
            self._subscriptions.clear()
            self.records.clear()
            self.events.dispose()
