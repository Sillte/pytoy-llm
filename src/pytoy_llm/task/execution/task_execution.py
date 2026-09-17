import time
import uuid
from dataclasses import dataclass, field
from functools import cached_property
from threading import RLock, Thread
from typing import Self

from pytoy_llm.shared.event import Event, EventEmitter
from pytoy_llm.task.execution.models import (
    TaskExecutionCancel,
    TaskExecutionExit,
    TaskExecutionHooks,
    TaskExecutionStart,
)
from pytoy_llm.task.models import ExecutionEvents, TaskRequest
from pytoy_llm.task.shared.outcome import is_error, is_success

from .models import TaskExecutionID, TaskExecutionStatus


@dataclass
class TaskExecution[T]:
    thread: Thread
    lock: RLock
    request: TaskRequest[T]
    events: ExecutionEvents
    start_emitter: EventEmitter[TaskExecutionStart]
    cancel_emitter: EventEmitter[TaskExecutionCancel]
    exit_emitter: EventEmitter[TaskExecutionExit[T]]
    status: TaskExecutionStatus = "created"
    id: TaskExecutionID = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=lambda: time.time())
    execution_exit: TaskExecutionExit[T] | None = None

    @classmethod
    def from_any(
        cls,
        thread: Thread,
        lock: RLock,
        task_request: TaskRequest[T],
        events: ExecutionEvents,
        exit_emitter: EventEmitter[TaskExecutionExit[T]],
        start_emitter: EventEmitter[TaskExecutionStart],
        cancel_emitter: EventEmitter[TaskExecutionCancel],
        id: TaskExecutionID,
    ) -> Self:
        return cls(
            thread=thread,
            lock=lock,
            request=task_request,
            events=events,
            start_emitter=start_emitter,
            cancel_emitter=cancel_emitter,
            exit_emitter=exit_emitter,
            id=id,
        )

    def start(self, hooks: TaskExecutionHooks) -> None:
        with self.lock:
            if self.status != "created":
                raise RuntimeError(
                    f"Start of `Task` is accepted only when `created`, but {self.status=}"
                )

            self.on_exit.map(lambda exit_entity: exit_entity.outcome).filter(
                is_success
            ).once().subscribe(lambda _: self.to_finish())
            self.on_exit.map(lambda exit_entity: exit_entity.outcome).filter(
                is_error
            ).once().subscribe(lambda _: self.to_error())
            self.on_exit.map(lambda exit_entity: exit_entity.outcome).filter(is_success).map(
                lambda success: success.value
            ).once().subscribe(hooks.on_result)
            self.on_exit.map(lambda exit_entity: exit_entity.outcome).filter(is_error).map(
                lambda error: error.exception
            ).once().subscribe(hooks.on_exception)

            self.status = "running"
            try:
                self.start_emitter.fire(TaskExecutionStart(id=self.id))
            finally:
                self.thread.start()

    def cancel(self) -> None:
        with self.lock:
            if self.status != "created":
                raise RuntimeError(
                    f"Cancel of `Task` is accepted only when `created`, but {self.status=}"
                )
            self.status = "canceled"
            self.cancel_emitter.fire(TaskExecutionCancel(id=self.id))

    @cached_property
    def on_exit(self) -> Event[TaskExecutionExit[T]]:
        return self.exit_emitter.event

    @cached_property
    def on_start(self) -> Event[TaskExecutionStart]:
        return self.start_emitter.event

    @cached_property
    def on_cancel(self) -> Event[TaskExecutionCancel]:
        return self.cancel_emitter.event

    @property
    def on_activity(self):
        return self.events.activity_emitter.event

    def record_exit(self, exit_entity: TaskExecutionExit[T]) -> None:
        with self.lock:
            self.execution_exit = exit_entity
            self.exit_emitter.fire(exit_entity)

    def dispose(self):
        self.start_emitter.dispose()
        self.cancel_emitter.dispose()
        self.exit_emitter.dispose()
        self.events.dispose()

    def to_finish(self) -> None:
        with self.lock:
            self.status = "finished"

    def to_error(self) -> None:
        with self.lock:
            self.status = "error"
