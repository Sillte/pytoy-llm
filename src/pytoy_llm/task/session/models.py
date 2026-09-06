import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from pytoy_llm.models.llm_activities import LLMActivity
from pytoy_llm.shared.event import Event, EventEmitter
from pytoy_llm.task.execution.models import TaskExecutionStatus
from pytoy_llm.task.models import TaskRequest
from pytoy_llm.task.models.exceptions import TaskExecutionException

type TaskSessionID = str
type TaskSessionStatus = Literal["idle", "running", "completed"]


@dataclass(frozen=True)
class TaskRecord:
    id: str
    request: TaskRequest
    status: TaskExecutionStatus
    output: Any = None
    exception: TaskExecutionException | None = None
    activities: tuple[LLMActivity, ...] = ()


@dataclass(frozen=True)
class TaskSessionRequest:
    id: TaskSessionID = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass(frozen=True)
class TaskSessionContext:
    pass


@dataclass(frozen=True)
class TaskSessionEvent:
    task_exit_emitter: EventEmitter = field(default_factory=EventEmitter)
    session_exit_emitter: EventEmitter[TaskSessionID] = field(default_factory=EventEmitter)
    activity_emitter: EventEmitter[LLMActivity] = field(default_factory=EventEmitter)

    @property
    def on_task_exit(self) -> Event:
        return self.task_exit_emitter.event

    @property
    def on_session_exit(self) -> Event[TaskSessionID]:
        return self.session_exit_emitter.event

    @property
    def on_activity(self) -> Event[LLMActivity]:
        return self.activity_emitter.event

    def emit_activity(self, activity: LLMActivity) -> None:
        self.activity_emitter.fire(activity)

    def dispose(self) -> None:
        self.task_exit_emitter.dispose()
        self.session_exit_emitter.dispose()
        self.activity_emitter.dispose()


@dataclass(frozen=True)
class TaskSessionQuery:
    status: Sequence[TaskSessionStatus] | None = None
    execution_status: Sequence[TaskExecutionStatus] | None = None
