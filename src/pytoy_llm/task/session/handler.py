from typing import Self, Sequence

from pytoy_llm.models.llm_activities import LLMActivity
from pytoy_llm.shared.event import Event
from pytoy_llm.task.execution.handler import TaskExecutionHandler
from pytoy_llm.task.execution.manager import TaskExecutionManager
from pytoy_llm.task.execution.models import TaskExecutionExit
from pytoy_llm.task.global_context import GlobalContext
from pytoy_llm.task.models import TaskContextState, TaskRequest

from .manager import TaskSessionManager
from .models import TaskRecord, TaskSessionID, TaskSessionQuery, TaskSessionRequest, TaskSessionStatus
from .session import TaskSession


class TaskSessionHandler:
    def __init__(self, id: TaskSessionID, *, manager: TaskSessionManager, execution_manager: TaskExecutionManager) -> None:
        self._id = id
        self._manager = manager
        self._execution_manager = execution_manager

    @classmethod
    def create(
        cls,
        request: TaskSessionRequest | None = None,
        *,
        manager: TaskSessionManager | None = None,
        execution_manager: TaskExecutionManager | None = None,
    ) -> Self:
        manager = manager or GlobalContext.get().session_manager
        execution_manager = execution_manager or GlobalContext.get().execution_manager

        request = request or TaskSessionRequest.from_any()
        session = TaskSession.from_any(context_state=request.context_state, kind=request.kind)
        manager.register(session)
        handler = cls(id=session.id, manager=manager, execution_manager=execution_manager)
        return handler

    @classmethod
    def query(
        cls,
        query: TaskSessionQuery | None = None,
        *,
        manager: TaskSessionManager | None = None,
        execution_manager: TaskExecutionManager | None = None,
    ) -> Sequence[Self]:
        manager = manager or GlobalContext.get().session_manager
        execution_manager = execution_manager or GlobalContext.get().execution_manager
        return [cls(id=session.id, manager=manager, execution_manager=execution_manager) for session in manager.select(query)]

    @property
    def id(self) -> TaskSessionID:
        return self._id

    @property
    def status(self) -> TaskSessionStatus | None:
        session = self._manager.get(self._id)
        return session.status if session else None

    @property
    def context_state(self) -> TaskContextState:
        session = self._require_session()
        return session.context_state

    def submit(self, request: TaskRequest) -> TaskExecutionHandler:
        session = self._require_session()
        return session.submit(request, execution_manager=self._execution_manager)

    @property
    def records(self) -> Sequence[TaskRecord]:
        return tuple(self._require_session().records.values())

    @property
    def on_exit(self) -> Event[TaskExecutionExit]:
        return self._require_session().events.on_task_exit

    @property
    def on_task_exit(self) -> Event[TaskExecutionExit]:
        return self._require_session().events.on_task_exit

    @property
    def on_session_exit(self) -> Event[TaskSessionID]:
        return self._require_session().events.on_session_exit

    @property
    def on_activity(self) -> Event[LLMActivity]:
        return self._require_session().events.on_activity

    def complete(self) -> None:
        self._require_session().complete()

    def terminate(self) -> None:
        session = self._manager.remove(self._id)
        if session is None:
            return
        session.terminate()

    def _require_session(self) -> TaskSession:
        session = self._manager.get(self._id)
        if session is None:
            raise ValueError(f"`session` does not exist; {self._id=}")
        return session
