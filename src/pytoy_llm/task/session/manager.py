from threading import RLock
from typing import Sequence

from .models import TaskSessionID, TaskSessionQuery
from .session import TaskSession


class TaskSessionManager:
    def __init__(self) -> None:
        self._lock = RLock()
        self._sessions: dict[TaskSessionID, TaskSession] = {}

    def register(self, session: TaskSession) -> None:
        with self._lock:
            self._sessions[session.id] = session

    def select(self, query: TaskSessionQuery | None = None) -> Sequence[TaskSession]:
        query = query or TaskSessionQuery()
        with self._lock:
            sessions = list(self._sessions.values())
        if query.kind:
            sessions = [session for session in sessions if session.kind == query.kind]
        if query.status:
            sessions = [session for session in sessions if session.status in query.status]
        if query.task_status:
            sessions = [
                session
                for session in sessions
                if any(record.status in query.task_status for record in session.records.values())
            ]
        return sessions

    def get(self, session_id: TaskSessionID) -> TaskSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def remove(self, session_id: TaskSessionID) -> TaskSession | None:
        with self._lock:
            return self._sessions.pop(session_id, None)
