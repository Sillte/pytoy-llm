from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from pytoy_llm.task.execution.manager import TaskExecutionManager


class GlobalContext:
    _instance: ClassVar["GlobalContext | None"] = None

    @classmethod
    def get(cls) -> GlobalContext:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @cached_property
    def execution_manager(self) -> TaskExecutionManager:
        from pytoy_llm.task.execution.manager import TaskExecutionManager

        return TaskExecutionManager()

    @cached_property
    def session_manager(self):
        from pytoy_llm.task.session.manager import TaskSessionManager

        return TaskSessionManager()
