from threading import Event

import pytest

from pytoy_llm.task.execution.manager import TaskExecutionManager
from pytoy_llm.task.models import ContextPatch, FunctionInvocationSpec, InvocationResult, TaskRequest
from pytoy_llm.task.session import TaskSessionHandler, TaskSessionManager


def wait_for_exit(session, task_id: str) -> None:
    completed = Event()
    session.on_task_exit.filter(lambda exit_entity: exit_entity.id == task_id).once().subscribe(lambda _: completed.set())
    assert completed.wait(timeout=1)


def test_session_propagates_context_between_task_executions() -> None:
    session_manager = TaskSessionManager()
    execution_manager = TaskExecutionManager()
    session = TaskSessionHandler.create(manager=session_manager, execution_manager=execution_manager)

    def remember(value: str, _context) -> InvocationResult[str]:
        return InvocationResult(output=value, context_patch=ContextPatch(state={"remembered": value}))

    first = session.submit(TaskRequest.from_invocation_spec(FunctionInvocationSpec.from_any(remember), "first"))
    wait_for_exit(session, first.id)

    observed: list[str] = []

    def read_context(value: str, context) -> str:
        observed.append(context.state["remembered"])
        return value

    second = session.submit(TaskRequest.from_invocation_spec(FunctionInvocationSpec.from_any(read_context), "second"))
    wait_for_exit(session, second.id)

    assert observed == ["first"]
    assert session.context_state.state == {"remembered": "first"}
    assert session.status == "idle"
    assert [record.id for record in session.records] == [first.id, second.id]


def test_failed_task_returns_session_to_idle_and_can_be_retried() -> None:
    session_manager = TaskSessionManager()
    execution_manager = TaskExecutionManager()
    session = TaskSessionHandler.create(manager=session_manager, execution_manager=execution_manager)
    attempts = 0

    def fail_once(value: str, _context) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("temporary failure")
        return value

    first = session.submit(TaskRequest.from_invocation_spec(FunctionInvocationSpec.from_any(fail_once), "retry"))
    wait_for_exit(session, first.id)
    second = session.submit(TaskRequest.from_invocation_spec(FunctionInvocationSpec.from_any(fail_once), "retry"))
    wait_for_exit(session, second.id)

    assert session.status == "idle"
    assert len(session.records) == 2
    assert session.records[0].exception is not None
    assert session.records[1].output == "retry"
    assert session.records[0].status == "error"
    assert session.records[1].status == "finished"


def test_completed_session_rejects_new_tasks_and_emits_session_exit() -> None:
    session_manager = TaskSessionManager()
    execution_manager = TaskExecutionManager()
    session = TaskSessionHandler.create(manager=session_manager, execution_manager=execution_manager)
    exited: list[str] = []
    session.on_session_exit.subscribe(exited.append)

    session.complete()

    assert session.status == "completed"
    assert exited == [session.id]


def test_terminate_removes_session_and_clears_retained_resources() -> None:
    session_manager = TaskSessionManager()
    execution_manager = TaskExecutionManager()
    session = TaskSessionHandler.create(manager=session_manager, execution_manager=execution_manager)

    task = session.submit(TaskRequest.from_invocation_spec(FunctionInvocationSpec.from_any(lambda value: value), "value"))
    wait_for_exit(session, task.id)
    session.terminate()

    assert session_manager.get(session.id) is None
    assert session.status is None


def test_terminate_does_not_accept_late_task_callbacks() -> None:
    session_manager = TaskSessionManager()
    execution_manager = TaskExecutionManager()
    session = TaskSessionHandler.create(manager=session_manager, execution_manager=execution_manager)
    started = Event()
    release = Event()

    def blocked(value: str) -> str:
        started.set()
        release.wait(timeout=1)
        return value

    session.submit(TaskRequest.from_invocation_spec(FunctionInvocationSpec.from_any(blocked), "value"))
    assert started.wait(timeout=1)
    session.terminate()
    release.set()

    assert session_manager.get(session.id) is None
    assert session.status is None
    with pytest.raises(ValueError, match="session.*does not exist"):
        _ = session.records


def test_session_manager_queries_registered_sessions() -> None:
    session_manager = TaskSessionManager()
    execution_manager = TaskExecutionManager()
    session = TaskSessionHandler.create(manager=session_manager, execution_manager=execution_manager)

    selected = TaskSessionHandler.query(manager=session_manager)

    assert [item.id for item in selected] == [session.id]
