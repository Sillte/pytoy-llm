from pytoy_llm.task.execution.models import TaskExecutionStatus
from pytoy_llm.task.session.models import TaskSessionQuery, TaskSessionStatus


def test_query_from_any_normalizes_defaults_and_single_values() -> None:
    query = TaskSessionQuery.from_any(status="idle", task_status="running")

    assert query.kind is None
    assert query.status == ("idle",)
    assert query.task_status == ("running",)


def test_query_from_any_preserves_multiple_values_and_none() -> None:
    statuses: tuple[TaskSessionStatus, ...] = ("idle", "running")
    task_statuses: tuple[TaskExecutionStatus, ...] = ("finished", "error")

    query = TaskSessionQuery.from_any(kind="interactive", status=statuses, task_status=task_statuses)

    assert query.kind == "interactive"
    assert query.status == statuses
    assert query.task_status == task_statuses

    empty_query = TaskSessionQuery.from_any()
    assert empty_query.status is None
    assert empty_query.task_status is None
