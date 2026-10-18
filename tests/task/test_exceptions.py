from pytoy_llm.task.models.exceptions import TaskUnknownException


def test_task_unknown_exception_captures_stack_trace() -> None:
    def fail_task() -> None:
        raise RuntimeError("task failed")

    try:
        fail_task()
    except RuntimeError as exc:
        task_exception = TaskUnknownException(exc)

    assert "fail_task" in task_exception.stack_trace
    assert "RuntimeError: task failed" in task_exception.stack_trace
    assert task_exception.stack_trace in str(task_exception)
