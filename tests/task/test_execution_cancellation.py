from threading import Event

from pytoy_llm.task.execution import TaskExecutionHandler, TaskExecutionHooks
from pytoy_llm.task.execution.manager import TaskExecutionManager
from pytoy_llm.task.models import (
    FunctionInvocationSpec,
    TaskCanceledException,
    TaskRequest,
    TaskSpec,
)
from pytoy_llm.task.shared.outcome import is_error, is_success


def test_async_execution_uses_request_id_for_task_result() -> None:
    request = TaskRequest(
        spec=TaskSpec.from_single_spec(
            FunctionInvocationSpec.from_any(lambda value: value),
            output_type=str,
        ),
        input="input",
    )
    task = TaskExecutionHandler.create(request, manager=TaskExecutionManager())
    exited = Event()
    task.on_exit.once().subscribe(lambda _: exited.set())

    task.start(TaskExecutionHooks.from_any())

    assert exited.wait(timeout=1)
    exit_entity = task.execution_exit
    assert exit_entity is not None
    assert is_success(exit_entity.outcome)
    result = exit_entity.outcome.value
    assert result.id == request.id
    assert result.traces[0].result.context.task_id == request.id


def test_cancel_running_execution_at_invocation_boundary() -> None:
    invocation_started = Event()
    allow_invocation_to_finish = Event()
    cancellation_observed: list[bool] = []
    second_invocation_called: list[bool] = []

    def blocking_invocation(value, context):
        invocation_started.set()
        allow_invocation_to_finish.wait(timeout=2)
        cancellation_observed.append(context.cancel_token.is_set())
        return value

    def second_invocation(value):
        second_invocation_called.append(True)
        return value

    request = TaskRequest(
        spec=TaskSpec.from_specs(
            [
                FunctionInvocationSpec.from_any(blocking_invocation),
                FunctionInvocationSpec.from_any(second_invocation),
            ]
        ),
        input="value",
    )
    manager = TaskExecutionManager()
    task = TaskExecutionHandler.create(request, manager=manager)
    events: list[str] = []
    exception_hooks: list[Exception] = []
    canceled = Event()
    exited = Event()
    task.on_cancel.once().subscribe(lambda _: (events.append("cancel"), canceled.set()))
    task.on_exit.once().subscribe(lambda _: (events.append("exit"), exited.set()))

    task.start(TaskExecutionHooks.from_any(on_exception=exception_hooks.append))
    assert invocation_started.wait(timeout=1)

    task.cancel()

    assert canceled.wait(timeout=1)
    assert events == ["cancel"]
    assert task.status == "canceling"
    assert len(manager.get_running()) == 1
    assert not exited.is_set()

    allow_invocation_to_finish.set()

    assert exited.wait(timeout=1)
    assert events == ["cancel", "exit"]
    assert task.status == "canceled"
    assert cancellation_observed == [True]
    assert second_invocation_called == []
    assert exception_hooks == []
    exit_entity = task.execution_exit
    assert exit_entity is not None
    assert is_error(exit_entity.outcome)
    assert isinstance(exit_entity.outcome.exception, TaskCanceledException)
    assert manager.get(task.id) is None
