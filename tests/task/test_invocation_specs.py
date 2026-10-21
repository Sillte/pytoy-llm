import uuid
from typing import Any, cast
from unittest.mock import Mock

import pytest

from pytoy_llm.models import LLMCancelException, LLMMessage, LLMRequest
from pytoy_llm.task.models import (
    AgentInvocationSpec,
    CancelToken,
    FailedSnapshot,
    FunctionInvocationSpec,
    InvocationException,
    InvocationHookException,
    InvocationHooks,
    InvocationResult,
    LLMInvocationSpec,
    SnapshotInvocationResult,
    TaskCanceledException,
    TaskContextState,
    TaskRequest,
    TaskSpec,
)
from pytoy_llm.task.models.context import ExecutionContext, RuntimeContextPatch
from pytoy_llm.task.models.invocation_specs import to_invocation_result
from pytoy_llm.task.shared.outcome import Success
from pytoy_llm.task.sync_executor import TaskSyncExecutor


def test_llm_invocation_spec_from_any_wraps_single_argument_creator() -> None:
    def create_request(value: Any) -> LLMRequest:
        return LLMRequest.from_prompt(user=str(value))

    spec = LLMInvocationSpec.from_any(create_request, output_type=str)

    assert spec.meta.name == "create_request"
    context = ExecutionContext(llm_param=None, connection=None, llm_messages=[])
    assert isinstance(spec.create_request("input", context), LLMRequest)


def test_agent_invocation_spec_from_any_accepts_context_creator() -> None:
    def create_request(value: Any, context: ExecutionContext) -> LLMRequest:
        return LLMRequest.from_prompt(user=f"{value}:{context.state}")

    spec = AgentInvocationSpec.from_any(create_request, output_type=str)

    assert spec.meta.name == "create_request"
    context = ExecutionContext(llm_param=None, connection=None, llm_messages=[])
    assert isinstance(spec.create_request("input", context), LLMRequest)


@pytest.mark.parametrize(
    ("spec_type", "facade_method"),
    [
        (LLMInvocationSpec, "completion_with_result"),
        (AgentInvocationSpec, "run_with_result"),
    ],
)
def test_llm_invocation_specs_forward_cancel_token(spec_type, facade_method, monkeypatch) -> None:
    cancel_token = CancelToken()
    facade = Mock()
    getattr(facade, facade_method).side_effect = RuntimeError("stop after forwarding")
    facade_factory = Mock(return_value=facade)
    monkeypatch.setattr("pytoy_llm.task.models.invocation_specs.LLMFacade", facade_factory)
    spec = spec_type.from_any(lambda value: str(value), output_type=str)
    context = ExecutionContext(
        llm_param=None,
        connection=None,
        llm_messages=[],
        cancel_token=cancel_token,
    )

    with pytest.raises(RuntimeError, match="stop after forwarding"):
        spec.invoke("input", context)

    assert getattr(facade, facade_method).call_args.kwargs["cancel_token"] is cancel_token


def test_invocation_spec_from_any_requires_output_type() -> None:
    def create_request(value: Any) -> LLMRequest:
        return LLMRequest.from_prompt(user=str(value))

    with pytest.raises(TypeError, match="output_type must be provided"):
        LLMInvocationSpec.from_any(create_request)


def test_invocation_spec_from_any_rejects_unsupported_creator_arity() -> None:
    def create_request(value: Any, context: ExecutionContext, extra: Any) -> LLMRequest:
        return LLMRequest.from_prompt(user=f"{value}:{context.state}:{extra}")

    with pytest.raises(ValueError, match="input and execution context"):
        LLMInvocationSpec.from_any(cast(Any, create_request), output_type=str)


def test_task_spec_cancellation_retains_prior_invocation_traces() -> None:
    token = CancelToken()
    invoked: list[str] = []

    def completed(value: Any, context: ExecutionContext) -> str:
        assert value == "input"
        assert context.cancel_token is token
        invoked.append("completed")
        return "completed"

    def cancel(value: Any, context: ExecutionContext) -> str:
        assert value == "completed"
        assert context.cancel_token is token
        invoked.append("cancel")
        token.set()
        return "not retained"

    def next_invocation(value: Any, context: ExecutionContext) -> str:
        assert context.cancel_token is token
        invoked.append("next")
        return str(value)

    task = TaskSpec.from_specs(
        [
            FunctionInvocationSpec.from_any(completed),
            FunctionInvocationSpec.from_any(cancel),
            FunctionInvocationSpec.from_any(next_invocation),
        ],
        output_type=str,
    )

    with pytest.raises(TaskCanceledException) as exc_info:
        task.run("input", TaskContextState(), cancel_token=token)

    assert invoked == ["completed", "cancel"]
    assert len(exc_info.value.completed_invocation_traces) == 1
    assert exc_info.value.completed_invocation_traces[0].result.output == "completed"
    exception_trace = exc_info.value.exception_trace
    assert "raise_if_cancelled" in exception_trace
    with pytest.raises(AttributeError):
        setattr(exc_info.value, "exception_trace", "changed")
    assert exc_info.value.exception_trace == exception_trace


def test_task_spec_maps_llm_cancel_exception_to_task_cancellation() -> None:
    cancel_token = CancelToken()
    messages = (LLMMessage.chat("request"),)
    llm_cancel_exception = LLMCancelException(messages=messages)

    def cancel(_value: Any, _context: ExecutionContext) -> str:
        raise llm_cancel_exception

    task = TaskSpec.from_single_spec(FunctionInvocationSpec.from_any(cancel), output_type=str)

    with pytest.raises(TaskCanceledException) as exc_info:
        task.run("input", TaskContextState(), cancel_token=cancel_token)

    assert exc_info.value.__cause__ is llm_cancel_exception
    assert llm_cancel_exception.messages == messages
    assert exc_info.value.completed_invocation_traces == ()


def test_task_spec_does_not_start_invocations_when_already_canceled() -> None:
    token = CancelToken()
    token.set()
    invoked: list[str] = []

    def invocation(value: Any, context: ExecutionContext) -> str:
        assert value == "input"
        assert context.cancel_token is token
        invoked.append("invocation")
        return str(value)

    task = TaskSpec.from_single_spec(FunctionInvocationSpec.from_any(invocation), output_type=str)

    with pytest.raises(TaskCanceledException) as exc_info:
        task.run("input", TaskContextState(), cancel_token=token)

    assert invoked == []
    assert exc_info.value.completed_invocation_traces == ()


def test_on_start_hook_failure_raises_invocation_hook_exception() -> None:
    hook_error = RuntimeError("start hook failed")

    def fail_start(_context: ExecutionContext) -> None:
        raise hook_error

    spec = FunctionInvocationSpec(
        invocator=lambda value, _context: value,
        hooks=InvocationHooks(on_start=fail_start),
    )

    with pytest.raises(InvocationHookException) as exc_info:
        spec.invoke("input", ExecutionContext(llm_param=None, connection=None, llm_messages=[]))

    assert exc_info.value.hook_name == "on_start"
    assert exc_info.value.hook_exception is hook_error
    assert exc_info.value.result is None
    assert exc_info.value.exception is None
    assert "start hook failed" in exc_info.value.stack_trace
    assert exc_info.value.__cause__ is hook_error


def test_invocation_results_hold_their_execution_context() -> None:
    first_context = ExecutionContext(llm_param=None, connection=None, llm_messages=[])
    second_context = ExecutionContext(llm_param=None, connection=None, llm_messages=[])
    first = InvocationResult[str](output="first", context=first_context)
    second = InvocationResult[str](output="second", context=second_context)

    first.context.state["changed"] = True
    first.context.llm_messages.append(LLMMessage.from_prompt(user="first"))

    assert first.context is first_context
    assert second.context.state == {}
    assert second.context.llm_messages == []


def test_empty_runtime_patch_leaves_execution_context_unchanged() -> None:
    context = ExecutionContext(
        llm_param=None,
        connection=None,
        llm_messages=[LLMMessage.from_prompt(user="existing")],
        state={"existing": "value"},
    )
    messages = context.llm_messages

    assert RuntimeContextPatch().apply(context) is context
    assert context.llm_messages is messages
    assert context.llm_messages == [LLMMessage.from_prompt(user="existing")]


def test_runtime_patch_replaces_messages_in_place_and_can_clear_history() -> None:
    message = LLMMessage.from_prompt(user="existing")
    context = ExecutionContext(llm_param=None, connection=None, llm_messages=[message])
    messages = context.llm_messages

    assert RuntimeContextPatch(llm_messages=[]).apply(context) is context
    assert context.llm_messages is messages
    assert context.llm_messages == []


def test_on_result_hook_edits_shared_context_state_and_messages() -> None:
    message = LLMMessage.from_prompt(user="hook update")
    context = ExecutionContext(
        llm_param=None,
        connection=None,
        llm_messages=[LLMMessage.from_prompt(user="existing")],
        state={"keep": "value", "remove": "value"},
    )

    def update_result(result: InvocationResult[str]) -> None:
        assert result.context is context
        result.context.state["added"] = result.output
        del result.context.state["remove"]
        result.context.llm_messages.append(message)

    spec = FunctionInvocationSpec(
        invocator=lambda _value, _context: "output",
        hooks=InvocationHooks(on_result=update_result),
    )

    trace = spec.invoke("input", context)

    assert trace.result.context is context
    assert context.state == {"keep": "value", "added": "output"}
    assert context.llm_messages[-1] is message


def test_runtime_messages_are_shared_with_result_hook() -> None:
    context = ExecutionContext(llm_param=None, connection=None, llm_messages=[])
    generated_message = LLMMessage.from_prompt(user="generated")
    hook_message = LLMMessage.from_prompt(user="hook")
    invocation_result = to_invocation_result(
        "output",
        context,
        runtime_patch=RuntimeContextPatch(llm_messages=[generated_message]),
    )

    def update_result(result: InvocationResult[str]) -> None:
        assert result.context is context
        assert result.context.llm_messages == [generated_message]
        result.context.llm_messages.append(hook_message)

    spec = FunctionInvocationSpec(
        invocator=lambda _value, _context: invocation_result,
        hooks=InvocationHooks(on_result=update_result),
    )

    trace = spec.invoke("input", context)

    assert trace.result.context is context
    assert context.llm_messages == [generated_message, hook_message]


def test_trace_snapshot_captures_post_hook_values_and_is_detached() -> None:
    output: list[str] = ["initial"]
    message = LLMMessage.from_prompt(user="hook")
    context = ExecutionContext(
        llm_param=None,
        connection=None,
        llm_messages=[],
        state={"nested": {"items": []}},
    )

    def update_result(result: InvocationResult[list[str]]) -> None:
        result.output.append("hook")
        result.context.state["nested"]["items"].append("hook")
        result.context.llm_messages.append(message)

    spec = FunctionInvocationSpec(
        invocator=lambda _value, _context: output,
        hooks=InvocationHooks(on_result=update_result),
    )

    trace = spec.invoke("input", context)
    snapshot = trace.snapshot_result

    assert snapshot is not None
    assert isinstance(snapshot, SnapshotInvocationResult)
    assert snapshot.output == ["initial", "hook"]
    assert snapshot.state == {"nested": {"items": ["hook"]}}
    assert snapshot.llm_messages == [message]

    output.append("later")
    context.state["nested"]["items"].append("later")
    context.llm_messages.append(LLMMessage.from_prompt(user="later"))

    assert snapshot.output == ["initial", "hook"]
    assert snapshot.state == {"nested": {"items": ["hook"]}}
    assert snapshot.llm_messages == [message]


def test_snapshot_deepcopy_failure_is_recorded_without_failing_invocation() -> None:
    class Uncopyable:
        def __deepcopy__(self, memo):
            raise RuntimeError("snapshot copy failed")

    exception_hook_calls: list[Exception] = []
    spec = FunctionInvocationSpec(
        invocator=lambda _value, _context: Uncopyable(),
        hooks=InvocationHooks(on_exception=exception_hook_calls.append),
    )

    trace = spec.invoke("input", ExecutionContext(llm_param=None, connection=None, llm_messages=[]))

    assert isinstance(trace.result.output, Uncopyable)
    assert isinstance(trace.snapshot_result, FailedSnapshot)
    assert trace.snapshot_result.error_type == "RuntimeError"
    assert trace.snapshot_result.message == "snapshot copy failed"
    assert exception_hook_calls == []


def test_on_result_hook_failure_keeps_mutations_without_rollback() -> None:
    hook_error = RuntimeError("result hook failed")
    message = LLMMessage.from_prompt(user="mutated before failure")
    context = ExecutionContext(
        llm_param=None,
        connection=None,
        llm_messages=[],
        state={"remove": True},
    )

    def fail_result(result: InvocationResult[Any]) -> None:
        result.context.state["added"] = True
        del result.context.state["remove"]
        result.context.llm_messages.append(message)
        raise hook_error

    spec = FunctionInvocationSpec(
        invocator=lambda value, _context: value,
        hooks=InvocationHooks(on_result=fail_result),
    )

    with pytest.raises(InvocationHookException) as exc_info:
        spec.invoke("input", context)

    assert exc_info.value.hook_name == "on_result"
    assert exc_info.value.hook_exception is hook_error
    assert exc_info.value.result is not None
    assert exc_info.value.result.output == "input"
    assert exc_info.value.exception is None
    assert context.state == {"added": True}
    assert context.llm_messages == [message]


def test_on_exception_hook_failure_retains_invocation_exception() -> None:
    hook_error = RuntimeError("exception hook failed")
    invocation_error = ValueError("invocation failed")

    def fail_exception(_exception: Exception) -> None:
        raise hook_error

    def fail_invocation(_value: Any, _context: ExecutionContext) -> str:
        raise invocation_error

    spec = FunctionInvocationSpec(
        invocator=fail_invocation,
        hooks=InvocationHooks(on_exception=fail_exception),
    )

    with pytest.raises(InvocationHookException) as exc_info:
        spec.invoke("input", ExecutionContext(llm_param=None, connection=None, llm_messages=[]))

    assert exc_info.value.hook_name == "on_exception"
    assert exc_info.value.hook_exception is hook_error
    assert exc_info.value.result is None
    assert exc_info.value.exception is invocation_error
    assert exc_info.value.__cause__ is hook_error


def test_sync_executor_passes_request_id_to_task_result() -> None:
    request = TaskRequest(
        spec=TaskSpec.from_single_spec(
            FunctionInvocationSpec.from_any(lambda value: value),
            output_type=str,
        ),
        input="input",
    )

    task_exit = TaskSyncExecutor().execute(request)

    assert task_exit.request_id == request.id
    assert isinstance(task_exit.outcome, Success)
    result = task_exit.outcome.value
    assert result.id == request.id
    assert result.traces[0].result.context.task_id == request.id


@pytest.mark.parametrize("task_id", [None, "request-id"])
def test_task_result_id_matches_execution_context_id(task_id: str | None) -> None:
    observed_task_ids: list[str] = []
    spec = FunctionInvocationSpec.from_any(
        lambda value, context: observed_task_ids.append(context.task_id) or value
    )
    task = TaskSpec.from_single_spec(spec, output_type=str)

    outcome = task.run("input", TaskContextState(), task_id=task_id)

    assert isinstance(outcome, Success)
    result = outcome.value
    assert result.id == result.traces[0].result.context.task_id
    assert observed_task_ids == [result.id]
    if task_id is not None:
        assert result.id == task_id
    else:
        assert uuid.UUID(result.id).version == 4


def test_task_run_returns_invocation_hook_exception_directly() -> None:
    hook_error = RuntimeError("start hook failed")

    def fail_start(_context: ExecutionContext) -> None:
        raise hook_error

    spec = FunctionInvocationSpec(
        invocator=lambda value, _context: value,
        hooks=InvocationHooks(on_start=fail_start),
    )
    task = TaskSpec.from_single_spec(spec, output_type=str)

    outcome = task.run("input", TaskContextState())

    assert outcome.kind == "Error"
    assert isinstance(outcome.exception, InvocationException)
    assert isinstance(outcome.exception.invocation_exception, InvocationHookException)
    assert outcome.exception.invocation_exception.hook_exception is hook_error
