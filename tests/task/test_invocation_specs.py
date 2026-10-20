from typing import Any, cast
from unittest.mock import Mock

import pytest

from pytoy_llm.models import LLMCancelException, LLMMessage, LLMRequest
from pytoy_llm.task.models import (
    AgentInvocationSpec,
    CancelToken,
    ContextPatch,
    FunctionInvocationSpec,
    InvocationException,
    InvocationHookException,
    InvocationHooks,
    InvocationResult,
    LLMInvocationSpec,
    TaskCanceledException,
    TaskContextState,
    TaskSpec,
)
from pytoy_llm.task.models.context import ExecutionContext, RuntimeContextPatch


def test_llm_invocation_spec_from_any_wraps_single_argument_creator() -> None:
    def create_request(value: Any) -> LLMRequest:
        return LLMRequest.from_prompt(user=str(value))

    spec = LLMInvocationSpec.from_any(create_request, output_type=str)

    assert spec.meta.name == "create_request"
    context = ExecutionContext(llm_param=None, connection=None, llm_messages=())
    assert isinstance(spec.create_request("input", context), LLMRequest)


def test_agent_invocation_spec_from_any_accepts_context_creator() -> None:
    def create_request(value: Any, context: ExecutionContext) -> LLMRequest:
        return LLMRequest.from_prompt(user=f"{value}:{context.state}")

    spec = AgentInvocationSpec.from_any(create_request, output_type=str)

    assert spec.meta.name == "create_request"
    context = ExecutionContext(llm_param=None, connection=None, llm_messages=())
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
        llm_messages=(),
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
        spec.invoke("input", ExecutionContext(llm_param=None, connection=None, llm_messages=()))

    assert exc_info.value.hook_name == "on_start"
    assert exc_info.value.hook_exception is hook_error
    assert exc_info.value.result is None
    assert exc_info.value.exception is None
    assert "start hook failed" in exc_info.value.stack_trace
    assert exc_info.value.__cause__ is hook_error


def test_invocation_result_defaults_to_independent_empty_patches() -> None:
    first = InvocationResult[str](output="first")
    second = InvocationResult[str](output="second")

    first.context_patch.state_updates["changed"] = True
    first.context_patch.state_removals.add("obsolete")
    first.context_patch.clear_state = True

    assert first.runtime_patch.llm_messages is None
    assert first.context_patch.llm_messages is None
    assert second.context_patch.state_updates == {}
    assert second.context_patch.state_removals == set()
    assert second.context_patch.clear_state is False


def test_empty_patches_leave_execution_context_unchanged() -> None:
    message = LLMMessage.from_prompt(user="existing")
    context = ExecutionContext(
        llm_param=None,
        connection=None,
        llm_messages=(message,),
        state={"existing": "value"},
    )

    assert RuntimeContextPatch().apply(context) == context
    assert ContextPatch().apply(context) == context


def test_empty_message_patches_clear_message_history() -> None:
    message = LLMMessage.from_prompt(user="existing")
    context = ExecutionContext(
        llm_param=None,
        connection=None,
        llm_messages=(message,),
    )

    assert RuntimeContextPatch(llm_messages=()).apply(context).llm_messages == ()
    assert ContextPatch(llm_messages=()).apply(context).llm_messages == ()


def test_context_patch_removes_then_updates_state_without_discarding_entries() -> None:
    context = ExecutionContext(
        llm_param=None,
        connection=None,
        llm_messages=(),
        state={"existing": "preserved", "updated": "old", "removed": "value"},
    )

    updated_context = ContextPatch(
        state_updates={"updated": "new", "added": "value"},
        state_removals={"removed", "updated", "missing"},
    ).apply(context)

    assert updated_context.state == {
        "existing": "preserved",
        "updated": "new",
        "added": "value",
    }
    assert context.state == {
        "existing": "preserved",
        "updated": "old",
        "removed": "value",
    }


def test_context_patch_can_clear_state_then_apply_updates() -> None:
    context = ExecutionContext(
        llm_param=None,
        connection=None,
        llm_messages=(),
        state={"existing": "value"},
    )

    updated_context = ContextPatch(
        state_updates={"new": "value"},
        clear_state=True,
    ).apply(context)

    assert updated_context.state == {"new": "value"}


def test_invocation_trace_owns_invocation_result() -> None:
    invocation_result = InvocationResult[str](output="output")
    spec = FunctionInvocationSpec(
        invocator=lambda _value, _context: invocation_result,
    )

    trace = spec.invoke("input", ExecutionContext(llm_param=None, connection=None, llm_messages=()))

    assert trace.result == invocation_result


def test_on_result_hook_can_mutate_context_patch_state() -> None:
    invocation_result = InvocationResult[str](
        output="output",
        context_patch=ContextPatch(state_updates={"status": "initial"}),
    )

    def update_result(result: InvocationResult[str]) -> None:
        result.context_patch.state_updates = {"status": "updated"}

    spec = FunctionInvocationSpec(
        invocator=lambda _value, _context: invocation_result,
        hooks=InvocationHooks(on_result=update_result),
    )

    trace = spec.invoke("input", ExecutionContext(llm_param=None, connection=None, llm_messages=()))

    assert trace.result.context_patch.state_updates == {"status": "updated"}


def test_on_result_hook_failure_retains_result() -> None:
    hook_error = RuntimeError("result hook failed")

    def fail_result(_result: InvocationResult[Any]) -> None:
        raise hook_error

    spec = FunctionInvocationSpec(
        invocator=lambda value, _context: value,
        hooks=InvocationHooks(on_result=fail_result),
    )

    with pytest.raises(InvocationHookException) as exc_info:
        spec.invoke("input", ExecutionContext(llm_param=None, connection=None, llm_messages=()))

    assert exc_info.value.hook_name == "on_result"
    assert exc_info.value.hook_exception is hook_error
    assert exc_info.value.result is not None
    assert exc_info.value.result.output == "input"
    assert exc_info.value.exception is None


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
        spec.invoke("input", ExecutionContext(llm_param=None, connection=None, llm_messages=()))

    assert exc_info.value.hook_name == "on_exception"
    assert exc_info.value.hook_exception is hook_error
    assert exc_info.value.result is None
    assert exc_info.value.exception is invocation_error
    assert exc_info.value.__cause__ is hook_error


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
