from typing import Any, cast

import pytest

from pytoy_llm.models import LLMRequest
from pytoy_llm.task.models import (
    AgentInvocationSpec,
    CancelToken,
    FunctionInvocationSpec,
    InvocationHooks,
    InvocationResult,
    LLMInvocationSpec,
    TaskCanceledException,
    TaskContextState,
    TaskSpec,
)
from pytoy_llm.task.models.context import ExecutionContext


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


def test_hook_exceptions_do_not_change_successful_invocation() -> None:
    def fail_hook(*_args: Any) -> None:
        raise RuntimeError("hook failed")

    spec = FunctionInvocationSpec(
        invocator=lambda value, _context: value,
        hooks=InvocationHooks(
            on_start=fail_hook,
            on_result=fail_hook,
        ),
    )

    trace = spec.invoke("input", ExecutionContext(llm_param=None, connection=None, llm_messages=()))

    assert trace.result.output == "input"


def test_invocation_trace_owns_invocation_result() -> None:
    invocation_result = InvocationResult[str](output="output")
    spec = FunctionInvocationSpec(
        invocator=lambda _value, _context: invocation_result,
    )

    trace = spec.invoke("input", ExecutionContext(llm_param=None, connection=None, llm_messages=()))

    assert trace.result == invocation_result


def test_hook_exception_does_not_replace_invocation_exception() -> None:
    def fail_hook(*_args: Any) -> None:
        raise RuntimeError("hook failed")

    def fail_invocation(_value: Any, _context: ExecutionContext) -> str:
        raise ValueError("invocation failed")

    spec = FunctionInvocationSpec(
        invocator=fail_invocation,
        hooks=InvocationHooks(on_exception=fail_hook),
    )

    with pytest.raises(ValueError, match="invocation failed"):
        spec.invoke("input", ExecutionContext(llm_param=None, connection=None, llm_messages=()))
