from typing import Any, cast

import pytest

from pytoy_llm.models import LLMMessage
from pytoy_llm.task.models import (
    AgentInvocationSpec,
    FunctionInvocationSpec,
    InvocationHooks,
    LLMInvocationSpec,
)
from pytoy_llm.task.models.context import ExecutionContext


def test_llm_invocation_spec_from_any_wraps_single_argument_creator() -> None:
    def create_messages(value: Any) -> LLMMessage:
        return LLMMessage.from_prompt(user=str(value))

    spec = LLMInvocationSpec.from_any(create_messages, output_type=str)

    assert spec.meta.name == "create_messages"
    context = ExecutionContext(llm_param=None, connection=None, llm_messages=())
    assert isinstance(spec.create_messages("input", context), LLMMessage)


def test_agent_invocation_spec_from_any_accepts_context_creator() -> None:
    def create_messages(value: Any, context: ExecutionContext) -> LLMMessage:
        return LLMMessage.from_prompt(user=f"{value}:{context.state}")

    spec = AgentInvocationSpec.from_any(create_messages, output_type=str)

    assert spec.meta.name == "create_messages"
    context = ExecutionContext(llm_param=None, connection=None, llm_messages=())
    assert isinstance(spec.create_messages("input", context), LLMMessage)


def test_invocation_spec_from_any_requires_output_type() -> None:
    def create_messages(value: Any) -> LLMMessage:
        return LLMMessage.from_prompt(user=str(value))

    with pytest.raises(TypeError, match="output_type must be provided"):
        LLMInvocationSpec.from_any(create_messages)


def test_invocation_spec_from_any_rejects_unsupported_creator_arity() -> None:
    def create_messages(value: Any, context: ExecutionContext, extra: Any) -> LLMMessage:
        return LLMMessage.from_prompt(user=f"{value}:{context.state}:{extra}")

    with pytest.raises(ValueError, match="input and execution context"):
        LLMInvocationSpec.from_any(cast(Any, create_messages), output_type=str)


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

    result = spec.invoke(
        "input", ExecutionContext(llm_param=None, connection=None, llm_messages=())
    )

    assert result.output == "input"


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
