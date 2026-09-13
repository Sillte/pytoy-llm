from typing import Any

import pytest

from pytoy_llm.models import LLMMessage
from pytoy_llm.task.models import AgentInvocationSpec, LLMInvocationSpec
from pytoy_llm.task.models.context import ExecutionContext


def test_llm_invocation_spec_from_any_wraps_single_argument_creator() -> None:
    def create_messages(value: Any) -> LLMMessage:
        return LLMMessage.from_prompt(user=str(value))

    spec = LLMInvocationSpec.from_any(create_messages, output_type=str)

    assert spec.meta.name == "create_messages"
    assert isinstance(spec.create_messages("input", None), LLMMessage)


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
        LLMInvocationSpec.from_any(create_messages, output_type=str)
