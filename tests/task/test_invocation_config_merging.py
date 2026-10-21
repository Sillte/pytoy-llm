from __future__ import annotations

from typing import ClassVar

from pytoy_llm.models import (
    CancelToken,
    LLMEventEmitters,
    LLMParam,
    LLMRequestLike,
    LLMResult,
    LLMToolsLike,
    UsageLimit,
)
from pytoy_llm.models.llm_metas import LLMOutputMeta, LLMTokens
from pytoy_llm.task.models import (
    AgentInvocationSpec,
    FunctionInvocationSpec,
    LLMInvocationSpec,
    TaskContextState,
    TaskSpec,
)
from pytoy_llm.task.models.context import ExecutionContext


class FakeFacade:
    instances: ClassVar[list[FakeFacade]] = []

    def __init__(
        self,
        *,
        connection: object | None,
        llm_param: LLMParam | None,
        event_emitters: LLMEventEmitters,
    ) -> None:
        self.connection = connection
        self.llm_param = llm_param
        self.event_emitters = event_emitters
        self.usage_limit: UsageLimit | None = None
        self.cancel_token: CancelToken | None = None
        self.instances.append(self)

    def completion_with_result(
        self,
        request: LLMRequestLike,
        *,
        output_type: type[str],
        cancel_token: CancelToken | None = None,
    ) -> LLMResult[str]:
        self.cancel_token = cancel_token
        return _llm_result()

    def run_with_result(
        self,
        request: LLMRequestLike,
        *,
        output_type: type[str],
        tools: LLMToolsLike,
        usage_limit: UsageLimit | None,
        cancel_token: CancelToken | None = None,
    ) -> LLMResult[str]:
        self.usage_limit = usage_limit
        self.cancel_token = cancel_token
        return _llm_result()


def _install_fake_facade(monkeypatch) -> None:
    FakeFacade.instances.clear()
    monkeypatch.setattr("pytoy_llm.task.models.invocation_specs.LLMFacade", FakeFacade)


def _llm_result() -> LLMResult[str]:
    return LLMResult(
        meta=LLMOutputMeta(tokens=LLMTokens(prompt=1, completion=1, total=2)),
        output="output",
        messages=(),
    )


def test_llm_invocation_merges_context_and_invocation_parameters(monkeypatch) -> None:
    _install_fake_facade(monkeypatch)
    spec = LLMInvocationSpec.from_any(
        lambda _input: "request",
        output_type=str,
        llm_param=LLMParam(temperature=0.7, top_p=0.8),
    )
    cancel_token = CancelToken()
    context = ExecutionContext(
        llm_param=LLMParam(temperature=0.2, max_tokens=500, top_p=0.5),
        connection=None,
        llm_messages=[],
        cancel_token=cancel_token,
    )

    spec.invoke("input", context)

    assert FakeFacade.instances[0].llm_param == LLMParam(temperature=0.7, max_tokens=500, top_p=0.8)
    assert FakeFacade.instances[0].cancel_token is cancel_token


def test_llm_invocation_keeps_none_when_both_parameter_sources_are_none(monkeypatch) -> None:
    _install_fake_facade(monkeypatch)
    spec = LLMInvocationSpec.from_any(lambda _input: "request", output_type=str)
    cancel_token = CancelToken()
    context = ExecutionContext(
        llm_param=None,
        connection=None,
        llm_messages=[],
        cancel_token=cancel_token,
    )

    spec.invoke("input", context)

    assert FakeFacade.instances[0].llm_param is None
    assert FakeFacade.instances[0].cancel_token is cancel_token


def test_agent_invocation_merges_usage_limits_and_llm_parameters(monkeypatch) -> None:
    _install_fake_facade(monkeypatch)
    spec = AgentInvocationSpec.from_any(
        lambda _input: "request",
        output_type=str,
        llm_param=LLMParam(temperature=0.7),
        usage_limit=UsageLimit(max_total_tokens=200),
    )
    cancel_token = CancelToken()
    context = ExecutionContext(
        llm_param=LLMParam(temperature=0.2, max_tokens=500),
        connection=None,
        llm_messages=[],
        usage_limit=UsageLimit(max_total_tokens=500, max_requests=10),
        cancel_token=cancel_token,
    )

    spec.invoke("input", context)

    assert FakeFacade.instances[0].llm_param == LLMParam(temperature=0.7, max_tokens=500)
    assert FakeFacade.instances[0].usage_limit == UsageLimit(max_total_tokens=200, max_requests=10)
    assert FakeFacade.instances[0].cancel_token is cancel_token


def test_task_spec_run_places_base_configuration_in_execution_context() -> None:
    contexts = []
    spec = FunctionInvocationSpec(
        invocator=lambda value, context: contexts.append(context) or value,
    )
    task = TaskSpec.from_single_spec(spec, output_type=str)
    llm_param = LLMParam(temperature=0.2)
    usage_limit = UsageLimit(max_requests=10)

    task.run(
        "input",
        TaskContextState(),
        llm_param=llm_param,
        usage_limit=usage_limit,
    )

    assert contexts[0].llm_param == llm_param
    assert contexts[0].usage_limit == usage_limit


def test_task_spec_run_captures_invocation_stack_trace() -> None:
    def fail_invocation(_input, _context):
        raise ValueError("invocation failed")

    task = TaskSpec.from_single_spec(FunctionInvocationSpec(invocator=fail_invocation))

    outcome = task.run("input", TaskContextState())

    assert outcome.kind == "Error"
    assert "fail_invocation" in outcome.exception.stack_trace
    assert "ValueError: invocation failed" in outcome.exception.stack_trace
    assert outcome.exception.stack_trace in str(outcome.exception)
