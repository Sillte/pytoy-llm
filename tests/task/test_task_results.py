from pytoy_llm.models import LLMTokens
from pytoy_llm.task.models.context import ExecutionContext, TaskContextState
from pytoy_llm.task.models.expenditures import LLMExpenditure
from pytoy_llm.task.models.invocation_results import (
    InvocationInfo,
    InvocationResult,
    InvocationTrace,
)
from pytoy_llm.task.models.task_results import TaskResult


def make_llm_trace(tokens: LLMTokens) -> InvocationTrace[None]:
    context = ExecutionContext.from_any(llm_param=None, connection=None, llm_messages=[])
    return InvocationTrace(
        input=None,
        result=InvocationResult(output=None, context=context),
        info=InvocationInfo(kind="llm", started_at=0, ended_at=1),
        expenditure=LLMExpenditure(tokens=tokens),
    )


def test_task_token_total_sums_llm_usage() -> None:
    result = TaskResult(
        input=None,
        output=None,
        context_state=TaskContextState(),
        traces=[
            make_llm_trace(LLMTokens(prompt=1, completion=1, total=2)),
            make_llm_trace(LLMTokens(prompt=3, completion=2, total=5)),
        ],
    )

    assert result.llm_tokens == LLMTokens(prompt=4, completion=3, total=7)


def test_task_token_total_is_zero_without_llm_invocations() -> None:
    result = TaskResult(input=None, output=None, context_state=TaskContextState())

    assert result.llm_tokens == LLMTokens(
        prompt=0, completion=0, total=0, cache_read=0, cache_write=0
    )
