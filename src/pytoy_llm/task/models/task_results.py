import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field

from pytoy_llm.models import LLMTokens
from pytoy_llm.task.models.context import TaskContextState
from pytoy_llm.task.models.expenditures import LLMExpenditure
from pytoy_llm.task.models.invocation_results import InvocationTrace


@dataclass(frozen=True)
class TaskResult[T]:
    output: T
    context_state: TaskContextState
    traces: Sequence[InvocationTrace] = field(default_factory=lambda: tuple())
    task_name: str = "NoTaskName"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def llm_tokens(self) -> LLMTokens:
        """Returns the total number of LLM tokens used in this task result."""
        tokens_list = [
            trace.expenditure.tokens
            for trace in self.traces
            if isinstance(trace.expenditure, LLMExpenditure)
        ]
        return LLMTokens.aggregate(tokens_list)
