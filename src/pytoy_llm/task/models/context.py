from pytoy_llm.task.models.context_protocols import LLMFacadeProtocol, LLMTaskContextProtocol  # NOQA
from pytoy_llm.task.models.repository import LLMTaskStateRepository
from pytoy_llm.task.models.schemas import LLMTaskArgument, LLMTaskSpecMeta


from pydantic import BaseModel

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LLMTaskContext[T: BaseModel | str]:
    task_argument: LLMTaskArgument
    task_meta: LLMTaskSpecMeta
    llm_facade: LLMFacadeProtocol[T]
    repository: LLMTaskStateRepository = field(default_factory=LLMTaskStateRepository)

    @property
    def initial_history(self):
        return self.task_argument.initial_history

    @property
    def initial_input(self):
        return self.task_argument.initial_input