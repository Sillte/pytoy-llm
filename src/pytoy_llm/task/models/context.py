from pytoy_llm.task.models.context_protocols import LLMFacadeProtocol, LLMTaskContextProtocol  # NOQA
from pytoy_llm.task.models.repository import LLMTaskStateRepository
from pytoy_llm.task.models.schemas import LLMTaskArgument, LLMTaskSpecMeta
from pytoy_llm.llm_facade import LLMFacade


from dataclasses import dataclass, field


@dataclass(frozen=True)
class LLMTaskContext:
    task_argument: LLMTaskArgument
    task_meta: LLMTaskSpecMeta
    llm_facade: LLMFacade
    repository: LLMTaskStateRepository = field(default_factory=LLMTaskStateRepository)

    @property
    def initial_history(self):
        return self.task_argument.initial_history

    @property
    def initial_input(self):
        return self.task_argument.initial_input
