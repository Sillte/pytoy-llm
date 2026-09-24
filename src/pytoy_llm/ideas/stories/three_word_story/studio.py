import shutil
from pathlib import Path
from typing import Self

from pytoy_llm.idea import IdeaSpace
from pytoy_llm.models import LLMRequest, UsageLimit
from pytoy_llm.task.models import AgentInvocationSpec, InvocationHooks
from pytoy_llm.task.models.task_specs import TaskSpec
from pytoy_llm.tools.idea_tool.idea_tool import IdeaTool

from .prompts import CONVENTION, SYSTEM_PROMPT


class ThreeWordStoryStudio:
    def __init__(self, idea_space: IdeaSpace):
        self._idea_space = idea_space
        self._idea_space.ensure_root_marker()

    @classmethod
    def from_any(cls, idea_space_folder: Path | str) -> Self:
        idea_space = IdeaSpace.from_path(idea_space_folder)
        return cls(idea_space=idea_space)

    @property
    def idea_space(self) -> IdeaSpace:
        return self._idea_space

    @property
    def folder_path(self) -> Path:
        return self.idea_space.folder_path

    def _prepare_convention(self) -> None:
        convention_path = self.folder_path / ".convention.md"
        if not self.folder_path.exists():
            self.folder_path.mkdir(exist_ok=True, parents=True)
            convention_path.write_text(CONVENTION)

    def clear(self) -> None:
        for child in self.folder_path.iterdir():
            if child.name == ".convention.md":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    def make_invocation_spec(
        self, usage_limit: UsageLimit | None = None
    ) -> AgentInvocationSpec[str]:
        self._prepare_convention()
        usage_limit = usage_limit or UsageLimit(max_total_tokens=2000000, max_requests=50)
        idea_tool = IdeaTool.from_any(
            idea_space_root=self.folder_path, workspace_root=self.folder_path
        )

        return AgentInvocationSpec.from_any(
            lambda input_: LLMRequest.from_prompt(
                system=SYSTEM_PROMPT.strip(),
                user=input_,
            ),
            output_type=str,
            tools=[idea_tool.tools],
            usage_limit=usage_limit,
            hooks=InvocationHooks.from_any(
                on_start=idea_tool.mark_llm_start,
                on_completion=idea_tool.mark_llm_finished,
            ),
        )

    def make_task_spec(self, usage_limit: UsageLimit | None = None) -> TaskSpec[str]:
        return TaskSpec.from_specs(
            invocation_specs=[
                self.make_invocation_spec(usage_limit=usage_limit),
            ],
        )
