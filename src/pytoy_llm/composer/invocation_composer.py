from functools import partial
from typing import Any

from pydantic import BaseModel

from pytoy_llm.composer.models import (
    SupplementarySectionsLike,
    SystemPromptSpec,
)
from pytoy_llm.composer.system_prompt_composer import SystemPromptComposer
from pytoy_llm.models import LLMRequest, LLMToolsLike
from pytoy_llm.task.models import AgentInvocationSpec, LLMInvocationSpec
from pytoy_llm.task.models.context import ExecutionContext
from pytoy_llm.task.models.metas import InvocationSpecMeta


class InvocationComposer[T: BaseModel | str]:
    def __init__(self, system_prompt_spec: SystemPromptSpec) -> None:
        self.system_prompt_spec = system_prompt_spec
        self.system_prompt_composer = SystemPromptComposer(self.system_prompt_spec)

    def make_request(
        self, user_prompt: str, supplementary_sections: SupplementarySectionsLike | None = None
    ) -> LLMRequest:
        system_prompt = self.system_prompt_composer.compose_prompt(
            supplementary_sections=supplementary_sections
        )
        return LLMRequest.from_prompt(user=user_prompt, system=system_prompt)

    def _create_request(
        self,
        input: Any,
        context: ExecutionContext,
        supplementary_sections: SupplementarySectionsLike | None,
    ) -> LLMRequest:
        input = str(input) if input else "No User Input"
        return self.make_request(
            user_prompt=str(input), supplementary_sections=supplementary_sections
        )

    def compose_llm_invocation_spec(
        self, supplementary_sections: SupplementarySectionsLike | None = None
    ) -> LLMInvocationSpec:
        return LLMInvocationSpec(
            create_request=partial(
                self._create_request, supplementary_sections=supplementary_sections
            ),
            output_type=self.system_prompt_spec.output_spec.output_type,
            meta=InvocationSpecMeta(
                name=self.system_prompt_spec.name, intent=self.system_prompt_spec.intent
            ),
        )

    def compose_agent_invocation_spec(
        self,
        tools: LLMToolsLike = tuple(),
        supplementary_sections: SupplementarySectionsLike | None = None,
    ) -> AgentInvocationSpec:
        return AgentInvocationSpec(
            create_request=partial(
                self._create_request, supplementary_sections=supplementary_sections
            ),
            output_type=self.system_prompt_spec.output_spec.output_type,
            meta=InvocationSpecMeta(
                name=self.system_prompt_spec.name, intent=self.system_prompt_spec.intent
            ),
            tools=tools,
        )
