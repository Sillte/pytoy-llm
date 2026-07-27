import json
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from pytoy_llm.materials.composers.models import (
    SectionDataComposer,
    SectionUsage,
    SystemPromptTemplate,
)
from pytoy_llm.materials.core import SectionData, warn_forbidden_headers
from pytoy_llm.models.llm_messages import LLMMessage
from pytoy_llm.task.models import LLMInvocationSpec
from pytoy_llm.task.models.context import ExecutionContext
from pytoy_llm.task.models.metas import InvocationSpecMeta


class InvocationPromptComposer:
    """
    Compose a full LLM prompt from:
    - LLMInvocationSpec (intent, rules)
    - SectionUsage
    - SectionData
    """

    def __init__(
        self,
        prompt_template: SystemPromptTemplate,
        section_usages: Sequence[SectionUsage] | None = None,
        section_data_list: Sequence[SectionData] | None = None,
    ):
        self.prompt_template = prompt_template
        self.section_usages = section_usages or []
        self.section_data_list = section_data_list or []

    def compose_prompt(self) -> str:
        # Invocation header
        invocation_header = f"# Task: {self.prompt_template.name}\n\n"
        warn_forbidden_headers(self.prompt_template.name, min_allowed_header_level=5)
        # Intent
        invocation_intent = f"## Task Intent\n\n{self.prompt_template.intent}\n\n"
        warn_forbidden_headers(self.prompt_template.intent, min_allowed_header_level=2)

        # Rules
        invocation_rules = ""
        if self.prompt_template.rules:
            invocation_rules = "## Rules\n\n" + "\n".join(f"* {rule}" for rule in self.prompt_template.rules) + "\n\n"

        # Role
        role_str = f"## Role\n{self.prompt_template.role}\n\n" if self.prompt_template.role else ""
        warn_forbidden_headers(role_str, min_allowed_header_level=2)

        # SectionUsage + SectionData
        sections_str = SectionDataComposer.compose_sections_with_usage(self.section_usages, self.section_data_list)

        # Output specification
        output_description = f"## Expected Output\n\n{self.prompt_template.output_description}\n\n"
        warn_forbidden_headers(self.prompt_template.output_description, min_allowed_header_level=2)

        if self.prompt_template.output_type:
            output_type_instruction = self.to_output_type_instruction(self.prompt_template.output_type)
            output_spec_str = f"## Output Specification\n\n{output_type_instruction}\n\n"
        else:
            output_spec_str = ""

        if self.prompt_template.reasoning_guidance:
            warn_forbidden_headers(self.prompt_template.reasoning_guidance, min_allowed_header_level=2)
        # Reasoning guidance
        reasoning = (
            f"## Reasoning Guidance\n\n{self.prompt_template.reasoning_guidance}\n\n"
            if self.prompt_template.reasoning_guidance
            else ""
        )

        # Compose final prompt
        prompt = "\n".join(
            [
                invocation_header,
                invocation_intent,
                invocation_rules,
                role_str,
                sections_str,
                output_description,
                output_spec_str,
                reasoning,
            ]
        )
        return prompt

    def to_output_type_instruction(self, output_type: type[BaseModel] | type[str]) -> str:
        """Return instruction to include in prompt based on BaseModel schema"""
        if isinstance(output_type, type) and issubclass(output_type, BaseModel):
            schema_json = json.dumps(output_type.model_json_schema())
            return f"Return output as JSON matching the following schema:\n```json\n{schema_json}```\n\n"
        elif output_type is str:
            return "Return output as plain text string."
        elif isinstance(output_type, BaseModel):
            return self.to_output_type_instruction(output_type.__class__)  # Fallback, it is not good.
        else:
            raise ValueError(f"Invalid output type `{output_type=}`")

    def compose_message(self, user_prompt: str | None = None) -> LLMMessage:
        system_prompt = self.compose_prompt()
        return LLMMessage.from_prompt(user=user_prompt, system=system_prompt)

    def compose_invocation_spec(self) -> LLMInvocationSpec:
        def create_messages(input: Any, context: ExecutionContext) -> Sequence[LLMMessage]:
            input = str(input) if input else None
            messages = self.compose_message(input)
            return [messages]

        return LLMInvocationSpec(
            create_messages=create_messages,
            output_type=self.prompt_template.output_type,
            meta=InvocationSpecMeta(name=self.prompt_template.name, intent=self.prompt_template.intent),
        )
