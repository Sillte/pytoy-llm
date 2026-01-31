import json
from typing import Annotated, Sequence, Any
from pytoy_llm.materials.core import  SectionData, warn_forbidden_headers
from pytoy_llm.materials.composers.models import SystemPromptTemplate, SectionUsage
from pytoy_llm.materials.composers.models import SectionDataComposer
from pytoy_llm.models import InputMessage
from pydantic import BaseModel

from pytoy_llm.task.models import LLMInvocationSpec, LLMTaskContext

class InvocationPromptComposer:
    """
    Compose a full LLM prompt from:
    - LLMInvocationSpec (intent, rules)
    - SectionUsage
    - SectionData
    """
    def __init__(self,
                 prompt_template: SystemPromptTemplate,
                 section_usages: Sequence[SectionUsage],
                 section_data_list: Sequence[SectionData]):
        self.prompt_template = prompt_template
        self.section_usages = section_usages
        self.section_data_list = section_data_list

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
        sections_str = SectionDataComposer.compose_sections_with_usage(
            self.section_usages,
            self.section_data_list
        )

        # Output specification
        output_description = f"## Expected Output\n\n{self.prompt_template.output_description}\n\n"
        warn_forbidden_headers(self.prompt_template.output_description, min_allowed_header_level=2)

        if self.prompt_template.output_spec:
            output_spec_instruction = self.to_output_spec_instruction(self.prompt_template.output_spec)
            output_spec_str = f"## Output Specification\n\n{output_spec_instruction}\n\n"
        else:
            output_spec_str = ""

        if self.prompt_template.reasoning_guidance:
            warn_forbidden_headers(self.prompt_template.reasoning_guidance, min_allowed_header_level=2)
        # Reasoning guidance
        reasoning = f"## Reasoning Guidance\n\n{self.prompt_template.reasoning_guidance}\n\n" if self.prompt_template.reasoning_guidance else ""

        # Compose final prompt
        prompt = "\n".join([
            invocation_header,
            invocation_intent,
            invocation_rules,
            role_str,
            sections_str,
            output_description,
            output_spec_str,
            reasoning
        ])
        return prompt

    def to_output_spec_instruction[T: BaseModel](self, output_spec: type[str] | type[T]) -> str:
        """Return instruction to include in prompt based on BaseModel schema"""
        if isinstance(output_spec, type) and issubclass(output_spec, BaseModel):
            schema_json = json.dumps(output_spec.model_json_schema())
            return f"Return output as JSON matching the following schema:\n```json\n{schema_json}```\n\n"
        elif output_spec is str:
            return "Return output as plain text string."
        elif isinstance(output_spec, BaseModel):
            return self.to_output_spec_instruction(output_spec.__class__)  # Fallback, it is not good.
        else:
            raise ValueError(f"Invalid output_spec type `{output_spec=}`")
    
    def compose_messages(self, user_prompt: str | None = None) -> Sequence[InputMessage]:
        system_prompt = self.compose_prompt()
        messages = [InputMessage(role="system", content=system_prompt)]
        if user_prompt:
            messages.append(InputMessage(role="user", content=user_prompt))
        return messages
    

    def compose_invocation_spec(self) -> LLMInvocationSpec:
        def create_messages(input: Any, context: LLMTaskContext) -> Sequence[InputMessage]:
            input = str(input) if input else None 
            messages = self.compose_messages(input)
            return messages
        return LLMInvocationSpec(create_messages=create_messages, output_spec=self.prompt_template.output_spec)