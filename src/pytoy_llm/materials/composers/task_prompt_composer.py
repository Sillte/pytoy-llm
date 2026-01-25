from pytoy_llm.materials.composers.models import LLMTask, SectionUsage, SectionDataComposer
from typing import Annotated, Sequence
from pydantic import BaseModel, Field
from pytoy_llm.materials.core import TextSectionData, ModelSectionData, SectionData
from pytoy_llm.models import InputMessage

class TaskPromptComposer:
    """
    Compose a full LLM prompt from:
    - LLMTask (intent, rules)
    - SectionUsage
    - SectionData
    """
    def __init__(self,
                 task: LLMTask,
                 section_usages: Sequence[SectionUsage],
                 section_data_list: Sequence[SectionData]):
        self.task = task
        self.section_usages = section_usages
        self.section_data_list = section_data_list

    def compose_prompt(self) -> str:
        # Task header
        task_header = f"# Task: {self.task.name}\n\n"

        # Intent
        task_intent = f"## Task Intent\n\n{self.task.intent}\n\n"

        # Rules
        task_rules = ""
        if self.task.rules:
            task_rules = "## Rules\n\n" + "\n".join(f"* {rule}" for rule in self.task.rules) + "\n\n"

        # Role
        role_str = f"## Role\n{self.task.role}\n\n" if self.task.role else ""

        # SectionUsage + SectionData
        sections_str = SectionDataComposer.compose_sections_with_usage(
            self.section_usages,
            self.section_data_list
        )

        # Output specification
        output_description = f"## Expected Output\n\n{self.task.output_description}\n\n"

        if self.task.output_spec:
            output_spec_str = f"## Output Specification\n\n{self.task.output_spec}\n\n"
        else:
            output_spec_str = ""

        # Reasoning guidance
        reasoning = f"## Reasoning Guidance\n\n{self.task.reasoning_guidance}\n\n" if self.task.reasoning_guidance else ""

        # Compose final prompt
        prompt = "\n".join([
            task_header,
            task_intent,
            task_rules,
            role_str,
            sections_str,
            output_description,
            output_spec_str,
            reasoning
        ])
        return prompt
    
    def compose_messages(self, user_prompt: str | None = None) -> Sequence[InputMessage]:
        system_prompt = self.compose_prompt()
        messages = [InputMessage(role="system", content=system_prompt)]
        if user_prompt:
            messages.append(InputMessage(role="user", content=user_prompt))
        return messages
