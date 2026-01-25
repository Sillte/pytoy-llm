# test_task_prompt_composer.py
import pytest
from pydantic import BaseModel
from typing import Sequence
from pytoy_llm.materials.composers.models import LLMTask, SectionUsage, SectionDataComposer
from pytoy_llm.materials.composers.task_prompt_composer import TaskPromptComposer
from pytoy_llm.materials.core import TextSectionData, ModelSectionData

class SampleModel(BaseModel):
    name: str
    value: int

def test_task_prompt_composer_basic():
    # --- サンプル Task ---
    task = LLMTask(
        name="Sample Task",
        intent="Rewrite the following text to be more concise.",
        rules=["Do not change meaning", "Keep technical terms intact"],
        output_description="Rewritten text as string",
        output_spec=str,
        reasoning_guidance="Consider sentence merging if it improves clarity.",
        role="Editor"
    )

    # --- サンプル SectionUsage ---
    section_usages = [
        SectionUsage(
            bundle_kind="TextExamples",
            usage_rule=[
                "Use these examples as reference.",
                "Follow the style shown in examples."
            ]
        )
    ]

    # --- サンプル SectionData ---
    text_section = TextSectionData(
        bundle_kind="TextExamples",
        description="Example sentences to guide rewriting",
        structured_text="This is a long example sentence that could be improved."
    )

    model_section = ModelSectionData[SampleModel](
        bundle_kind="ModelData",
        description="Sample model instances",
        data=[SampleModel(name="a", value=1), SampleModel(name="b", value=2)]
    )

    section_data_list = [text_section, model_section]

    # --- Compose prompt ---
    composer = TaskPromptComposer(task, section_usages, section_data_list)
    prompt_str = composer.compose_prompt()

    # --- 簡単なチェック ---
    assert "Sample Task" in prompt_str
    assert "Rewrite the following text" in prompt_str
    assert "### Usage for SECTION (bundle_kind = `TextExamples`)" in prompt_str
    assert "Example sentences to guide rewriting" in prompt_str
    assert "Sample model instances" in prompt_str

    messages = composer.compose_messages(user_prompt=None)
    assert len(messages) == 1

    messages = composer.compose_messages(user_prompt="UserPrompt")
    assert len(messages) == 2



