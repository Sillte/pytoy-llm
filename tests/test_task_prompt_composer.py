from pydantic import BaseModel

from pytoy_llm.composers.models import OutputSpec, SystemPromptSpec
from pytoy_llm.composers.system_prompt_composer import SystemPromptComposer
from pytoy_llm.materials.models import (
    MaterialSection,
    MaterialUsage,
    ModelMaterialData,
    TextMaterialData,
    build_material_sections,
)
from pytoy_llm.models.llm_messages import LLMMessage


class SampleModel(BaseModel):
    name: str
    value: int


def test_system_prompt_composer_with_material_sections():
    # --- System prompt specification ---
    prompt_spec = SystemPromptSpec.from_any(
        name="Sample invocation",
        intent="Rewrite the following text to be more concise.",
        rules=[
            "Do not change meaning.",
            "Keep technical terms intact.",
        ],
        output_spec=OutputSpec(
            output_type=str,
            description="Rewritten text as string.",
        ),
        guidance_role="Editor",
        reasoning_guidance="Consider sentence merging if it improves clarity.",
    )

    # --- Material usage ---
    text_usage = MaterialUsage(
        usage="\n".join(
            [
                "Use these examples as reference.",
                "Follow the style shown in examples.",
            ]
        ),
    )

    model_usage = MaterialUsage(
        usage="\n".join(
            [
                "Use these examples as models.",
                "Utilize the observed structure when constructing output.",
            ]
        ),
    )

    # --- Material data ---
    text_material = TextMaterialData(
        description="Example sentences to guide rewriting.",
        content="This is a long example sentence that could be improved.",
    )

    model_material = ModelMaterialData[SampleModel](
        description="Sample model instances.",
        instances=[
            SampleModel(name="a", value=1),
            SampleModel(name="b", value=2),
        ],
    )

    # --- Material sections ---
    sections = [
        MaterialSection(
            name="TextExamples",
            usage=text_usage,
            data=text_material,
        ),
        MaterialSection(
            name="ModelData",
            usage=model_usage,
            data=model_material,
        ),
    ]

    supplementary_sections = build_material_sections(sections)

    # --- Compose system prompt ---
    composer = SystemPromptComposer(prompt_spec)
    prompt_str = composer.compose_prompt(
        supplementary_sections=supplementary_sections,
    )

    # --- Basic prompt checks ---
    assert "Sample invocation" in prompt_str
    assert "Rewrite the following text" in prompt_str
    assert "Do not change meaning." in prompt_str
    assert "Keep technical terms intact." in prompt_str

    # Output specification
    assert "Task Output" in prompt_str
    assert "plain text string" in prompt_str
    assert "Rewritten text as string." in prompt_str

    # Auxiliary guidance
    assert "Auxiliary Guidance" in prompt_str
    assert "Editor" in prompt_str
    assert "Consider sentence merging" in prompt_str

    # Supplementary sections
    assert "Supplementary Sections" in prompt_str
    assert "TextExamples" in prompt_str
    assert "ModelData" in prompt_str
    assert "Example sentences to guide rewriting." in prompt_str
    assert "Sample model instances." in prompt_str

    # Material usage
    assert "Follow the style shown in examples." in prompt_str
    assert "Utilize the observed structure when constructing output." in prompt_str

    print(prompt_str)

    # --- Compose message ---
    message = LLMMessage.from_prompt(
        user="UserPrompt",
        system=prompt_str,
    )
    assert isinstance(message, LLMMessage)


if __name__ == "__main__":
    test_system_prompt_composer_with_material_sections()
