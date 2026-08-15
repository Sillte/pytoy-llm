from collections.abc import Sequence
from textwrap import dedent
from typing import Literal

from pydantic import BaseModel

from pytoy_llm import completion
from pytoy_llm.composers.models import OutputSpec, SystemPromptSpec
from pytoy_llm.composers.system_prompt_composer import SystemPromptComposer
from pytoy_llm.materials.basemodels import BaseModelBundle
from pytoy_llm.materials.models import (
    MaterialSection,
    MaterialUsage,
    TextMaterialData,
    build_material_sections,
)
from pytoy_llm.models.llm_messages import LLMMessage


def construct_basemodel[T: BaseModel](
    user_prompt: str,
    instances: Sequence[T],
    output_mode: Literal["python_code", "instance"] = "python_code",
    *,
    explanation: str | None = None,
) -> str | T:
    """
    explanation: If addtional explanation is necessary for construction of BaseModel.
    Please provide them.
    """
    if not instances:
        raise ValueError("Must provide at least one `instances`.")

    system_prompt = make_system_prompt(instances, output_mode, explanation=explanation)
    message = LLMMessage.from_prompt(system=system_prompt, user=user_prompt)
    output_type = str if output_mode == "python_code" else type(instances[0])
    return completion(message, output_type=output_type)


def make_system_prompt[T: BaseModel](
    instances: Sequence[T], output_mode: Literal["python_code", "instance"] = "python_code", *, explanation: str | None = None
) -> str:
    if not instances:
        raise ValueError("Must provide at least one `instances`.")
    interest_type = type(instances[0])

    output_schema = interest_type.model_json_schema()
    target_class_name = output_schema["title"]

    sections: list[MaterialSection] = []

    bundle = BaseModelBundle(data=instances)

    usage = MaterialUsage(
        usage="\n".join(
            [
                "Use these examples as reference.",
                "Follow the structure exactly.",
                "The output MUST be regarded natual as one of examples of the reference.",
                "Respect field descriptions as guidance.",
            ]
        ),
    )
    sections.append(MaterialSection(name="BaseModelBundle", usage=usage, data=bundle.model_section_data))

    # Decide output instruction
    if output_mode == "python_code":
        output_description = dedent(f"""
        The output must be a statement of python code. 
        Produce valid Python code that constructs a `{target_class_name}` instance.
        Use `None` or `null` for fields if necessary when you cannot infer them from the user's input.
        Do not include explanations or comments.

        Example of outputs:
        ------------------------------
        ```python
        BaseModelClass(param_int=5, 
                       param_str="hello", 
                       param_cls=ChildClass(val=2))
        ```
        """)
        output_type = str
    else:  # instance
        output_description = dedent(f"""
        Produce a valid `{target_class_name}` instance directly via `json`.
        Use `None` for fields if necessary when you cannot infer them from the user's input.
        Do not include explanations or comments.
        """).strip()

        output_type = interest_type

    prompt_spec = SystemPromptSpec.from_any(
        name="Construct BaseModel Instances",
        intent=(
            f"Construct a useful `{target_class_name}` instance from the user's request.\n"
            "Use the provided examples and their field descriptions as the primary guidance for construction."
        ),
        rules=[
            "Do NOT invent new relationships that are not supported by the examples.",
            "Do NOT add explanations or commentary.",
        ],
        output_spec=OutputSpec(output_type=output_type, description=output_description),
    )
    if explanation:
        section_data = TextMaterialData(description=explanation, content=explanation)
        usage = MaterialUsage(usage="This section provides problem-specific hints not covered by the examples.")
        sections.append(MaterialSection(name="AddtionalExplanation", usage=usage, data=section_data))

    composer = SystemPromptComposer(prompt_spec)
    return composer.compose_prompt(supplementary_sections=build_material_sections(sections))


if __name__ == "__main__":
    from collections.abc import Sequence

    from pydantic import BaseModel

    class SampleModel(BaseModel):
        name: str
        value: int

    # --- 既存例（参考用） ---
    examples: Sequence[SampleModel] = [
        SampleModel(name="example1", value=10),
        SampleModel(name="example2", value=20),
        SampleModel(name="example3", value=50),
        SampleModel(name="example4", value=100),
    ]

    # --- ユーザーの意図は曖昧に与える ---
    user_input = "Create a SampleModel instance with a popular name and a high value. Refer to the examples for guidance."

    explanation = "The maximum value of `SampleModel.value` is about 10000."

    # --- LLMに投げる ---
    result_instance = construct_basemodel(
        user_prompt=user_input, instances=examples, output_mode="python_code", explanation=explanation
    )

    print("Generated SampleModel:", result_instance)
