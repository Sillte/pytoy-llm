from pytoy_llm import completion
from pytoy_llm.models import InputMessage
from pydantic import BaseModel
from textwrap import dedent
from typing import Sequence, Literal

from pytoy_llm.materials.composers.models import LLMTask, SectionUsage, SectionData, TextSectionData
from pytoy_llm.materials.core import TextSectionData
from pytoy_llm.materials.basemodels import BaseModelBundle
from pytoy_llm.materials.composers.task_prompt_composer import TaskPromptComposer


def construct_basemodel[T:BaseModel](user_prompt: str,
                        instances: Sequence[BaseModel],
                        output_type: type[T],
                        output_format: type[str] | type[T], 
                     *, explanation: str | None = None) -> str | T:
    """
    explanation: If addtional explanation is necessary for construction of BaseModel.
    Please provide them.
    """

    if output_format is str:
        system_prompt = make_system_prompt(output_type, instances, "python_code", explanation=explanation)
    else:
        system_prompt = make_system_prompt(output_type, instances, "instance", explanation=explanation)
    user_message= InputMessage(role="user", content=user_prompt)
    messages = [InputMessage(role="system", content=system_prompt), user_message] 
    return completion(messages, output_format=output_format)
                        
def make_system_prompt(
    output_cls: type[BaseModel],
    instances: Sequence[BaseModel],
    output_mode: Literal["python_code", "instance"] = "python_code",
    *,
    explanation : str | None = None
) -> str:

    output_schema = output_cls.model_json_schema()
    target_class_name = output_schema["title"]

    bundle = BaseModelBundle(data=instances)
    section_data_list: list[SectionData] = [bundle.model_section_data]
    usages: list[SectionUsage]  = []
    
    usage = SectionUsage(
        bundle_kind=bundle.bundle_kind,
        usage_rule=[
            "Use these examples as reference.",
            "Follow the structure exactly.",
            "The output MUST be regarded natual as one of examples of the reference.",
            "Respect field descriptions as guidance."
        ]
    )
    usages.append(usage)

    # Decide output instruction
    if output_mode == "python_code":
        output_spec = dedent(f"""
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
    else:  # instance
        output_spec = dedent(f"""
        Produce a valid `{target_class_name}` instance directly via `json`.
        Use `None` for fields if necessary when you cannot infer them from the user's input.
        Do not include explanations or comments.
        """).strip()

    task = LLMTask(
        name="Construct BaseModel Instances",
        intent=(
        "Construct instances strictly following the examples provided.\n"
        "Do not invent new relationships not observed in the instances.\n"
        "Follow the field descriptions as guidance.\n"
        ),
        rules=[
            "Do NOT invent new relationships not observed in the instances.",
            "Do NOT add extra explanations or commentary."
        ],
        output_description=f"Instance of {target_class_name}",
        output_spec=output_spec,
        role=f"You are a construction assistant. You have responsibility and pride for generating useful `{target_class_name}`"
    )
    if explanation:
        bundle_kind = "AdditionalExplanation"
        section_data = TextSectionData(bundle_kind=bundle_kind,
                        description=explanation,
                        structured_text=explanation)
        usage = SectionUsage(bundle_kind=bundle_kind, 
                             usage_rule=["This section provides problem-specific hints not covered by the examples."])
        usages.append(usage)
        section_data_list.append(section_data)
        

    composer = TaskPromptComposer(task, usages, section_data_list)
    return composer.compose_prompt()



if __name__ == "__main__":
    from pydantic import BaseModel
    from typing import Sequence

    class SampleModel(BaseModel):
        name: str
        value: int

    # --- 既存例（参考用） ---
    examples: Sequence[SampleModel] = [
        SampleModel(name="example1", value=10),
        SampleModel(name="example2", value=20),
        SampleModel(name="example3", value=50),
    ]


    # --- ユーザーの意図は曖昧に与える ---
    user_input = (
        "Create a SampleModel instance with a common name and a high value. "
        "Refer to the examples for guidance."
    )

    explanation = "The maximum value of `SampleModel.value` is about 10000."

    # --- LLMに投げる ---
    result_instance = construct_basemodel(
        user_prompt=user_input,
        instances=examples,
        output_type=SampleModel,
        output_format=SampleModel,  # 直接BaseModelインスタンス,
        explanation=explanation
    )

    print("result_instance", result_instance)

    print("Generated SampleModel:", result_instance)
