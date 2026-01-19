from pytoy_llm import completion
from pytoy_llm.models import InputMessage
from pydantic import BaseModel
from textwrap import dedent
from typing import Sequence, Literal


def construct_basemodel[T:BaseModel](user_prompt: str,
                        instances: Sequence[BaseModel],
                        output_type: type[T],
                        output_format: type[str] | type[T]) -> str | T:

    if output_format is str:
        system_prompt = make_system_prompt(output_type, instances, "python_code")
    else:
        system_prompt = make_system_prompt(output_type, instances, "instance")
    user_prompt = make_user_prompt(user_prompt)

    messages = [InputMessage(role="system", content=system_prompt), 
                InputMessage(role="user", content=user_prompt), 
    ]
    return completion(messages, output_format=output_format)
                        


def make_system_prompt(
    output_cls: type[BaseModel],
    instances: Sequence[BaseModel],
    output_mode: Literal["python_code", "instance"] = "python_code",
) -> str:
    """
    Generate a system prompt for LLM.
    
    output_mode:
        - "python_code": produce Python code constructing the target object
        - "instance": produce the actual BaseModel instance
    """
    output_schema = output_cls.model_json_schema()
    target_class_name = output_schema["title"]

    # collect all unique classes
    schema_origin_cls_list = list(set(type(elem) for elem in instances) | {output_cls})

    # JSON schema string
    def _to_one_json_schema(cls: type[BaseModel], tag_name="json-schema") -> str:
        return f"<{tag_name}> {cls.model_json_schema()}/<{tag_name}>"
    json_schema_str = "\n".join(_to_one_json_schema(cls) for cls in schema_origin_cls_list)

    # Instances string
    def _to_one_instance(inst: BaseModel, tag_name="json") -> str:
        return f"<{tag_name}> {inst.model_dump_json()}/<{tag_name}>"
    instances_str = "\n".join(_to_one_instance(item) for item in instances)

    # Decide output instruction
    if output_mode == "python_code":
        output_instr = dedent(f"""
        Produce valid Python code that constructs a `{target_class_name}` instance.
        Use `None` or `null` for fields if necessary when you cannot infer them from the user's input.
        Do not include explanations or comments.
        Example:
        ```python
        ParameterClass(param_int=5, 
                       param_str="hello", 
                       param_cls=ChildClass(val=2))
        ```
        """)
    else:  # instance
        output_instr = dedent(f"""
        Produce a valid `{target_class_name}` instance directly.
        Use `None` for fields if necessary when you cannot infer them from the user's input.
        Do not include explanations or comments.
        """).strip()

    system_prompt = dedent(f"""
    =====================
    LLM Role / Rules
    =====================

    You are a construction assistant.
    Your role is to produce a valid instance of `{target_class_name}` strictly following the examples provided.

    Rules:
    - Do NOT invent new relationships not observed in the instances.
    - Do NOT include other information than ones to construct the target object.
    - Do not add extra explanations or commentary.
    - Respect field descriptions as guidance.

    =====================
    References (Context Information)
    =====================

    ## Json Schema
    {json_schema_str}

    ## Instances
    {instances_str}

    =====================
    Output Format
    =====================

    {output_instr}
    """).strip()

    return system_prompt


def make_user_prompt(user_prompt: str) -> str:
    """
    Wrap the user request into a separate prompt message.
    """
    return dedent(f"""
    =====================
    User Request
    =====================
    {user_prompt}
    """).strip()

