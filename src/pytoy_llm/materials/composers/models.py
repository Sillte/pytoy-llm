import json
from typing import Annotated, Sequence
from pydantic import BaseModel, Field
from pytoy_llm.materials.core import TextSectionData, ModelSectionData, SectionData

class LLMTask(BaseModel):
    name: Annotated[
        str,
        Field(description="Human-readable task name")
    ]

    intent: Annotated[
        str,
        Field(description="What the LLM is expected to do")
    ]

    rules: Annotated[
        Sequence[str],
        Field(description="Hard constraints the LLM must follow")
    ]

    output_description: Annotated[
        str,
        Field(description="Semantic meaning of the output")
    ]

    output_spec: Annotated[
        type | str,
        Field(description="Specification of the expected output (type or format hint)")
    ]

    reasoning_guidance: Annotated[
        str | None,
        Field(
            description=(
                "Optional guidance for the LLM's reasoning process "
                "(e.g., chain-of-thought, tree-of-thought). "
                "Intended for performance optimization and may be ignored."
            )
        )
    ] = None

    role: Annotated[
        str | None,
        Field(description="Role or persona the LLM should assume for this task")
    ] = None



class SectionUsage(BaseModel, frozen=True):
    """
    Represents how a Section (identified by bundle_kind) should be used
    in a system prompt or LLM interaction.
    """
    bundle_kind: Annotated[
        str,
        Field(description="Identifier of the Section. Corresponds to `bundle_kind`")
    ]
    usage_rule: Annotated[
        Sequence[str],
        Field(description="List of rules describing how this Section should be used. Each entry becomes a bullet point in the prompt.")
    ]

    @classmethod
    def compose_usage_rule(cls) -> str:
        return ("## Usage of SECTIONS with `bundle_kind`\n\n"
                "This prompt includes the following SECTIONS with `bundle_kind`. \n"
                "Please utilize them as following instructions.\n"
                )
        
    @classmethod
    def compose_from_usages(cls, usages: Sequence["SectionUsage"]) -> str:
        if not usages:
            return "\n\n"
        parts = []
        parts.append(cls.compose_usage_rule())
        for usage in usages:
            parts.append(usage.compose_fragment())
            parts.append("\n")
        return "\n".join(parts)


    def compose_fragment(self) -> str:
        """
        Generate a human-readable representation of the usage rules
        suitable for inclusion in a system prompt.
        """
        rules_text = "\n".join(f"* {rule}" for rule in self.usage_rule)
        return f"### Usage for SECTION (bundle_kind = `{self.bundle_kind}`)\n\n{rules_text}\n"

class SectionDataComposer:
    """
    Responsible only for converting a single SectionData into
    its string representation suitable for LLM prompt.
    """
    def __init__(self, section: SectionData):
        self.section = section
        
    @classmethod
    def compose_sections_with_usage(cls, usages: Sequence["SectionUsage"], data_list: Sequence["SectionData"]) -> str:
        # 照合チェック: usage の bundle_kind が sections に存在するか
        section_bundle_kinds = {data.bundle_kind for data in data_list}
        for usage in usages:
            if usage.bundle_kind not in section_bundle_kinds:
                raise ValueError(
                    f"SectionUsage.bundle_kind='{usage.bundle_kind}' "
                    f"does not match any SectionData.bundle_kind"
                )
        usage_section = SectionUsage.compose_from_usages(usages) 
        sections = [cls(data).compose() for data in data_list]
        return "\n\n".join([usage_section, *sections])
            
    def compose(self) -> str:
        header = f"----------[SECTION (bundle_kind=`{self.section.bundle_kind}`)]----------\n"
        if isinstance(self.section, TextSectionData):
            body = self.section.compose_str()
        elif isinstance(self.section, ModelSectionData):
            body = self.section.compose_str()
        else:
            raise TypeError(f"Unknown SectionData type: {type(self.section)}")
        return f"{header}\n{body}\n\n"
            
