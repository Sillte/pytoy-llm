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
        section_bundle_kinds = {data.bundle_kind: data for data in data_list}
        kind_to_pair: dict[str, tuple[SectionUsage, SectionData]] = {}
        for usage in usages:
            if usage.bundle_kind not in section_bundle_kinds:
                raise ValueError(
                    f"SectionUsage.bundle_kind='{usage.bundle_kind}' "
                    f"does not match any SectionData.bundle_kind"
                )
            kind_to_pair[usage.bundle_kind] = (usage, section_bundle_kinds[usage.bundle_kind])

        lines = ["--------------------SECTIONS---------------------",]
        
        for bundle_kind, (usage, data) in kind_to_pair.items():
            lines.append(f"## Section (bundle_kind=`{bundle_kind}`)")
            lines.append(f"### Section Instruction (bundle_kind=`{bundle_kind}`)")
            lines.append("1. Read the `Section Usage` rules carefully.")
            lines.append("2. Read the `Section Data` content rules carefully.")
            lines.append("3. Refer to `Usage` to determine how to utilize `Data`.")
            lines.append(f"### Section Usage (bundle_kind=`{bundle_kind}`)")
            lines.append(f"#### Section Rules for (bundle_kind=`{bundle_kind}`)")
            rules_text = "\n".join(f"* {rule}" for rule in usage.usage_rule)
            lines.append(rules_text)
            lines.append(f"### Section Data (bundle_kind=`{bundle_kind}`)")
            lines.append(data.compose_str())
            lines.append("\n\n")
        lines.append("-----------------------------------------")
        return "\n".join(lines)
            
    def compose(self) -> str:
        header = f"----------[SECTION (bundle_kind=`{self.section.bundle_kind}`)]----------\n"
        if isinstance(self.section, TextSectionData):
            body = self.section.compose_str()
        elif isinstance(self.section, ModelSectionData):
            body = self.section.compose_str()
        else:
            raise TypeError(f"Unknown SectionData type: {type(self.section)}")
        return f"{header}\n{body}\n\n"
            
