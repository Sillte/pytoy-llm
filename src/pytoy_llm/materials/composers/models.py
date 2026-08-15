from collections.abc import Sequence
from dataclasses import dataclass

# NEW DESIGN....
from pytoy_llm.composers.models import SupplementarySectionProtocol, SupplementarySections
from pytoy_llm.materials.models import MaterialData, MaterialUsage


@dataclass(frozen=True)
class MaterialSection(SupplementarySectionProtocol):
    name: str
    usage: MaterialUsage
    section_data: MaterialData

    def compose(self, header_depth: int) -> str:
        header_prefix = "#" * header_depth + " "
        sub_header_depth = header_depth + 1
        sub_header_prefix = "#" * sub_header_depth + " "

        header_title = f"{header_prefix}MATERIAL {self.name}\n\n"
        header_usage = f"{sub_header_prefix}MATERIAL {self.name} -Task usage-\n\n"
        body_usage = self._compose_usage_text_body(self.usage)
        header_instance = f"{sub_header_prefix}Materials {self.name} -Instances-\n\n"
        body_instance = self._compose_instances_text_body(self.section_data)

        return "\n".join([header_title, header_usage, body_usage, header_instance, body_instance])

    def _compose_usage_text_body(self, usage: MaterialUsage) -> str:
        rules_text = "\n".join(f"* {rule}" for rule in self.usage.usage_rule)
        return f"{rules_text}\n\n"

    def _compose_instances_text_body(self, section_data: MaterialData) -> str:
        return section_data.compose_body()


def build_material_sections(material_sections: Sequence[MaterialSection]) -> SupplementarySections:
    description = """Supplementary Section consists of `MATERIAL`. 
`Task usage` describes how to utilize the information for solving Task.
`Instances` describes the data itself and its meta information.  
``
    """.strip()
    return SupplementarySections(sections=material_sections, description=description)
