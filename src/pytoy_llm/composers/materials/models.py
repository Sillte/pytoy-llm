from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Self

from pydantic import BaseModel, Field

from pytoy_llm.composer.models import SupplementarySectionProtocol, SupplementarySections
from pytoy_llm.materials.models import MaterialData


class MaterialUsage(BaseModel, frozen=True):
    """Describes how the material should be used when solving the task."""

    usage: Annotated[
        str,
        Field(
            description=("Human-readable guidance describing how the material should be used when solving the task."),
        ),
    ]


def build_material_sections(material_sections: Sequence[MaterialSection]) -> SupplementarySections:
    description = """Supplementary Section consists of `MATERIAL`. 
`Task Usage` describes how to utilize the information for solving Task.
`Data` describes the data itself and its meta information.  
``
    """.strip()
    return SupplementarySections(sections=material_sections, description=description)


@dataclass(frozen=True)
class MaterialSection(SupplementarySectionProtocol):
    name: str
    usage: MaterialUsage
    data: MaterialData

    def compose(self, header_depth: int) -> str:
        header_prefix = "#" * header_depth + " "
        sub_header_depth = header_depth + 1
        sub_header_prefix = "#" * sub_header_depth + " "

        header_title = f"{header_prefix}Material {self.name}"
        header_usage = f"{sub_header_prefix}Task Usage"
        header_data = f"{sub_header_prefix}Data"
        body_data = self.data.compose_explanation(sub_header_depth)
        return _join_blocks([header_title, header_usage, self.usage.usage, header_data, body_data])

    @classmethod
    def from_any(
        cls,
        name: str,
        usage: MaterialUsage | str,
        data: MaterialData,
    ) -> Self:
        if isinstance(usage, str):
            usage = MaterialUsage(usage=usage)

        return cls(
            name=name,
            usage=usage,
            data=data,
        )

    @classmethod
    def build_supplementary_sections(cls, material_sections: Sequence[Self]) -> SupplementarySections:
        return build_material_sections(material_sections)


def _join_blocks(blocks: Sequence[str]) -> str:
    blocks = [block.strip("\n") for block in blocks]
    blocks = [block for block in blocks if block]
    return "\n\n".join(blocks)
