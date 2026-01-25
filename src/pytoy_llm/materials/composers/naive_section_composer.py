from typing import Sequence, assert_never
from pytoy_llm.materials.core import SectionData, TextSectionData, ModelSectionData
import json


class NaiveSectionComposer:
    """Only dump the information of `SectionData`.
    """
    def __init__(self, sections: Sequence[SectionData]) -> None:
        self._sections: Sequence[SectionData] = sections
        
    def compose_prompt(self) -> str:
        separator = "=" * 80 + "\n\n"
        contents: list[str] = []
        for section in self._sections:
            header = f"----------[SECTION {section.bundle_kind} ]----------\n"
            body = section.compose_str()
            content = f"{header}\n{body}\n\n"
            contents.append(content)
        return separator.join(contents)


