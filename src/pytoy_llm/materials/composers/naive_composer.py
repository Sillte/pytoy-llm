from typing import Sequence, assert_never
from pytoy_llm.materials.core import SectionData, TextSectionData, ModelSectionData
import json


class NaiveComposer:
    """Only dump the information of `SectionData`.
    """
    def __init__(self, sections: Sequence[SectionData]) -> None:
        self._sections: Sequence[SectionData] = sections

    def from_model_section_data(self, model_section_data: ModelSectionData) -> str:
        data_dump = [
            item.model_dump() for item in model_section_data.data
        ]
        if (schema_model:=model_section_data.schema_model):
            json_schemas = [schema_model.model_json_schema()]
        else:
            classes = set(type(item) for item in model_section_data.data)
            json_schemas = [cls.model_json_schema() for cls in classes]

        fragments = []
        for schema in json_schemas:
            fragment = ("\n```json\n"
                        f"{json.dumps(schema, indent=2, ensure_ascii=False)}\n"
                        f"```\n")
            fragments.append(fragment)
        return (
            f"----------[SECTION {model_section_data.bundle_kind} ]----------\n"
            "### Description\n\n"
            f"{model_section_data.description}\n\n"
            "### Json Schemas used in this SECTION\n\n"
            f"{'\n\n'.join(fragments)}\n\n"
            "### Json Data\n\n"
            f"{json.dumps(data_dump, indent=4, ensure_ascii=False)}\n\n"
        )


    def from_text_section_data(self, text_section_data: TextSectionData) -> str:
        data = text_section_data

        return (
            f"----------[SECTION {data.bundle_kind} ]----------\n"
            f"### Description\n\n"
            f"{data.description}\n\n"
            f"### Structured Text\n\n"
            f"{data.structured_text}\n"
        )

    def compose(self) -> str:
        separator = "=" * 80 + "\n\n"
        contents: list[str] = []
        for section in self._sections:
            if isinstance(section, TextSectionData):
                content = self.from_text_section_data(section)
            elif isinstance(section, ModelSectionData):
                content = self.from_model_section_data(section)
            else:
                assert_never(section)
            contents.append(content)
        return separator.join(contents)


