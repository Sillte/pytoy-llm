from collections.abc import Sequence

from pydantic import BaseModel

from pytoy_llm.materials.models import ModelMaterialData, TextMaterialData


class BaseModelBundle[T: BaseModel](BaseModel, frozen=True):
    """Container holding multiple `pydantic.BaseModel`"""

    data: Sequence[T]

    @property
    def text_section_data(self) -> TextMaterialData:
        structured_text = self.structured_text
        description = self.description
        return TextMaterialData(content=structured_text, description=description)

    @property
    def model_section_data(self) -> ModelMaterialData:
        # Note: `TextFileBundleData` requires a memory space of
        # text data.
        # If we would like to use the big data,
        # `chunk` or `iter` iteration is necessary regarding `data`.
        return ModelMaterialData(description=self.description, instances=self.data)

    @property
    def description(self) -> str:
        description = (
            "This section contains multiple instances of `pydantic.BaseModel`\n"
            "Both of Json Schemas and Json Data are given as below."
        )
        return description

    @property
    def structured_text(self) -> str:
        """Returns a structured text representation of the documents for LLM consumption."""
        cls_names = [str(type(elem)) for elem in self.data]
        return (
            f"===BaseModelList===\n"
            f"Classes:{','.join(cls_names)}\n"
            f"When you would like to use json-schemas or json-instances, please notify the caller.\n"
            f"Because in this mode, JsonSchema and JsonInstances are ommited."
        )
