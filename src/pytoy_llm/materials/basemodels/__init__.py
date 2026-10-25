from collections.abc import Sequence

from pydantic import BaseModel

from pytoy_llm.materials.models import JsonMaterialData, TextMaterialData


class BaseModelMaterial[T: BaseModel](BaseModel, frozen=True):
    """Container holding multiple `pydantic.BaseModel`"""

    data: Sequence[T]

    @property
    def text_material_data(self) -> TextMaterialData:
        structured_text = self.structured_text
        description = self.description
        return TextMaterialData(content=structured_text, description=description)

    @property
    def json_material_data(self) -> JsonMaterialData:
        # Note: This requires a memory space of
        # text data.
        # If we would like to use the big data,
        # `chunk` or `iter` iteration is necessary regarding `data`.
        return JsonMaterialData.from_basemodel(description=self.description, instances=self.data)

    @property
    def description(self) -> str:
        description = (
            "This section contains multiple instances of `pydantic.BaseModel`\n"
            "A JSON Schema and the JSON data are provided below."
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
