import json 
from typing import Annotated, Sequence, Any, Mapping, Type, Literal
from pydantic import BaseModel, Field

# --- 基本型の定義 ---
type StructuredText = Annotated[
    str,
    Field(
        description=(
            "Human-readable, LLM-friendly text with explicit structure. "
            "Not machine-validated, not self-describing, and no implicit semantics."
        )
    ),
]

type BundleKind = Annotated[
    str,
    Field(description="Identifier for the kind/type of Section.")
]

type SectionDescription = Annotated[
    str,
    Field(description="Human-readable explanation of the section's purpose and contents.")
]

class TextSectionData(BaseModel, frozen=True):
    """Section representing a structured text fragment."""
    bundle_kind: BundleKind
    description: SectionDescription
    structured_text: StructuredText
    type: Literal["text"] = "text"
    
    def compose_str(self) -> str:
        return (
        f"### Description\n\n{self.description}\n\n"
        f"### Structured Text\n\n{self.structured_text}\n"
        )


class ModelSectionData[T: BaseModel](BaseModel, frozen=True):
    """Section representing a sequence of BaseModel instances with optional schema."""
    bundle_kind: BundleKind
    description: SectionDescription
    data: Sequence[T] = Field(..., description="Sequence of BaseModel instances represented by this section.")
    schema_model: Annotated[type[T] | None, Field(description=(
            "Optional schema model describing the structure of the data. "
            "If None, the schema is inferred from the elements in `data`."
        ))] = None

    type: Literal["model"] = "model"

    def compose_str(self) -> str:
        data_dump = [item.model_dump() for item in self.data]
        json_schemas = (
            [self.schema_model.model_json_schema()]
            if self.schema_model
            else [cls.model_json_schema() for cls in set(type(item) for item in self.data)]
        )
        schema_fragments = "\n\n".join(
            "\n```json\n" + json.dumps(schema, indent=2, ensure_ascii=False) + "\n```"
            for schema in json_schemas
        )
        return (
            f"### Description\n\n{self.description}\n\n"
            f"### Json Schemas used in this SECTION\n\n{schema_fragments}\n\n"
            f"### Json Data\n\n{json.dumps(data_dump, indent=4, ensure_ascii=False)}\n\n"
        )
    

type SectionData = TextSectionData | ModelSectionData
