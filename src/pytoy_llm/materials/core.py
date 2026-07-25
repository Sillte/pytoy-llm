import json
import warnings
from collections.abc import Sequence
from typing import Annotated, Literal

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

type BundleKind = Annotated[str, Field(description="Identifier for the kind/type of Section.")]

type SectionDescription = Annotated[str, Field(description="Human-readable explanation of the section's purpose and contents.")]


class TextSectionData(BaseModel, frozen=True):
    """Section representing a structured text fragment."""

    bundle_kind: BundleKind
    description: SectionDescription
    structured_text: StructuredText
    type: Literal["text"] = "text"

    def compose_str(self) -> str:
        warn_forbidden_headers(self.description)

        # The below comment out for `warn_forbiddenn_headers` is intentional.
        # Since `structure_text` is free format inside the tag.
        # warn_forbidden_headers(self.structured_text)

        return f"#### Description\n\n{self.description}\n\n#### Structured Text\n\n{self.structured_text}\n"


class ModelSectionData[T: BaseModel](BaseModel, frozen=True):
    """Section representing a sequence of BaseModel instances with optional schema."""

    bundle_kind: BundleKind
    description: SectionDescription
    instances: Sequence[T] = Field(..., description="Sequence of BaseModel instances represented by this section.")
    schema_model: Annotated[
        type[T] | None,
        Field(
            description=(
                "Optional schema model describing the structure of the data. "
                "If None, the schema is inferred from the elements in `data`."
            )
        ),
    ] = None

    type: Literal["model"] = "model"

    def compose_str(self) -> str:
        warn_forbidden_headers(self.description)
        if not self.instances:
            return f"#### Desciption\n\n{self.description}No json exist.\n"

        data_parts = []
        for item in self.instances:
            part = f"```json\n{item.model_dump_json()}```"
            data_parts.append(part)
        json_instance_str = "\n".join(data_parts)

        json_schemas = (
            [self.schema_model.model_json_schema()]
            if self.schema_model
            else [cls.model_json_schema() for cls in set(type(item) for item in self.instances)]
        )
        schema_fragments = "\n\n".join(
            "\n```json\n" + json.dumps(schema, indent=2, ensure_ascii=False) + "\n```" for schema in json_schemas
        )
        return (
            f"#### Description\n\n{self.description}\n\n"
            f"#### Json Schemas\n\n{schema_fragments}\n\n"
            f"#### Json Instance\n\n{json_instance_str}\n\n"
        )


type SectionData = TextSectionData | ModelSectionData


def warn_forbidden_headers(text: str, min_allowed_header_level: int = 4, skip_first: bool = True) -> None:
    """
    Check each line. Warn if a header is too high (e.g., # or ##)
    compared to the minimum allowed header level.
    """
    if skip_first:
        start = 1
    else:
        start = 0
    for i, line in enumerate(text.splitlines(), start=start):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            header_level = len(stripped) - len(stripped.lstrip("#"))
            if header_level < min_allowed_header_level:
                warnings.warn(
                    f"Line {i}: header level {header_level} "
                    f"is below minimum allowed ({min_allowed_header_level}). "
                    "Consider deeper headers for injected structure.",
                    UserWarning,
                )
