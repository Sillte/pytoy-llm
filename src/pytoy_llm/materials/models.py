import json
import warnings
from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, Field

type StructuredText = Annotated[
    str,
    Field(
        description=(
            "Human-readable, LLM-friendly text with explicit structure. "
            "Not machine-validated, not self-describing, and no implicit semantics."
        )
    ),
]


class TextMaterialData(BaseModel, frozen=True):
    """
    Material data represented as structured human-readable text.

    The text is provided as contextual material for the LLM and
    does not have machine-validated semantics.
    """

    description: Annotated[
        str,
        Field(
            description=("Human-readable description of what this material contains and what it represents."),
        ),
    ]

    content: StructuredText

    type: Literal["text"] = "text"

    def compose_explanation(self, parent_header_depth: int) -> str:
        """Compose the material body under a parent Markdown section."""
        # The below comment out for `warn_forbiddenn_headers` is intentional.
        # Since `structure_text` is free format inside the tag.
        sub_header_depth = parent_header_depth + 1
        sub_header_prefix = "#" * sub_header_depth
        header_description = f"{sub_header_prefix} Description"
        header_content = f"{sub_header_prefix} Content"
        return _join_blocks([header_description, self.description, header_content, self.content])


class ModelMaterialData[T: BaseModel](BaseModel, frozen=True):
    """
    Material data represented as JSON instances with their JSON Schema.

    The instances provide concrete examples of the material,
    while the JSON Schema describes their structure and fields.
    """

    description: Annotated[str, Field(description="Human-readable explanation of the section's purpose and contents.")]
    instances: Sequence[T] = Field(
        ...,
        description="Concrete JSON instances representing the material.",
    )
    schema_model: Annotated[
        type[T] | None,
        Field(
            description=(
                "Optional model used to generate the JSON Schema describing "
                "the structure of the material. If omitted, the schema is "
                "inferred from the provided instances."
            )
        ),
    ] = None

    type: Literal["model"] = "model"

    def compose_explanation(self, parent_header_depth: int) -> str:

        sub_header_depth = parent_header_depth + 1
        sub_header_prefix = "#" * sub_header_depth
        warn_forbidden_headers(self.description, sub_header_depth)
        blocks = [f"{sub_header_prefix} Description", self.description]

        if self.schema_model is None and (not self.instances):
            return _join_blocks([*blocks, "No data exists"])

        json_schemas = (
            [self.schema_model.model_json_schema()]
            if self.schema_model
            else [cls.model_json_schema() for cls in set(type(item) for item in self.instances)]
        )
        schema_fragments = "\n\n".join(
            "\n```json\n" + json.dumps(schema, indent=2, ensure_ascii=False) + "\n```" for schema in json_schemas
        )
        blocks = [*blocks, f"{sub_header_prefix} JSON Schemas", schema_fragments]

        data_parts = [f"```json\n{item.model_dump_json()}```" for item in self.instances]
        if data_parts:
            json_instance_str = "\n".join(data_parts)
        else:
            json_instance_str = "**NO DATA**"
        blocks = [*blocks, f"{sub_header_prefix} JSON Instances", json_instance_str]
        return _join_blocks(blocks)


type MaterialData = TextMaterialData | ModelMaterialData


def _join_blocks(blocks: Sequence[str]) -> str:
    blocks = [block.strip("\n") for block in blocks]
    blocks = [block for block in blocks if block]
    return "\n\n".join(blocks)


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
