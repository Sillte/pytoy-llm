import builtins
import json
import warnings
from collections.abc import Sequence
from typing import Annotated, Literal, Self, Union, cast

from pydantic import BaseModel, Field, JsonValue, TypeAdapter
from typing_extensions import TypeForm

_JSON_VALUE_ADAPTER = TypeAdapter(JsonValue)

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
            description=(
                "Human-readable description of what this material contains and what it represents."
            ),
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


class JsonMaterialData(BaseModel, frozen=True):
    """
    Material data represented by JSON data accompanied by its JSON Schema.

    The JSON Schema describes the structure of the data, while the JSON
    value provides the concrete data to be presented to the LLM.
    """

    description: Annotated[
        str,
        Field(
            description=(
                "Human-readable description of what this material contains and what it represents."
            ),
        ),
    ]

    json_schema: Annotated[
        JsonValue,
        Field(description="JSON Schema describing the structure of the data."),
    ]

    data: Annotated[
        JsonValue,
        Field(description="Concrete JSON instance corresponding to the JSON Schema."),
    ]

    type: Literal["json"] = "json"

    @classmethod
    def from_basemodel[T: BaseModel](
        cls,
        description: str,
        instances: Sequence[T],
        schema_model: builtins.type[T] | None = None,
    ) -> Self:
        model_types = tuple(dict.fromkeys(type(instance) for instance in instances))
        if schema_model is not None:
            model_types = (schema_model,)

        if model_types:
            instance_type = cast(
                TypeForm[BaseModel],
                model_types[0] if len(model_types) == 1 else Union[model_types],
            )
            item_schema = TypeAdapter(instance_type).json_schema()
            definitions = item_schema.pop("$defs", None)
            json_schema = {"type": "array", "items": item_schema}
            if definitions is not None:
                json_schema["$defs"] = definitions
        else:
            json_schema = {"type": "array", "items": {}}

        data = _JSON_VALUE_ADAPTER.validate_python(
            [instance.model_dump(mode="json") for instance in instances]
        )
        return cls(description=description, json_schema=json_schema, data=data)

    def compose_explanation(self, parent_header_depth: int) -> str:
        """Compose the material body under a parent Markdown section."""
        sub_header_depth = parent_header_depth + 1
        sub_header_prefix = "#" * sub_header_depth

        warn_forbidden_headers(self.description, sub_header_depth)

        header_description = f"{sub_header_prefix} Description"
        header_schema = f"{sub_header_prefix} JSON Schema"
        header_data = f"{sub_header_prefix} JSON Instance"

        schema_text = _compose_json_block(self.json_schema)
        data_text = _compose_json_block(self.data)

        return _join_blocks(
            [
                header_description,
                self.description,
                header_schema,
                schema_text,
                header_data,
                data_text,
            ]
        )


type MaterialData = TextMaterialData | JsonMaterialData


def _compose_json_block(value: JsonValue) -> str:
    return "```json\n" + json.dumps(value, indent=2, ensure_ascii=False) + "\n```"


def _join_blocks(blocks: Sequence[str]) -> str:
    blocks = [block.strip("\n") for block in blocks]
    blocks = [block for block in blocks if block]
    return "\n\n".join(blocks)


def warn_forbidden_headers(text: str, min_allowed_header_level: int = 4) -> None:
    """
    Check each line. Warn if a header is too high (e.g., # or ##)
    compared to the minimum allowed header level.
    """

    for i, line in enumerate(text.splitlines()):
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
