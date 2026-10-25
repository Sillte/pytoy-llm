import json
from pathlib import Path

from pydantic import BaseModel

from pytoy_llm.materials.models import JsonMaterialData


class AModel(BaseModel):
    a: int


class BModel(BaseModel):
    b: str


class PathModel(BaseModel):
    path: Path


def test_model_section_single_type():
    section = JsonMaterialData.from_basemodel(
        description="single model test",
        instances=[AModel(a=1), AModel(a=2)],
    )

    dumped = section.compose_explanation(1)
    schema_text = json.dumps(section.json_schema)
    assert '"type": "array"' in schema_text
    assert section.data == [{"a": 1}, {"a": 2}]
    assert '"a": 1' in dumped
    assert '"a": 2' in dumped


def test_model_section_union_types():
    section = JsonMaterialData.from_basemodel(
        description="union model test",
        instances=[AModel(a=1), BModel(b="x")],
    )

    dumped = section.compose_explanation(3)

    schema_text = json.dumps(section.json_schema)
    assert schema_text.count('"$ref"') == 2
    assert '"anyOf"' in schema_text
    assert section.data == [{"a": 1}, {"b": "x"}]
    assert '"a": 1' in dumped
    assert '"b": "x"' in dumped


def test_model_section_empty_data():
    section = JsonMaterialData.from_basemodel(
        description="empty test",
        instances=[],
        schema_model=AModel,
    )

    dumped = section.compose_explanation(3)
    assert section.data == []
    schema_text = json.dumps(section.json_schema)
    assert '"a": {"title": "A", "type": "integer"}' in schema_text
    assert "[]" in dumped


def test_model_section_empty_data_without_schema_model():
    section = JsonMaterialData.from_basemodel(
        description="empty test",
        instances=[],
    )

    assert json.dumps(section.json_schema) == '{"type": "array", "items": {}}'


def test_basemodel_data_is_converted_to_json_compatible_values():
    section = JsonMaterialData.from_basemodel(
        description="Path model test",
        instances=[PathModel(path=Path("folder/file.txt"))],
    )

    assert section.data == [{"path": str(Path("folder/file.txt"))}]


def test_compose_str_contains_sections():
    section = JsonMaterialData.from_basemodel(
        description="compose test",
        instances=[AModel(a=1)],
    )

    text = section.compose_explanation(3 - 1)

    assert "### Description" in text
    assert "### JSON Schema" in text
    assert "### JSON Instance" in text
    assert '"a": 1' in text
