from pydantic import BaseModel

from pytoy_llm.materials.models import ModelMaterialData


class AModel(BaseModel):
    a: int


class BModel(BaseModel):
    b: str


def test_model_section_single_type():
    section = ModelMaterialData(
        description="single model test",
        instances=[AModel(a=1), AModel(a=2)],
    )

    dumped = section.compose_explanation(1)
    assert '"a":1' in dumped
    assert '"a":2' in dumped


def test_model_section_union_types():
    section = ModelMaterialData(
        description="union model test",
        instances=[AModel(a=1), BModel(b="x")],
    )

    dumped = section.compose_explanation(3)

    assert '"a":1' in dumped
    assert '"b":"x"' in dumped


def test_model_section_empty_data():
    section = ModelMaterialData(
        description="empty test",
        instances=[],
    )

    dumped = section.compose_explanation(3)
    assert "No" in dumped


def test_compose_str_contains_sections():
    section = ModelMaterialData(
        description="compose test",
        instances=[AModel(a=1)],
    )

    text = section.compose_explanation(3 - 1)

    assert "### Description" in text
    assert "### JSON Schemas" in text
    assert "### JSON Instance" in text
    assert '"a":1' in text
