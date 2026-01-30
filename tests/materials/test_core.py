from pydantic import BaseModel
from pytoy_llm.materials.core import ModelSectionData 

class AModel(BaseModel):
    a: int

class BModel(BaseModel):
    b: str


def test_model_section_single_type():
    section = ModelSectionData(
        bundle_kind="TestBundle",
        description="single model test",
        instances=[AModel(a=1), AModel(a=2)],
    )

    dumped = section.compose_str()
    assert '"a":1' in dumped
    assert '"a":2' in dumped

def test_model_section_union_types():
    section = ModelSectionData(
        bundle_kind="TestBundle",
        description="union model test",
        instances=[AModel(a=1), BModel(b="x")],
    )

    dumped = section.compose_str()

    assert '"a":1' in dumped
    assert '"b":"x"' in dumped

def test_model_section_empty_data():
    section = ModelSectionData(
        bundle_kind="TestBundle",
        description="empty test",
        instances=[],
    )

    dumped = section.compose_str()
    assert "No" in dumped

def test_compose_str_contains_sections():
    section = ModelSectionData(
        bundle_kind="TestBundle",
        description="compose test",
        instances=[AModel(a=1)],
    )

    text = section.compose_str()

    assert "### Description" in text
    assert "### Json Schemas" in text
    assert "### Json Instance" in text
    assert '"a":1' in text
