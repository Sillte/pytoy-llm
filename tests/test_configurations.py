import pytest
import json
from typing import Any
from pydantic import ValidationError, BaseModel
from pytoy_llm.configurations import ConfigurationClient
from pytoy_llm.models import Connection


def generate_expected_data_from_examples(klass: type[BaseModel]) -> dict[str, Any]:
    """Generate the data from `examles` attributes of `BaseModel`."""
    expected = {}
    for field_name, field_info in klass.model_fields.items():
        if field_info.examples and len(field_info.examples) > 0:
            expected[field_name] = field_info.examples[0]
        else:
            msg = f"`{klass}` does not have apt examples for `{field_name=}`"
            raise RuntimeError(msg)
    return expected


@pytest.fixture
def client():
    return ConfigurationClient()


@pytest.fixture
def test_name():
    return "__pytest_connection__"


def test_initialize_connection_file(client: ConfigurationClient, test_name: str):
    path = client.initialize_connection_file(test_name)
    assert path.exists()

    content = json.loads(path.read_text())
    assert "api_key" in content
    assert content["api_key"] == ""  # 初期状態は空文字（モデルの初期化とは別）


def test_get_connection_success(client, test_name):
    """Whether `Example` can be regarded as success."""
    path = client.get_connection_path(test_name)
    valid_data = generate_expected_data_from_examples(Connection)
    path.write_text(json.dumps(valid_data))

    conn = client.get_connection(test_name)
    assert isinstance(conn, Connection)


def test_llm_connection_validation_error():
    with pytest.raises(ValidationError):
        Connection(model="   ", base_url="url", api_key="key")
