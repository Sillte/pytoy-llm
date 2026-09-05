import json

import pytest
from pydantic import ValidationError

from pytoy_llm.connection_configuration import ConnectionConfiguration, _make_default_connection_dict
from pytoy_llm.models.connections import Connection


@pytest.fixture
def client():
    return ConnectionConfiguration()


@pytest.fixture
def test_name():
    return "__pytest_connection__"


def test_initialize_connection_file(client: ConnectionConfiguration, test_name: str):
    path = client.initialize_connection_file(test_name)
    assert path.exists()

    content = json.loads(path.read_text())
    assert "api_key" in content
    assert content["api_key"] == "SECRET"  # 初期状態は空文字（モデルの初期化とは別）


def test_get_connection_success(client, test_name):
    """Whether `Example` can be regarded as success."""
    path = client.get_connection_path(test_name)
    valid_data = _make_default_connection_dict(kind="local")
    path.write_text(json.dumps(valid_data))

    conn = client.get_connection(test_name)
    assert isinstance(conn, Connection)


def test_llm_connection_validation_error():
    with pytest.raises(ValidationError):
        Connection(model="   ", base_url="url", api_key="key")
