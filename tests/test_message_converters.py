import pytest
from litellm import ModelResponse
from pydantic import BaseModel
from pytoy_llm.models import InputMessage
from pytoy_llm.litellm_client import ModelResponseConverter


# --- Mock Data ---
@pytest.fixture
def mock_response():
    # LiteLLMのModelResponseを模したデータ
    response_data = {
        "choices": [
            {
                "message": {"role": "assistant", "content": '{"answer": "fine"}'},
                "finish_reason": "stop",
            }
        ],
        "model": "gemini/gemini-2.0-flash",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    return ModelResponse(**response_data)


class DummyStructuredModel(BaseModel):
    answer: str


# --- InputConverter Tests ---


def test_input_converter_str():
    res = InputMessage.from_any("hello")
    assert res.role == "user"
    assert res.content == "hello"


def test_input_converter_json_str():
    # JSON文字列が正しくパースされるか
    json_input = '{"role": "system", "content": "you are vim"}'
    res = InputMessage.from_any(json_input)
    assert res.role == "system"
    assert res.content == "you are vim"


def test_input_converter_sequence():
    # リストで渡して一括変換できるか
    msgs = [InputMessage.from_any(item) for item in ["hi", {"role": "assistant", "content": "ho"}]]
    assert len(msgs) == 2
    assert msgs[0].role == "user"
    assert msgs[1].role == "assistant"


def test_model_response_converter_to_str(mock_response):
    converter = ModelResponseConverter(str, str)
    result = converter.convert(mock_response, input_messages=[])
    assert isinstance(result, str)

def test_model_response_converter_to_basemodel(mock_response):
    converter = ModelResponseConverter(DummyStructuredModel, DummyStructuredModel)
    result = converter.convert(mock_response, input_messages=[])
    assert result.answer == "fine"