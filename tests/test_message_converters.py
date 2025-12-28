import pytest
from litellm import ModelResponse
from pydantic import BaseModel
from pytoy_llm.models import LLMOutputModel, SyncOutputFormat
from pytoy_llm.converters import InputConverter, OutputConverter


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
    conv = InputConverter()
    # 単なる文字列が user ロールになるか
    res = conv.to_llm_message("hello")
    assert res.role == "user"
    assert res.content == "hello"


def test_input_converter_json_str():
    conv = InputConverter()
    # JSON文字列が正しくパースされるか
    json_input = '{"role": "system", "content": "you are vim"}'
    res = conv.to_llm_message(json_input)
    assert res.role == "system"
    assert res.content == "you are vim"


def test_input_converter_sequence():
    conv = InputConverter()
    # リストで渡して一括変換できるか
    msgs = conv.to_llm_messages(["hi", {"role": "assistant", "content": "ho"}])
    assert len(msgs) == 2
    assert msgs[0].role == "user"
    assert msgs[1].role == "assistant"


# --- OutputConverter Tests ---


def test_output_converter_str(mock_response):
    conv = OutputConverter()
    # モード "str" で文字列が返るか
    output_format = SyncOutputFormat.from_any("str")
    res = conv.to_output(mock_response, output_format)
    assert res == '{"answer": "fine"}'


def test_output_converter_all(mock_response):
    conv = OutputConverter()
    # モード "all" で ModelResponse がそのまま返るか
    output_format = SyncOutputFormat.from_any("all")
    res = conv.to_output(mock_response, output_format)
    assert isinstance(res, ModelResponse)
    assert res.model == "gemini/gemini-2.0-flash"


def test_output_converter_custom_model(mock_response):
    conv = OutputConverter()
    # LLMOutputModel (CustomLLMOutputModel継承) への変換
    output_format = SyncOutputFormat.from_any(LLMOutputModel)
    res = conv.to_output(mock_response, output_format)
    assert isinstance(res, LLMOutputModel)
    assert res.content == '{"answer": "fine"}'
    assert res.usage.total_tokens == 15


def test_output_converter_structured_json(mock_response):
    conv = OutputConverter()
    # 任意のBaseModelへのパース (Structured Output)
    output_format = SyncOutputFormat.from_any(DummyStructuredModel)
    res = conv.to_output(mock_response, output_format)
    assert isinstance(res, DummyStructuredModel)
    assert res.answer == "fine"
