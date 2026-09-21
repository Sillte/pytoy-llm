from pytoy_llm.activity_sinks import PrintActivitySink
from pytoy_llm.llm_facade import LLMFacade
from pytoy_llm.models import LLMParam

if __name__ == "__main__":
    param = LLMParam(reasoning_effort="low")

    result = LLMFacade(activity_sink=PrintActivitySink(), llm_param=param).completion(
        "Reply with a short greeting.",
        output_type=str,
    )

    assert isinstance(result, str)
    assert result.strip()
