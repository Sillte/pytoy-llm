from pytoy_llm.activity_sinks import PrintActivitySink
from pytoy_llm.llm_facade import LLMFacade

if __name__ == "__main__":
    result = LLMFacade(activity_sink=PrintActivitySink()).completion(
        "Reply with a short greeting.",
        output_type=str,
    )

    assert isinstance(result, str)
    assert result.strip()
