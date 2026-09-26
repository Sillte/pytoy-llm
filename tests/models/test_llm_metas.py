from pytoy_llm.models import LLMTokens


def test_llm_tokens_aggregate_sums_token_counts() -> None:
    result = LLMTokens.aggregate(
        [
            LLMTokens(prompt=10, completion=3, total=13),
            LLMTokens(prompt=4, completion=2, total=6),
        ]
    )

    assert result == LLMTokens(prompt=14, completion=5, total=19)


def test_llm_tokens_aggregate_returns_zero_for_empty_iterable() -> None:
    assert LLMTokens.aggregate(iter(())) == LLMTokens(
        prompt=0, completion=0, total=0, cache_read=0, cache_write=0
    )
