from pytoy_llm.models import CancelToken
from pytoy_llm.task.models import CancelToken as TaskCancelToken


def test_cancel_token_is_exported_from_models_and_reused_by_task() -> None:
    token = CancelToken()

    assert TaskCancelToken is CancelToken
    assert not token.is_set()

    token.set()

    assert token.is_set()
