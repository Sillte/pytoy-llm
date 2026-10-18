from threading import Thread

from pytoy_llm.models import CancelToken
from pytoy_llm.task.models import CancelToken as TaskCancelToken


def test_cancel_token_is_exported_from_models_and_reused_by_task() -> None:
    token = CancelToken()

    assert TaskCancelToken is CancelToken
    assert not token.is_set()

    token.set()

    assert token.is_set()


def test_cancel_token_fires_on_cancel_event_once() -> None:
    token = CancelToken()
    notifications: list[None] = []
    token.on_cancel.subscribe(notifications.append)

    workers = [Thread(target=token.set) for _ in range(8)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert notifications == [None]
