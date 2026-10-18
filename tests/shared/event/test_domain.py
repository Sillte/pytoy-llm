from threading import Thread

from pytoy_llm.shared.event import EventEmitter


def test_event_emitter_supports_concurrent_subscribe_fire_and_dispose() -> None:
    emitter = EventEmitter[int]()
    errors: list[Exception] = []

    def emit_and_dispose() -> None:
        try:
            for value in range(1000):
                subscription = emitter.event.subscribe(lambda _: None)
                emitter.fire(value)
                subscription.dispose()
        except Exception as exc:
            errors.append(exc)

    workers = [Thread(target=emit_and_dispose) for _ in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert errors == []


def test_event_emitter_listeners_can_change_subscriptions_during_fire() -> None:
    emitter = EventEmitter[int]()

    def listener(_value: int) -> None:
        subscription = emitter.event.subscribe(lambda _: None)
        subscription.dispose()

    emitter.event.subscribe(listener)

    emitter.fire(1)
