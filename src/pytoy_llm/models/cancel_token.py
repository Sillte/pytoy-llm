from threading import Event


class CancelToken:
    """Thread-safe, one-way signal that a task or LLM execution should stop at a checkpoint."""

    def __init__(self) -> None:
        self._event = Event()

    def set(self) -> None:
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()
