from threading import Event as ThreadEvent
from threading import Lock

from pytoy_llm.shared.event import Event, EventEmitter


class CancelToken:
    """Thread-safe, one-way signal that a task or LLM execution should stop at a checkpoint."""

    def __init__(self) -> None:
        self._event = ThreadEvent()
        self._lock = Lock()
        self._on_cancel_emitter = EventEmitter[None]()

    @property
    def on_cancel(self) -> Event[None]:
        """Event fired on cancellation; late subscribers should also check `is_set()`."""
        return self._on_cancel_emitter.event

    def set(self) -> None:
        with self._lock:
            if self._event.is_set():
                return
            self._event.set()
        self._on_cancel_emitter.fire(None)

    def is_set(self) -> bool:
        return self._event.is_set()
