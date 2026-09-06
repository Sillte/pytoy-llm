from __future__ import annotations

from typing import Any, Callable, Protocol, TypeGuard, overload

from typing_extensions import TypeIs

type Listener[T] = Callable[[T], Any]
type Dispose = Callable[[], None]


class Disposable:
    def __init__(self, dispose: Dispose):
        self._dispose = dispose

    def dispose(self) -> None:
        self._dispose()


type Subscribe[T] = Callable[[Listener[T]], "Disposable"]


class EventProtocol[T](Protocol):
    def subscribe(self, listener: Listener[T]) -> Disposable: ...

    def once(self) -> "EventProtocol[T]": ...

    def map[R](self, transform: Callable[[T], R]) -> "EventProtocol[R]": ...

    @overload
    def filter[R](
        self,
        predicate: Callable[[T], TypeIs[R]],
    ) -> "EventProtocol[R]": ...

    @overload
    def filter[R](
        self,
        predicate: Callable[[T], TypeGuard[R]],
    ) -> "EventProtocol[R]": ...

    @overload
    def filter(
        self,
        predicate: Callable[[T], bool],
    ) -> "EventProtocol[T]": ...

    def filter(self, predicate: Callable[[T], bool]) -> "EventProtocol[Any]": ...


class Event[T]:
    def __init__(self, subscribe: Subscribe[T]):
        self._subscribe = subscribe

    def subscribe(self, listener: Listener[T]) -> Disposable:
        return self._subscribe(listener)

    def __call__(self, listener: Listener[T]) -> Disposable:
        # For decorator.
        return self.subscribe(listener)

    def once(self) -> Event:
        from pytoy_llm.task.shared.event import utils

        return utils.once(self)

    def map[R](self, transform: Callable[[T], R]) -> Event[R]:
        from pytoy_llm.task.shared.event import utils

        return utils.map_event(self, transform)

    @overload
    def filter[R](
        self,
        predicate: Callable[[T], TypeIs[R]],
    ) -> Event[R]: ...

    @overload
    def filter[R](
        self,
        predicate: Callable[[T], TypeGuard[R]],
    ) -> Event[R]: ...

    @overload
    def filter(
        self,
        predicate: Callable[[T], bool],
    ) -> Event[T]: ...

    def filter(self, predicate: Callable[[T], bool]) -> Event[Any]:
        from pytoy_llm.task.shared.event import utils

        return utils.filter(self, predicate)


class EventEmitter[T]:
    def __init__(self) -> None:
        self._listeners: list[Listener[T]] = []
        self.event = Event[T](self._subscribe)

    def _subscribe(self, listener: Listener[T]) -> Disposable:
        self._listeners.append(listener)

        def dispose():
            # For idempotency,
            try:
                self._listeners.remove(listener)
            except (ValueError, RuntimeError):
                pass

        return Disposable(dispose)

    def fire(self, value: T) -> None:
        for listener in list(self._listeners):
            listener(value)

    def dispose(self) -> None:
        self._listeners.clear()
