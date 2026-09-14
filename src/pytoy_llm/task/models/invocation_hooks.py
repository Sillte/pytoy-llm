import inspect
from dataclasses import dataclass
from typing import Callable, cast

from .context import ExecutionContext
from .invocation_results import InvocationResult


def _normalize[T](
    arg: Callable[[T], None] | Callable[[], None] | None,
) -> Callable[[T], None] | None:
    if arg is None:
        return arg

    if not callable(arg):
        raise TypeError(f"{arg} is not callable")

    params = list(inspect.signature(arg).parameters.values())
    if len(params) == 1:
        return cast(Callable[[T], None], arg)
    elif len(params) == 0:

        def wrapped(_: T) -> None:
            return arg()  # type: ignore
    else:
        raise ValueError(f"Hook must accept either zero or one argument: {len(params)=}")

    return cast(Callable[[T], None], wrapped)


@dataclass(frozen=True)
class InvocationHooks[T]:
    on_start: Callable[[ExecutionContext], None] | None = None
    on_result: Callable[[InvocationResult[T]], None] | None = None
    on_exception: Callable[[Exception], None] | None = None

    @classmethod
    def from_any(
        cls,
        on_start: Callable[[ExecutionContext], None] | Callable[[], None] | None = None,
        on_completion: Callable[[Exception | InvocationResult[T]], None]
        | Callable[[], None]
        | None = None,
        on_result: Callable[[InvocationResult[T]], None] | Callable[[], None] | None = None,
        on_exception: Callable[[Exception], None] | Callable[[], None] | None = None,
    ):
        if on_completion is not None:
            if on_result is not None or on_exception is not None:
                raise ValueError(
                    "When `on_completion` is set, `on_result` and `on_exception` must be None."
                )
            return cls(
                on_start=_normalize(on_start),
                on_result=_normalize(on_completion),
                on_exception=_normalize(on_completion),
            )
        return cls(
            on_start=_normalize(on_start),
            on_result=_normalize(on_result),
            on_exception=_normalize(on_exception),
        )
