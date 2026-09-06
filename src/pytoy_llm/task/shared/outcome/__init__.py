from dataclasses import dataclass

from typing_extensions import TypeIs


@dataclass(frozen=True)
class Success[T]:
    value: T


@dataclass(frozen=True)
class Error[E]:
    exception: E


type Outcome[T, E] = Success[T] | Error[E]


def is_success[T, E](outcome: Outcome[T, E]) -> TypeIs[Success[T]]:
    return isinstance(outcome, Success)


def is_error[T, E](outcome: Outcome[T, E]) -> TypeIs[Error[E]]:
    return isinstance(outcome, Error)
