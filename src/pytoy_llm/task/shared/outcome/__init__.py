from dataclasses import dataclass
from typing import Literal, Protocol, overload, reveal_type, runtime_checkable

from typing_extensions import TypeIs

SuccessLiteral = Literal["Success"]
ErrorLiteral = Literal["Error"]


@runtime_checkable
class SuccessLike[T](Protocol):
    @property
    def kind(self) -> SuccessLiteral: ...
    @property
    def value(self) -> T: ...


@runtime_checkable
class ErrorLike[E](Protocol):
    @property
    def kind(self) -> ErrorLiteral: ...
    @property
    def exception(self) -> E: ...


@dataclass(frozen=True)
class Success[T]:
    value: T

    @property
    def kind(self) -> SuccessLiteral:
        return "Success"


@dataclass(frozen=True)
class Error[E]:
    exception: E

    @property
    def kind(self) -> ErrorLiteral:
        return "Error"


type Outcome[T, E] = Success[T] | Error[E]
type OutcomeLike[T, E] = SuccessLike[T] | ErrorLike[E]


@overload
def is_success[T, E](
    outcome: Outcome[T, E],
) -> TypeIs[Success[T]]: ...


@overload
def is_success[T, E](
    outcome: OutcomeLike[T, E],
) -> TypeIs[SuccessLike[T]]: ...


def is_success[T, E](outcome: OutcomeLike[T, E]) -> bool:
    return isinstance(outcome, SuccessLike)


@overload
def is_error[T, E](
    outcome: Outcome[T, E],
) -> TypeIs[Error[E]]: ...


@overload
def is_error[T, E](
    outcome: OutcomeLike[T, E],
) -> TypeIs[ErrorLike[E]]: ...


def is_error[T, E](
    outcome: OutcomeLike[T, E],
) -> bool:
    return isinstance(outcome, ErrorLike)
