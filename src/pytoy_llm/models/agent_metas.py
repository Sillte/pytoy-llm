from typing import Self

from pydantic import BaseModel


class UsageLimit(BaseModel, frozen=True):
    max_total_tokens: int | None = None
    max_requests: int | None = None

    def merge(self, other: Self) -> Self:
        """Combine limits by keeping the smaller non-None value for each cap."""
        return type(self)(
            max_total_tokens=_strictest_limit(self.max_total_tokens, other.max_total_tokens),
            max_requests=_strictest_limit(self.max_requests, other.max_requests),
        )


def _strictest_limit(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)
