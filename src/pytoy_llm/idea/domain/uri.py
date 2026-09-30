from dataclasses import dataclass
from typing import Self
from urllib.parse import urlsplit, urlunsplit


@dataclass(frozen=True)
class Uri:
    scheme: str
    authority: str
    path: str
    query: str | None = None
    fragment: str | None = None

    @classmethod
    def from_any(cls, arg: str | Self) -> Self:
        if not isinstance(arg, str):
            return arg

        result = urlsplit(arg)

        return cls(
            scheme=result.scheme,
            authority=result.netloc,
            path=result.path,
            query=result.query or None,
            fragment=result.fragment or None,
        )

    def __str__(self) -> str:
        return urlunsplit(
            (self.scheme, self.authority, self.path, self.query or "", self.fragment or "")
        )
