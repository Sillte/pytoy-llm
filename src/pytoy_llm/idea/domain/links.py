from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol, Self

from .uri import Uri


@dataclass(frozen=True, order=True)
class TextPosition:
    """
    0-based index inside the text.
    """

    line: int
    col: int


@dataclass(frozen=True)
class TextRange:
    start: TextPosition
    end: TextPosition


@dataclass(frozen=True)
class MarkdownLinkSource:
    text_range: TextRange
    target: Uri
    caption: str | None = None
    fragment: str | None = None

    @property
    def start(self) -> TextPosition:
        return self.text_range.start


type LinkSource = MarkdownLinkSource


@dataclass(frozen=True)
class LineLocation:
    line: int = 0
    col: int = 0

    @classmethod
    def from_any(cls, line: int, col: int = 0) -> Self:
        return cls(line=line, col=col)


@dataclass(frozen=True)
class AnchorLocation:
    anchor: str


type Location = LineLocation | AnchorLocation


@dataclass(frozen=True)
class ResolvedLocalLink:
    target_path: Path
    target_location: Location
    link_source: LinkSource
    source_path: Path

    @classmethod
    def from_any(
        cls,
        target_path: Path,
        target_location: Location,
        link_source: LinkSource,
        source_path: Path,
    ) -> Self:
        target_path = target_path.resolve()
        source_path = source_path.resolve()
        return cls(
            target_path=target_path,
            target_location=target_location,
            link_source=link_source,
            source_path=source_path,
        )

    @property
    def target_uri(self) -> str:
        return self.target_path.resolve().as_uri()


@dataclass(frozen=True)
class ResolvedRemoteLink:
    uri: Uri
    link_source: LinkSource
    source_path: Path

    @classmethod
    def from_any(
        cls,
        uri: Uri,
        link_source: LinkSource,
        source_path: Path,
    ) -> Self:
        return cls(uri=uri, link_source=link_source, source_path=source_path)

    @property
    def target_uri(self) -> Uri:
        return self.uri


type ResolvedLink = ResolvedLocalLink | ResolvedRemoteLink


@dataclass(frozen=True)
class UnresolvedLink:
    link_source: LinkSource
    source_path: Path
    reason: str = "Unknown"


class LinkSourceExtractor(Protocol):
    def extract(self, text: str) -> Iterator[LinkSource]: ...


@dataclass(frozen=True)
class IdeaLink:
    source_path: Path
    source_text_range: TextRange
    uri: Uri
    target_location: Location | None = None
    target_path: Path | None = None


@dataclass(frozen=True)
class UnresolvedIdeaLink:
    source_path: Path
    source_text_range: TextRange
    reason: str
