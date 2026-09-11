from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol, Self

from .uri import path_from_file_uri


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
    target: str
    caption: str | None = None
    fragment: str | None = None

    @property
    def start(self) -> TextPosition:
        return self.text_range.start


@dataclass(frozen=True)
class WikiLinkSource:
    text_range: TextRange
    target: str
    caption: str | None = None
    fragment: str | None = None

    @property
    def start(self) -> TextPosition:
        return self.text_range.start


type LinkSource = MarkdownLinkSource | WikiLinkSource


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
    file_path: Path
    location: Location
    link_source: LinkSource
    source_path: Path

    @classmethod
    def from_any(
        cls,
        file_path: Path,
        location: Location,
        link_source: LinkSource,
        source_path: Path,
        *,
        root: Path,
    ) -> Self:
        file_path = file_path.resolve()
        root = root.resolve()
        source_path = source_path.resolve()
        if file_path.is_relative_to(root):
            return cls(
                file_path=file_path,
                location=location,
                link_source=link_source,
                source_path=source_path,
            )
        raise ValueError(f"Given `{file_path=}` is outside of `{root=}`")

    @property
    def target_uri(self) -> str:
        return self.file_path.resolve().as_uri()


@dataclass(frozen=True)
class ResolvedRemoteLink:
    url: str
    link_source: LinkSource
    source_path: Path

    @classmethod
    def from_any(
        cls,
        url: str,
        link_source: LinkSource,
        source_path: Path,
    ) -> Self:
        return cls(url=url, link_source=link_source, source_path=source_path)

    @property
    def target_uri(self) -> str:
        return self.url


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
    uri: str
    target_location: Location | None = None

    @property
    def target_file_path(self) -> Path | None:
        return path_from_file_uri(self.uri)


@dataclass(frozen=True)
class UnresolvedIdeaLink:
    source_path: Path
    source_text_range: TextRange
    reason: str
